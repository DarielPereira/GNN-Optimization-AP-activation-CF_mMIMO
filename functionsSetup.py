import numpy as np
import numpy.linalg as alg
import math
from sklearn.cluster import KMeans
import os

# Set the environment variable to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'

from functionsUtils import db2pow, localScatteringR, drawingSetup, drawing3Dvectors
# from functionsClustering_PilotAlloc import pilotAssignment, drawPilotAssignment


def generateSetup(L, K, N, T, cell_side, ASD_varphi, bool_testing,
    seed = 0):
    """Generates realizations of the setup
    INPUT>
    :param L: number of APs
    :param K: Number of UEs in the network
    :param N: Number of antennas per AP
    :param Q:
    :param cell_side: Cell side length
    :param ASD_varphi: Angular standard deviation in the local scattering model
                       for the azimuth angle (in radians)
    :param seed: Seed number of pseudorandom number generator


    OUTPUT>
    gainOverNoisedB: matrix with dimensions LxK where element (l,k) is the channel gain
                            (normalized by noise variance) between AP l and UE k
    R: matrix with dimensions N x N x L x K where (:,:,l,k) is the spatial correlation
                            matrix between  AP l and UE k (normalized by noise variance)
    APpositions: matrix of dimensions Lx1 containing the APs' locations as complex numbers,
                        where the real part is the horizontal position and the imaginary part is the
                        vertical position
    UEpositions: matrix of dimensions Lx1 containing the UEs' locations as complex numbers,
                        where the real part is the horizontal position and the imaginary part is the
                        vertical position
    distances: matrix of dimensions LxK where element (l,k) is the distance en meters between
                      Ap l and UE k
    M: matrix with dimensions QxL where element (q,l) is '1' if AP l is connected to CPU q, and '0' otherwise
    """

    # Set the seed if in testing mode
    if bool_testing:
        np.random.seed(seed)

    # Simulation Setup Configuration Parameters
    squarelength = cell_side    # length of one side the coverage area in m

    B = 20*10**6                # communication bandwidth in Hz
    noiseFigure = 7             # noise figure in dB
    noiseVariancedBm = -174+10*np.log10(B) + noiseFigure        #noise power in dBm

    alpha = 36.7                # path loss parameters for the path loss model
    constantTerm = -30.5

    distanceVertical = 10       # height difference between the APs and the UEs in meters
    antennaSpacing = 0.5        # half-wavelength distance

    # Set the number of APs to be a square number (if not already)
    if L**0.5 % 1 != 0:
        L_ = math.ceil(L**0.5)**2
    else:
        L_ = L

    AP_spacing = squarelength/(L_**0.5)

    # Regular AP deployment
    APpositions = []
    for i in np.linspace(AP_spacing/2, squarelength - AP_spacing/2, int(L_**0.5)):
        for j in np.linspace(AP_spacing/2, squarelength - AP_spacing/2, int(L_**0.5)):
            APpositions.append(complex(i, j) + complex((squarelength/50)*np.random.randn(1)[0], (squarelength/50)*np.random.randn(1)[0]))
    APpositions = np.array(APpositions).reshape(-1, 1)

    # Remove additional APs
    idx_remove = np.random.choice(range(L_), L_-L, replace=False)
    APpositions = np.delete(APpositions, idx_remove, axis=0)

    # M matrix stores the connections between the APs and the CPUs
    nbrOfCPUs = math.ceil(L / T)
    M = np.zeros((nbrOfCPUs, L), dtype=int)

    # Clustering mode
    clusteringMode = 'Kbeams'

    if clusteringMode == 'sequential':
        for i in range(nbrOfCPUs):
            for j in range(L):
                if (j) // T == i:
                    M[i, j] = 1

    elif clusteringMode == 'Kbeams':
        # Convert complex numbers to 2D (real and imaginary parts)
        data = np.column_stack((APpositions.real, APpositions.imag))

        # Apply k-means clustering
        num_clusters = nbrOfCPUs  # Choose the number of clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
        kmeans.fit(data)

        # Cluster centers and labels
        labels = kmeans.labels_
        for l in range(L):
            M[labels[l], l] = 1

    # To save the results
    gainOverNoisedB = np.zeros((L, K))
    R = np.zeros((N, N, L, K), dtype=complex)
    distances = np.zeros((L, K))
    UEpositions = np.zeros((K, 1), dtype=complex)


    # Add UEs
    for k in range(K):
        # generate a random UE location with uniform distribution
        UEposition = (np.random.rand() + 1j*np.random.rand())*squarelength     # Uncomment when tests finish

        # compute distance from new UE to all the APs
        distances[:, k] = np.sqrt(distanceVertical**2+np.abs(APpositions-UEposition)**2)[:, 0]

        # Compute the channel gain divided by noise power
        gainOverNoisedB[:, k] = constantTerm - alpha * np.log10(distances[:, k]) - noiseVariancedBm

        # store the UE position
        UEpositions[k] = UEposition

    # # setup map
    # drawingSetup(UEpositions, APpositions, np.zeros((K), dtype=int), title="Setup Map", squarelength=squarelength)


    # Compute correlation matrices
    for k in range(K):
        # run over the APs
        for l in range(L):  # Go through all APs
            angletoUE_varphi = np.angle(UEpositions[k] - APpositions[l])
            angletoUE_theta = np.arcsin(distanceVertical / distances[l, k])

            # Generate the approximate spatial correlation matrix using the local scattering model by scaling
            # the normalized matrices with the channel gain
            R[:, :, l, k] = db2pow(gainOverNoisedB[l, k]) * localScatteringR(N, angletoUE_varphi, ASD_varphi,
                                                                             antennaSpacing)

    return gainOverNoisedB, distances, R, APpositions, UEpositions, M


def get_F_G_matrices(gainOverNoisedB, L, K, f):
    """Return the F matrix with dimensions LxK where element (l,k) is '1' if AP L is one of the F preferred APs of UE k
    INPUT>
    :param gainOverNoisedB: matrix with dimensions LxK where element (l,k) is the channel gain
                            (normalized by noise variance) between AP l and UE k
    :param L: number of APs
    :param K: number of UEs
    :param f: number of potential APs to be selected by each UE

    OUTPUT>
    F: matrix with dimensions LxK where element (l,k) is '1' if AP L is one of the F preferred APs of UE k
    G: matrix with dimensions LxL where element (l,l') is '1' if APs l and l' have some UE in common
    """

    # Initialize the F and G matrices
    F = np.zeros((L, K), dtype=int)
    G = np.zeros((L, L), dtype=int)


    # Go through all UEs
    for k in range(K):
        # Find the indices of the F largest elements
        indices = np.argpartition(gainOverNoisedB[:, k], -f)[-f:]

        # Set the corresponding elements in the F matrix to '1'
        F[indices, k] = 1

    # Compute the adjacency matrix for the only-AP graph
    G[:] = F@F.T - np.diag(np.diag(F@F.T))

    return F, G

