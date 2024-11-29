import numpy as np

from functionsSetup import *
from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
import math

from functionsClustering import cfMIMO_clustering
from functionsPilotAlloc import pilotAssignment
from functionsChannelEstimates import channelEstimates
from functionsComputeSE_uplink import functionComputeSE_uplink
from functionsUtils import save_results, load_results


##Setting Parameters
configuration = {
    'nbrOfSetups': 10,             # number of Monte-Carlo setups
    'nbrOfRealizations': 3,       # number of channel realizations per setup
    'L': 400,                       # number of APs
    'N': 1,                     # number of antennas per AP
    'tau_c': 200,                 # length of the coherence block
    'tau_p': 10,                  # length of the pilot sequences
    'T': 1,                       # pilot reuse factor
    'p': 100,                     # uplink transmit power per UE in mW
    'ASD_varphi': math.radians(10),         # Azimuth angle - Angular Standard Deviation in the local scattering model
    'ASD_theta': math.radians(15),          # Elevation angle - Angular Standard Deviation in the local scattering model
}

algorithms = {
    'Kmeans': ['Kmeans_locations', 'bf_bAPs_iC'],
    'Kfootprints': ['Kfootprints', 'bf_bAPs_iC'],
    'DCC': ['no_clustering', 'DCC'],
    'DCPA': ['no_clustering', 'DCPA'],
    'balanced_random': ['no_clustering', 'balanced_random'],
    'random': ['no_clustering', 'random'],
}

nbrOfSetups = configuration['nbrOfSetups']
nbrOfRealizations = configuration['nbrOfRealizations']
L = configuration['L']
N = configuration['N']
tau_c = configuration['tau_c']
tau_p = configuration['tau_p']
T = configuration['T']
p = configuration['p']
ASD_varphi = configuration['ASD_varphi']
ASD_theta = configuration['ASD_theta']

setups = [50, 100, 150, 200, 250, 300, 350, 400]

results = {
    'Kmeans_locations': np.zeros((len(setups))),
    'Kfootprints': np.zeros((len(setups))),
    'DCC': np.zeros((len(setups))),
    'DCPA': np.zeros((len(setups))),
    'balanced_random': np.zeros((len(setups))),
    'random': np.zeros((len(setups))),
}

for idx, setup in enumerate(setups):
    K = setup

    for algorithm in algorithms:
        cl_mode = algorithms[algorithm][0]
        pa_mode = algorithms[algorithm][1]

        NMSEs = 0

        print(f'number of UEs K: {K}')
        print('Clustering mode: ' + cl_mode)
        print('Pilot allocation mode: ' + pa_mode)

        # iterate over the setups
        for iter in range(nbrOfSetups):
            print("Setup iteration {} of {}".format(iter+1, nbrOfSetups))

            # Generate one setup with UEs and APs at random locations
            gainOverNoisedB, distances, R, APpositions, UEpositions = (
                generateSetup(L, K, N, tau_p, ASD_varphi, ASD_theta, nbrOfRealizations, seed=iter))

            UE_clustering \
                = cfMIMO_clustering(gainOverNoisedB, R, tau_p, APpositions, UEpositions, mode=cl_mode)

            pilotIndex, D = pilotAssignment(UE_clustering, R, gainOverNoisedB, K, tau_p, L, N, mode=pa_mode)

            # Compute NMSE for all the UEs
            system_NMSE, UEs_NMSE, worst_userXpilot, best_userXpilot \
                = functionComputeNMSE_uplink(D, tau_p, N, K, L, R, pilotIndex)

            NMSEs += system_NMSE/nbrOfSetups
            print('System NMSE: {}'.format(system_NMSE))

        if cl_mode in results:
            results[cl_mode][idx] = NMSEs
        elif pa_mode in results:
            results[pa_mode][idx] = NMSEs

file_name = f'./GRAPHs/VARIABLES_SAVED/NMSE_L_{L}_N_{N}_NbrSetps_{nbrOfSetups}_50_400.pkl'
save_results(results, file_name)
