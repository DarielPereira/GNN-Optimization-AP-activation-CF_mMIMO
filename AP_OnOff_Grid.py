import numpy as np
import math
import matplotlib.pyplot as plt
import numpy.linalg as linalg

from functionsUtils import db2pow, localScatteringR
from functionsAllocation import AP_OnOff_GlobalHeuristics
from functionsSetup import generateSetup
from functionsChannelEstimates import channelEstimates
from functionsComputeSE_uplink import functionComputeSE_uplink



ASD_varphi = math.radians(10)
ASD_theta = math.radians(15)
antennaSpacing = 0.5
distanceVertical = 10
noiseFigure = 7
B = 20*10**6
noiseVariancedBm = -174+10*np.log10(B) + noiseFigure        #noise power in dBm
alpha = 36.7                # pathloss parameters for the path loss model
constantTerm = -30.5
sigma_sf = 4

p = 100
tau_p = 100
tau_c = 400
L = 32
T = 4  # number of APs connected to each CPU
Q = 2  # max number of APs served by each CPU
f = 5  # number of potential APs to be selected by each UE
N = 4
K = 1

cell_side = 200

pilotIndex = np.array([0], dtype=int)

nbrOfRealizations = 20

gainOverNoisedB, distances, R, APpositions, UEpositions, M = (
    generateSetup(L, K, N, T, cell_side, ASD_varphi, bool_testing=True))

Grid_SE_fixed = np.zeros((20, 20))
Grid_SE_pixelbased = np.zeros((20, 20))

for idxi, i in enumerate(range(0, 200, 10)):
    for idxj, j in enumerate(range(0, 200, 10)):
        print(i, j)

        UEpositions[0, 0] = complex(i, j)

        distances[:, 0] = np.sqrt(distanceVertical ** 2 + np.abs(APpositions - UEpositions[0, 0]) ** 2)[:, 0]
        gainOverNoisedB[:, 0] = constantTerm - alpha * np.log10(distances[:, 0]) - noiseVariancedBm + (np.sqrt(sigma_sf**2))*np.random.randn(L)

        for l in range(L):  # Go through all APs
            angletoUE_varphi = np.angle(UEpositions[0, 0] - APpositions[l])

            # Generate the approximate spatial correlation matrix using the local scattering model by scaling
            # the normalized matrices with the channel gain
            R[:, :, l, 0] = db2pow(gainOverNoisedB[l, 0]) * localScatteringR(N, angletoUE_varphi, ASD_varphi,
                                                                             antennaSpacing)

        Hhat, H, B, C = channelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p)


        best_APstate, best_sum_SE, best_SEs = AP_OnOff_GlobalHeuristics(p, nbrOfRealizations, R,
                                                                        gainOverNoisedB, tau_p,
                                                                        tau_c, Hhat,
                                                                        H, B, C, L, K, N, Q, M, f,
                                                                        comb_mode= 'MMSE', heuristic_mode= 'local_ES',
                                                                        GNN_mode= 'Gains')

        Grid_SE_pixelbased[idxj, idxi] = best_sum_SE


        best_APstate_fixed, best_sum_SE, best_SEs = AP_OnOff_GlobalHeuristics(p, nbrOfRealizations, R,
                                                                        gainOverNoisedB, tau_p,
                                                                        tau_c, Hhat,
                                                                        H, B, C, L, K, N, Q, M, f,
                                                                        comb_mode= 'MMSE', heuristic_mode = 'fixed',
                                                                        GNN_mode= 'Gains')

        Grid_SE_fixed[idxj, idxi] = best_sum_SE




x = np.arange(0, 200, 10)
y = np.arange(0, 200, 10)

# Determine the global min and max values for consistent scaling
vmin = min(Grid_SE_pixelbased.min(), Grid_SE_fixed.min())
vmax = max(Grid_SE_pixelbased.max(), Grid_SE_fixed.max())

# Plot for Grid_SE_pixelbased
fig, ax0 = plt.subplots()
im0 = plt.pcolormesh(x, y, Grid_SE_pixelbased[:-1, :-1], vmin=vmin, vmax=vmax)
ax0.set_title('dot product (Pixel-based)')
plt.scatter(APpositions.real, APpositions.imag, c='mediumblue', marker='^', s=8)
fig.colorbar(im0, ax=ax0)
plt.show()

np.savez(f'./GRAPHs/VARIABLES_SAVED/Grid_SE_pixelbased',
         grid_product=Grid_SE_pixelbased, AP_positions=APpositions)

# Plot for Grid_SE_fixed
fig, ax0 = plt.subplots()
im0 = plt.pcolormesh(x, y, Grid_SE_fixed[:-1, :-1], vmin=vmin, vmax=vmax)
ax0.set_title('dot product (Fixed)')
plt.scatter(APpositions.real, APpositions.imag, c='mediumblue', marker='^', s=8)
fig.colorbar(im0, ax=ax0)
plt.show()

np.savez(f'./GRAPHs/VARIABLES_SAVED/Grid_SE_fixed',
         grid_product=Grid_SE_fixed, AP_positions=APpositions, best_APstate_fixed = best_APstate_fixed)



print('end')