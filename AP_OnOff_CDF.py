"""
This script generates the data for the CDF plots of the SE per UE when different AP On/Off algorithms are used.
"""

import numpy as np
import math

from functionsAllocation import PilotAssignment, AP_OnOff_GlobalHeuristics
from functionsSetup import generateSetup
from functionsChannelEstimates import channelEstimates
from functionsUtils import save_results

##Setting Parameters
configuration = {
    'nbrOfSetups': 10,             # number of communication network setups
    'K': 100,                    # number of UEs
    'nbrOfRealizations': 3,      # number of channel realizations per sample
    'Q': 3,                       # max number of APs served by each CPU
    'T': 6,                       # number of APs connected to each CPU
    'f': 1,                        # number of potential APs to be selected by each UE
    'tau_c': 200,                 # length of the coherence block
    'tau_p': 20,                  # length of the pilot sequences
    'p': 100,                     # uplink transmit power per UE in mW
    'cell_side': 1000,            # side of the square cell in m
    'ASD_varphi': math.radians(10),         # Azimuth angle - Angular Standard Deviation in the local scattering model
    'comb_mode': 'MMSE',           # combining method used to evaluate optimization
    'GNN_mode': 'Gains'
}

algorithms = ['bestgains_individualAPs', 'Q_random', 'GNN', 'successive_local_ES']

nbrOfSetups = configuration['nbrOfSetups']
K = configuration['K']
nbrOfRealizations = configuration['nbrOfRealizations']
Q = configuration['Q']
T = configuration['T']
f = configuration['f']
tau_c = configuration['tau_c']
tau_p = configuration['tau_p']
p = configuration['p']
cell_side = configuration['cell_side']
ASD_varphi = configuration['ASD_varphi']
comb_mode = configuration['comb_mode']
GNN_mode = configuration['GNN_mode']

configurations = [(100, 4)]

results = {
    'bestgains_individualAPs': np.zeros((len(configurations), K * nbrOfSetups)),
    'Q_random': np.zeros((len(configurations), K * nbrOfSetups)),
    'GNN': np.zeros((len(configurations), K * nbrOfSetups)),
    'successive_local_ES': np.zeros((len(configurations), K * nbrOfSetups)),
}

for idx, configuration in enumerate(configurations):
    L = configuration[0]
    N = configuration[1]

    # iterate over the setups
    for iter in range(nbrOfSetups):
        print("Setup iteration {} of {}".format(iter + 1, nbrOfSetups))

        # Generate one setup with UEs and APs at random locations
        gainOverNoisedB, distances, R, APpositions, UEpositions, M = (
            generateSetup(L, K, N, T, cell_side, ASD_varphi, bool_testing=True, seed=iter + 100))

        # Compute AP and pilot assignment
        pilotIndex = PilotAssignment(R, gainOverNoisedB, tau_p, L, K, N, mode='DCC')

        # Generate channel realizations with estimates and estimation error matrices
        Hhat, H, B, C = channelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p)

        for algorithm in algorithms:
            print(f'number of APs L: {L}')
            print(f'number of antennas N: {N}')
            print('Algorithm: ' + algorithm)

            best_APstate, best_sum_SE, best_SEs = AP_OnOff_GlobalHeuristics(p, nbrOfRealizations, R, gainOverNoisedB,
                                                                            tau_p, tau_c, Hhat,
                                                                            H, B, C, L, K, N, Q, M, f,
                                                                            comb_mode, algorithm, GNN_mode)


            results[algorithm][idx, iter * K:(iter+1) * K] = best_SEs[:]

# Saving the results    NOTE: /GRAPHs/VARIABLES_SAVED/ path must be created in case it does not exist
file_name = f'./GRAPHs/VARIABLES_SAVED/SE_CDF_K_{K}_NbrSetps_{nbrOfSetups}_L_100_N_4.pkl'
save_results(results, file_name)