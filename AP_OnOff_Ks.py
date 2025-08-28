import numpy as np
import math

from functionsAllocation import PilotAssignment, AP_OnOff_GlobalHeuristics
from functionsSetup import generateSetup
from functionsChannelEstimates import channelEstimates
from functionsUtils import save_results

##Setting Parameters
configuration = {
    'nbrOfSetups': 20,             # number of communication network setups
    'L': 100,                     # number of APs
    'N': 4,                       # number of antennas per AP
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

algorithms = ['bestgains_individualAPs', 'Q_random', 'GNN', 'local_SG']   #, 'successive_local_ES'

nbrOfSetups = configuration['nbrOfSetups']
L = configuration['L']
N = configuration['N']
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

configurations = [50, 75, 100, 125, 150, 175, 200]

results = {
    'bestgains_individualAPs': np.zeros((len(configurations))),
    'Q_random': np.zeros((len(configurations))),
    'GNN': np.zeros((len(configurations))),
    'local_SG': np.zeros((len(configurations)))
}

for idx, configuration in enumerate(configurations):
    K = configuration

    # iterate over the setups
    for iter in range(nbrOfSetups):
        print("Setup iteration {} of {}".format(iter + 1, nbrOfSetups))

        # Generate one setup with UEs and APs at random locations
        gainOverNoisedB, distances, R, APpositions, UEpositions, M = (
            generateSetup(L, K, N, T, cell_side, ASD_varphi, bool_testing=True, seed=iter + 300))

        # Compute AP and pilot assignment
        pilotIndex = PilotAssignment(R, gainOverNoisedB, tau_p, L, K, N, mode='DCC')

        # Generate channel realizations with estimates and estimation error matrices
        Hhat, H, B, C = channelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p)

        for algorithm in algorithms:
            print(f'number of UEs: {K}')
            print('Algorithm: ' + algorithm)

            best_APstate, best_sum_SE, best_SEs = AP_OnOff_GlobalHeuristics(p, nbrOfRealizations, R, gainOverNoisedB,
                                                                            tau_p, tau_c, Hhat,
                                                                            H, B, C, L, K, N, Q, M, f,
                                                                            comb_mode, algorithm, GNN_mode)


            results[algorithm][idx] += best_sum_SE/ nbrOfSetups


file_name = f'./GRAPHs/VARIABLES_SAVED/SE_Ks_NbrSetps_{nbrOfSetups}_L_100_N_4_AVERAGE_Kmeans.pkl'
save_results(results, file_name)