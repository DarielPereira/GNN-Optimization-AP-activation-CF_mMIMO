import math

from sympy.matrices.expressions.factorizations import QofQR

from functionsUtils import save_results
import numpy as np
import torch as th
import random

from functionsAllocation import PilotAssignment, AP_OnOff_GlobalHeuristics
from functionsSetup import generateSetup
from functionsChannelEstimates import channelEstimates



##Setting Parameters
configuration = {
    'nbrOfSetups': 10,             # number of communication network setups
    'nbrOfConnectedUEs_range': [1, 150],            # number of UEs to insert
    'nbrOfRealizations': 2,      # number of channel realizations per sample
    'L': 100,                     # number of APs
    'N': 4,                       # number of antennas per AP
    'Q': 4,                       # max number of APs served by each CPU
    'T': 8,                       # number of APs connected to each CPU
    'f': 1,                        # number of potential APs to be selected by each UE
    'tau_c': 400,                 # length of the coherence block
    'tau_p': 100,                  # length of the pilot sequences
    'p': 100,                     # uplink transmit power per UE in mW
    'cell_side': 1000,            # side of the square cell in m
    'ASD_varphi': math.radians(10),         # Azimuth angle - Angular Standard Deviation in the local scattering model
    'comb_mode': 'MMSE',           # combining method used to evaluate optimization
    'heuristic_mode': 'successive_local_SG',   # heuristic mode used to solve the optimization
                                            # ['exhaustive_search', 'sequential_greedy', 'best_individualAPs',
                                            # 'local_ES', 'local_SG', 'Q_random', 'successive_local_SG',
                                            # 'successive_local_ES', 'bestgains_individualAPs', 'GNN']
    'GNN_mode': 'Gains'           # mode used to generate the GNN input
                                            # ['CorrMat', 'Gains']
    }

print('### CONFIGURATION PARAMETERS ###')
for param in configuration:
    print(param+f': {configuration[param]}')
print('###  ###\n')

nbrOfSetups = configuration['nbrOfSetups']
nbrOfConnectedUEs_range = configuration['nbrOfConnectedUEs_range']
nbrOfRealizations = configuration['nbrOfRealizations']
L = configuration['L']
N = configuration['N']
Q = configuration['Q']
T = configuration['T']
f = configuration['f']
tau_c = configuration['tau_c']
tau_p = configuration['tau_p']
p = configuration['p']
cell_side = configuration['cell_side']
ASD_varphi = configuration['ASD_varphi']
comb_mode = configuration['comb_mode']
heuristic_mode = configuration['heuristic_mode']
GNN_mode = configuration['GNN_mode']

# To store the sum-SE values for each setup
sum_SEs = np.zeros(nbrOfSetups)

# To store the AP states for each setup
AP_states = []

# Run over all the setups
for setup_iter in range(nbrOfSetups):

    # sample the number of connected UEs from a uniform distribution in the specified range (nbrOfConnectedUEs_range)
    # random.seed(setup_iter+nbrOfSetups)
    # K = random.randint(nbrOfConnectedUEs_range[0], nbrOfConnectedUEs_range[1])
    K = 300

    print(f'Generating setup {setup_iter+1}/{nbrOfSetups} with {K} connected UEs......')

    # Generate one setup with UEs and APs at random locations
    gainOverNoisedB, distances, R, APpositions, UEpositions, M = (
        generateSetup(L, K, N, T, cell_side, ASD_varphi, bool_testing=True,  seed=setup_iter+100))

    # Compute AP and pilot assignment
    pilotIndex = PilotAssignment(R, gainOverNoisedB, tau_p, L, K, N, mode='DCC')

    # Generate channel realizations with estimates and estimation error matrices
    Hhat, H, B, C = channelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p)

    best_APstate, best_sum_SE, best_SEs = AP_OnOff_GlobalHeuristics(p, nbrOfRealizations, R, gainOverNoisedB, tau_p, tau_c, Hhat,
                                             H, B, C, L, K, N, Q, M, f,
                   comb_mode, heuristic_mode, GNN_mode)


    # Print the results
    print(f'Best AP state: {best_APstate}')
    print(f'Best sum SE: {best_sum_SE}')
    print(f'Best SEs: {best_SEs}')

    # Store the sum-SE values
    sum_SEs[setup_iter] = best_sum_SE

    # Store the AP states
    AP_states.append(best_APstate)





