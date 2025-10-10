"""
This script generates the data for the plots of sum-SE when considering different number of UEs.
"""

import numpy as np
import math

from functionsAllocation import PilotAssignment, AP_OnOff_GlobalHeuristics
from functionsSetup import generateSetup
from functionsChannelEstimates import channelEstimates
from functionsUtils import save_results
import torch as th
from functionsUtils import db2pow, binary_combinations, pow2db
from functionsComputeSE_uplink import functionComputeSE_uplink
from functionsSetup import get_F_G_matrices
from functionsGraphHandling import bipartitegraph_generation
from functionsGraphHandling import GNN_CorrMat, GNN_Gains


##Setting Parameters
configuration = {
    'nbrOfSetups': 20,             # number of communication network setups
    'K': 100,                     # number of UEs
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

algorithms = ['GNN']   #, 'successive_local_ES'

nbrOfSetups = configuration['nbrOfSetups']
K = configuration['K']
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

configurations = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500]

results = {
    'GNN': np.zeros((len(configurations))),
}

for idx, configuration in enumerate(configurations):
    estimationErrorVariance = configuration

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
            print(f'Estimation error variance: {estimationErrorVariance}')
            print('Algorithm: ' + algorithm)

            # Get the gainOverNoisedB in power scale
            gainOverNoise = db2pow(gainOverNoisedB)

            # Generate the gainOverNoisedB values with estimation error
            gainOverNoise = gainOverNoise + estimationErrorVariance * np.random.randn(L, K)
            gainOverNoise = np.maximum(gainOverNoise, 0.00001)

            # Bring back to dB scale
            gainOverNoisedB = pow2db(gainOverNoise)

            # Get the F matrix with preferred APs for each UE
            F, G = get_F_G_matrices(gainOverNoisedB, L, K, f)

            # Store the graph information
            # Generate the list of edges in the graphs
            G_sameCPU = np.zeros((L, L), dtype=int)
            G_sameCPU_full = np.zeros((L, L), dtype=int)
            for c in range(M.shape[0]):
                G_sameCPU[np.where(M[c, :] == 1)[0], :] = G[np.where(M[c, :] == 1)[0], :] * M[c, :]
                G_sameCPU_full[np.where(M[c, :] == 1)[0], :] = M[c, :]

            G_sameCPU_full = G_sameCPU_full - np.identity(L)
            G_diffCPU = G - G_sameCPU

            G_sameCPU_graph = th.tensor(np.transpose(np.nonzero(G_sameCPU))).T
            G_sameCPU_fullgraph = th.tensor(np.transpose(np.nonzero(G_sameCPU_full))).T
            G_diffCPU_graph = th.tensor(np.transpose(np.nonzero(G_diffCPU))).T

            F_graph, UE_features = bipartitegraph_generation(F, R, gainOverNoisedB, GNN_mode)

            # Create the GNN
            match GNN_mode:
                case 'CorrMat':
                    GNN = GNN_CorrMat(UE_features.shape[1])
                case 'Gains':
                    GNN = GNN_Gains(UE_features.shape[1])
                case _:
                    raise ValueError('ERROR: GNN mode mismatching')

            GNN.load_model(f'./AP_TrainingData/' + GNN_mode +
                           '/Model_L_12_N_4_Q_2_T_4_f_5_taup_100_NbrSamp_20000_Epochs_7_SAGEConv_sum.pt')

            # Compute the prediction
            GNN_output = GNN(G_sameCPU_fullgraph, G_diffCPU_graph,
                             UE_features, F_graph, L)

            APs_probabilities = th.sigmoid(GNN_output).detach().numpy().flatten()

            # To store the best AP state
            best_APstate = np.zeros((L))

            for c in range(M.shape[0]):
                best_APstate[np.argsort((APs_probabilities * M[c, :]).flatten())[-Q:]] = 1

            # D vector common to all the UEs
            D = np.ones((L, K))

            # Compute SE for centralized and distributed uplink operations for the case when all APs serve all the UEs
            SE_MMSE, SE_P_RZF, SE_MR, SE_P_MMSE = functionComputeSE_uplink(Hhat, H, D, best_APstate, C, tau_c,
                                                                           tau_p,
                                                                           nbrOfRealizations, N, K, L, p)

            match comb_mode:
                case 'MMSE':
                    SE = SE_MMSE
                case 'P_RZF':
                    SE = SE_P_RZF
                case 'MR':
                    SE = SE_MR
                case 'P_MMSE':
                    SE = SE_P_MMSE
                case _:
                    print('ERROR: Combining mode mismatching')
                    SE = 0

            best_sum_SE = np.sum(SE)
            best_SEs = SE.flatten()


            results[algorithm][idx] += best_sum_SE/ nbrOfSetups


file_name = f'./GRAPHs/VARIABLES_SAVED/SE_EstimationErrors_NbrSetps_{nbrOfSetups}_L_100_N_4_AVERAGE_Kmeans.pkl'
save_results(results, file_name)