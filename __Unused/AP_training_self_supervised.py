import os
import torch as th
import random
import numpy as np
import glob
import math
from functionsGraphHandling import GNN_CorrMat, GNN_Gains, bipartitegraph_generation
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from functionsAllocation import PilotAssignment
from functionsSetup import generateSetup, get_F_G_matrices
from functionsChannelEstimates import channelEstimates

##Setting Parameters
configuration = {
    'nbrOfSetups': 10,  # number of communication network setups
    'nbrOfConnectedUEs_range': [1, 150],  # number of UEs to insert
    'nbrOfRealizations': 2,  # number of channel realizations per sample
    'L': 100,  # number of APs
    'N': 4,  # number of antennas per AP
    'Q': 4,  # max number of APs served by each CPU
    'T': 8,  # number of APs connected to each CPU
    'tau_c': 400,  # length of the coherence block
    'tau_p': 100,  # length of the pilot sequences
    'p': 100,  # uplink transmit power per UE in mW
    'cell_side': 1000,  # side of the square cell in m
    'ASD_varphi': math.radians(10),  # Azimuth angle - Angular Standard Deviation in the local scattering model
    'comb_mode': 'MMSE',  # combining method used to evaluate optimization
    'potentialAPs_mode': 'F_highestChannelGain', # mode used to select the potential APs
    'f': 5,                        # number of potential APs to be selected by each UE
    'heuristic_mode': 'exhaustive_search',   # heuristic mode used to solve the optimization
    'GNN_mode': 'Gains',           # mode used to generate the GNN input ['CorrMat', 'Gains']
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
tau_c = configuration['tau_c']
tau_p = configuration['tau_p']
p = configuration['p']
cell_side = configuration['cell_side']
ASD_varphi = configuration['ASD_varphi']
comb_mode = configuration['comb_mode']
potentialAPs_mode = configuration['potentialAPs_mode']
f = configuration['f']
heuristic_mode = configuration['heuristic_mode']
GNN_mode = configuration['GNN_mode']

bool_testing = False
bool_load_model = False         # revise the file name in the next code block

if bool_testing:
    print('Testing mode enabled')
    print('Setting seed to 0')
    seed = 0
    th.manual_seed(seed)
    random.seed(seed)

# Model, optimizer, and loss
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
match GNN_mode:
    case 'CorrMat':
        input_size = N**2
        model = GNN_CorrMat(UE_feature_size=input_size).to(device)
    case 'Gains':
        model = GNN_Gains(UE_feature_size=1).to(device)
    case _:
        raise ValueError('Invalid GNN mode')

# Load the model if it exists
if bool_load_model:
    model_directory = f'Model_L_12_N_4_Q_2_T_4_f_5_taup_100_NbrSamp_20000_Epochs_7_SAGEConv_sum.pt'
    model.load_model(model_directory)

optimizer = th.optim.Adam(model.parameters(), lr=0.0001)

# Run over all the setups
for setup_iter in tqdm(range(nbrOfSetups), desc="Generating Setups", unit="setup"):

    # sample the number of connected UEs from a uniform distribution in the specified range (nbrOfConnectedUEs_range)
    K = random.randint(nbrOfConnectedUEs_range[0], nbrOfConnectedUEs_range[1])

    # print(f'Generating setup {setup_iter + 1}/{nbrOfSetups} with {K} connected UEs......')

    # Generate one setup with UEs and APs at random locations
    gainOverNoisedB, distances, R, APpositions, UEpositions, M = (
        generateSetup(L, K, N, T, cell_side, ASD_varphi, bool_testing=bool_testing, seed=setup_iter))

    # Compute AP and pilot assignment
    pilotIndex = PilotAssignment(R, gainOverNoisedB, tau_p, L, K, N, mode='DCC')

    # Generate channel realizations with estimates and estimation error matrices
    Hhat, H, B, C = channelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p)

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

    GNN_output = model(G_sameCPU_fullgraph, G_diffCPU_graph,
                                           UE_features, F_graph, L)

    APs_probabilities = th.sigmoid(GNN_output).flatten()

    ##From this point, the code is not complete. The next steps are to:
    # 1. Compute the best APs to activate
    # 2. Compute the sum SE for the best APs to activate as in the AP_OnOff_GlobalHeuristics function
    # 3. Use the negative of the sum SE as the loss function
    # 4. Update the model using the optimizer
    ##Why is incomplete:
    # 1. The objective function is not continuous or derivable with respect to the model parameters because of
    #    the combinatorial nature of the problem. This is a limitation for the self-supervised learning approach.



