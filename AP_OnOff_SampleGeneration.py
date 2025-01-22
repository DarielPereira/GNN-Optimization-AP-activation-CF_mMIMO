### This script generates and stores the setups' buffer from where the training samples are later generated
import os

# Set the environment variable to avoid memory leak
os.environ['OMP_NUM_THREADS'] = '1'

import torch as th
import math
import random
import numpy as np
from datetime import datetime

from functionsAllocation import PilotAssignment, AP_OnOff_GlobalHeuristics
from functionsSetup import generateSetup, get_F_G_matrices
from functionsChannelEstimates import channelEstimates
from functionsGraphHandling import SampleBuffer, bipartitegraph_generation, DualGraphDataset
from torch_geometric.data import Data
from tqdm import tqdm

##Setting Parameters
configuration = {
    'nbrOfSetups': 20000,             # number of communication network setups
    'nbrOfConnectedUEs_range': [6, 10],            # number of UEs to insert
    'nbrOfRealizations': 5,      # number of channel realizations per sample
    'L': 12,                     # number of APs
    'N': 4,                       # number of antennas per AP
    'Q': 2,                       # max number of APs served by each CPU
    'T': 4,                       # number of APs connected to each CPU
    'tau_c': 400,                 # length of the coherence block
    'tau_p': 100,                  # length of the pilot sequences
    'p': 100,                     # uplink transmit power per UE in mW
    'cell_side': 400,            # side of the square cell in m
    'ASD_varphi': math.radians(10),         # Azimuth angle - Angular Standard Deviation in the local scattering model
    'comb_mode': 'MMSE',           # combining method used to evaluate optimization
    'potentialAPs_mode': 'F_highestChannelGain', # mode used to select the potential APs
    'f': 5,                        # number of potential APs to be selected by each UE
    'bool_testing': False,           # set to 'True' to enable testing mode
    'heuristic_mode': 'exhaustive_search'   # heuristic mode used to solve the optimization
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
bool_testing = configuration['bool_testing']
heuristic_mode = configuration['heuristic_mode']

# Create the sample storage buffer
setupsBuffer = SampleBuffer()

# Create a buffer to storage graph-related information
graphBuffer = SampleBuffer()

# Run over all the setups
for setup_iter in tqdm(range(nbrOfSetups), desc="Generating Setups", unit="setup"):

    # Set the seed if in testing mode
    if bool_testing:
        random.seed(setup_iter)

    # sample the number of connected UEs from a uniform distribution in the specified range (nbrOfConnectedUEs_range)
    K = random.randint(nbrOfConnectedUEs_range[0], nbrOfConnectedUEs_range[1])

    # print(f'Generating setup {setup_iter + 1}/{nbrOfSetups} with {K} connected UEs......')

    # Generate one setup with UEs and APs at random locations
    gainOverNoisedB, distances, R, APpositions, UEpositions, M = (
        generateSetup(L, K, N, T, cell_side, ASD_varphi, bool_testing=bool_testing, seed=setup_iter))

    # Get the F matrix with preferred APs for each UE
    F, G = get_F_G_matrices(gainOverNoisedB, L, K, f)

    # Compute AP and pilot assignment
    pilotIndex = PilotAssignment(R, gainOverNoisedB, tau_p, L, K, N, mode='DCC')

    # Generate channel realizations with estimates and estimation error matrices
    Hhat, H, B, C = channelEstimates(R, nbrOfRealizations, L, K, N, tau_p, pilotIndex, p)

    best_APstate, best_sum_SE, best_SEs = AP_OnOff_GlobalHeuristics(p, nbrOfRealizations, R, gainOverNoisedB, tau_p,
                                                                    tau_c, Hhat,
                                                                    H, B, C, L, K, N, Q, M, f,
                                                                    comb_mode, heuristic_mode)

    # Store the setup information
    setupsBuffer.add([R, F, G, best_APstate])

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

    F_graph, UE_features = bipartitegraph_generation(F, R, gainOverNoisedB)

    graphBuffer.add([G_sameCPU_graph, G_sameCPU_fullgraph, G_diffCPU_graph, F_graph, UE_features, best_APstate])

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# # Save the setup buffer
# file_name = (
# f'./AP_TRAININGDATA/newData/SetupBuffer_Comb_'
# +comb_mode+f'_L_{L}_N_{N}_Q_{Q}_T_{T}_f_{f}_taup_{tau_p}_NbrSamp_{len(setupsBuffer.storage)}_'+timestamp+'.pkl')
#
# setupsBuffer.save(file_name)

# Create the data set to store the graphs
dataset = DualGraphDataset(graphs = [])
dataset.parameters = configuration

# Load the information from the buffer to the dataset
for sample in graphBuffer.storage:
    G_graph = Data(sameCPU_edge=sample[0], full_sameCPU_edge=sample[1], diffCPU_edge=sample[2],
                   y=th.tensor(sample[5], dtype=th.float))
    F_graph = Data(edge_index=sample[3], x=sample[4])
    dataset.add_sample(G_graph, F_graph)

file_name = (
f'./AP_TrainingData/Dataset_L_{L}_N_{N}_Q_{Q}_T_{T}_f_{f}_taup_{tau_p}_NbrSamp_{len(graphBuffer.storage)}.pt')

# Save the data set
dataset.save(file_name)


