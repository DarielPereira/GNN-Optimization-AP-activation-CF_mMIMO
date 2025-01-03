import torch
import torch as th
from collections import deque
import random
import numpy as np
import pickle

from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.nn import GCNConv, SAGEConv


class GNN_model(th.nn.Module):
    def __init__(self, UE_feature_size, L):
        super().__init__()
        self.L = L
        self.bipartite = SAGEConv(UE_feature_size, 8, root_weight=False)
        self.sameCPU = GCNConv(8, 8)
        self.diffCPU = GCNConv(8, 8, add_self_loops=False)
        self.linear = th.nn.Linear(8, 1)
        self.sigmoid = th.nn.Sigmoid()

    def forward(self, G_sameCPU_graph, G_diffCPU_graph, UE_features, F_graph):
        UE_features = UE_features.to(th.float)
        G_sameCPU_graph = G_sameCPU_graph.to(th.int64)
        G_diffCPU_graph = G_diffCPU_graph.to(th.int64)
        F_graph = F_graph.to(th.int64)

        AP_features = self.bipartite((UE_features, th.zeros((self.L, UE_features.size(1)))), F_graph)

        sameCPU_embeddings = self.sameCPU(AP_features, G_sameCPU_graph)
        diffCPU_embeddings = self.diffCPU(AP_features, G_diffCPU_graph)

        embeddings = sameCPU_embeddings+diffCPU_embeddings

        predicted_AP_assignment = self.sigmoid(self.linear(embeddings))

        return predicted_AP_assignment


class SampleBuffer(object):

    def __init__(self, batch_size=1, max_size=10000000):
        self.storage = deque(maxlen=max_size)
        self.batch_size = batch_size

    def add(self, transition):
        self.storage.append(transition)

    def sample(self):
        minibatch = random.sample(self.storage, self.batch_size)

    def save(self, filename):
        # Save using pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.storage, f)

    def load(self, filename):
        # Load using pickle
        with open(filename, 'rb') as f:
            self.storage = pickle.load(f)


class DualGraphDataset(Dataset):
    def __init__(self, graphs, transform=None, pre_transform=None):
        super().__init__()
        self.data_list = graphs  # Store (undirected_graph, bipartite_graph) pairs
        self.transform = transform
        self.pre_transform = pre_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def add_sample(self, undirected_graph, bipartite_graph):
        """Add a sample to the dataset."""
        self.data_list.append((undirected_graph, bipartite_graph))

    def len(self):
        return len(self.data_list)


def custom_collate(data_list):
    undirected_graphs = [d[0] for d in data_list]
    bipartite_graphs = [d[1] for d in data_list]

    # Use `torch_geometric.loader.DataLoader` for batching
    batched_undirected = DataLoader(undirected_graphs, batch_size=len(undirected_graphs))
    batched_bipartite = DataLoader(bipartite_graphs, batch_size=len(bipartite_graphs))

    return batched_undirected, batched_bipartite


def bipartitegraph_generation(F, R):
    '''This function creates the edge list and feature matrix for the bipartite graph between APs and UEs
    INPUT>
    :param F: matrix with dimensions LxK where element (l,k) is '1' if AP L is one of the F preferred APs of UE k
    :param R: matrix with dimensions N x N x L x K where (:,:,l,k) is the spatial correlation
                            matrix between  AP l and UE k (normalized by noise variance)
    OUTPUT>
    edge_list: tensor with dimensions 2 x ... where the first row is the UE index and the second row is the AP index
    feature_matrix: tensor with a NN feature for each UE node
    '''

    F_graph = th.tensor(np.transpose(np.nonzero(F.T))).T

    F_graph_adapted = th.zeros((F_graph.shape[1], 2))
    UE_features = th.zeros((F_graph.shape[1], R.shape[0]*R.shape[1]), dtype=th.cfloat)
    for idx, k in enumerate(F_graph[0, :]):
        F_graph_adapted[idx, :] = torch.tensor([idx, F_graph[1, idx]])
        UE_features[idx, :] = torch.tensor(R[:, :, F_graph[1, idx], k].flatten())

    return F_graph_adapted.T, UE_features