"""
This module contains classes and functions for handling graph-based data structures. It includes implementations of
Graph Neural Network (GNN) models.
"""

import torch
import torch as th
from collections import deque
import random
import numpy as np
import pickle

from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.nn import GCNConv, SAGEConv
from functionsUtils import db2pow
from tqdm import tqdm


class GNN_model(th.nn.Module):
    def __init__(self):
        super().__init__()

    def model_train(self, train_loader, optimizer, loss_fn):
        self.train()
        total_loss = 0
        with tqdm(total=len(train_loader), desc="Training", unit="batch") as pbar:
            for batch in train_loader:
                optimizer.zero_grad()

                batched_undirected, batched_bipartite = batch

                # Compute the prediction
                predicted_AP_assignment = self(batched_undirected.full_sameCPU_edge,
                                               batched_undirected.diffCPU_edge,
                                               batched_bipartite.x, batched_bipartite.edge_index,
                                               L=batched_undirected.y.size(0))

                loss = loss_fn(predicted_AP_assignment.flatten(), batched_undirected.y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        return total_loss / len(train_loader)

    def model_validate(self, val_loader, loss_fn):
        self.eval()
        total_loss = 0
        with th.no_grad():
            for batch in val_loader:
                batched_undirected, batched_bipartite = batch

                # Compute the prediction
                predicted_AP_assignment = self(batched_undirected.full_sameCPU_edge,
                                               batched_undirected.diffCPU_edge,
                                               batched_bipartite.x, batched_bipartite.edge_index,
                                               L=batched_undirected.y.size(0))

                loss = loss_fn(predicted_AP_assignment.flatten(), batched_undirected.y)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def save_model(self, file_path):
        th.save(self.state_dict(), file_path)
        print(f'Model saved to {file_path}')

    def load_model(self, file_path):
        self.load_state_dict(th.load(file_path))
        self.eval()
        print(f'Model loaded from {file_path}')

class GNN_CorrMat(GNN_model):
    def __init__(self, UE_feature_size):
        super().__init__()
        self.bipartite = SAGEConv(2*UE_feature_size, 48, root_weight=False, aggr='sum')
        self.sameCPU = GCNConv(48, 48, normalize = False, add_self_loops=False)
        self.diffCPU = GCNConv(48, 48, normalize = False, add_self_loops=False)
        self.linear1 = th.nn.Linear(144, 100)
        self.linear2 = th.nn.Linear(100, 1)
        self.sigmoid = th.nn.Sigmoid()

    def forward(self, G_sameCPU_graph, G_diffCPU_graph, UE_features, F_graph, L):
        UE_features = th.cat((UE_features.real, UE_features.imag), dim=1).to(th.float)
        G_sameCPU_graph = G_sameCPU_graph.to(th.int64)
        G_diffCPU_graph = G_diffCPU_graph.to(th.int64)
        F_graph = F_graph.to(th.int64)

        AP_features = self.bipartite((UE_features, th.zeros((L, UE_features.size(1)))), F_graph)

        sameCPU_embeddings = self.sameCPU(AP_features, G_sameCPU_graph)
        diffCPU_embeddings = self.diffCPU(AP_features, G_diffCPU_graph)

        embeddings = th.cat((AP_features, sameCPU_embeddings, diffCPU_embeddings), dim=1)

        # predicted_AP_assignment = self.sigmoid(self.linear(embeddings))
        predicted_AP_assignment = th.nn.functional.relu(self.linear1(embeddings))
        predicted_AP_assignment = self.linear2(predicted_AP_assignment)

        return predicted_AP_assignment

class GNN_Gains(GNN_model):
    def __init__(self, UE_feature_size):
        super().__init__()
        self.bipartite = SAGEConv(UE_feature_size, 12, root_weight=False, aggr='sum')
        self.sameCPU = GCNConv(12, 12, normalize=False, add_self_loops=False)
        self.diffCPU = GCNConv(12, 12, normalize=False, add_self_loops=False)
        self.linear1 = th.nn.Linear(36, 36)
        self.linear2 = th.nn.Linear(36, 1)
        self.sigmoid = th.nn.Sigmoid()

    def forward(self, G_sameCPU_graph, G_diffCPU_graph, UE_features, F_graph, L):
        UE_features = UE_features.to(th.float)
        G_sameCPU_graph = G_sameCPU_graph.to(th.int64)
        G_diffCPU_graph = G_diffCPU_graph.to(th.int64)
        F_graph = F_graph.to(th.int64)

        AP_features = self.bipartite((UE_features, th.zeros((L, UE_features.size(1)))), F_graph)

        sameCPU_embeddings = self.sameCPU(AP_features, G_sameCPU_graph)
        diffCPU_embeddings = self.diffCPU(AP_features, G_diffCPU_graph)

        embeddings = th.cat((AP_features, sameCPU_embeddings, diffCPU_embeddings), dim=1)

        # predicted_AP_assignment = self.sigmoid(self.linear(embeddings))
        predicted_AP_assignment = th.nn.functional.relu(self.linear1(embeddings))
        predicted_AP_assignment = self.linear2(predicted_AP_assignment)

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
    def __init__(self, graphs = None, transform=None, pre_transform=None):
        super().__init__()
        self.data_list = graphs  # Store (undirected_graph, bipartite_graph) pairs
        self.transform = transform
        self.pre_transform = pre_transform
        self.parameters = {}

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def add_sample(self, undirected_graph, bipartite_graph):
        """Add a sample to the dataset."""
        self.data_list.append((undirected_graph, bipartite_graph))

    def len(self):
        return len(self.data_list)

    def combine(self, origen_filenames, destination_filename):
         data = th.load(origen_filenames[0])
         self.data_list = data['data_list']
         for file in origen_filenames[1:]:
            data = th.load(file)
            self.data_list += data['data_list']

         self.save(destination_filename)

    def save(self, file_path):
        th.save({'data_list': self.data_list, 'parameters': self.parameters}, file_path)
        print(f'Dataset and parameters saved to {file_path}')

    def load(self, file_path):
        data = th.load(file_path)
        self.data_list = data['data_list']
        self.parameters = data['parameters']
        print(f'Dataset and parameters loaded from {file_path}')


def custom_collate(data_list):
    undirected_graphs = [d[0] for d in data_list]
    bipartite_graphs = [d[1] for d in data_list]

    # Use `torch_geometric.loader.DataLoader` for batching
    batched_undirected = DataLoader(undirected_graphs, batch_size=len(undirected_graphs))
    batched_bipartite = DataLoader(bipartite_graphs, batch_size=len(bipartite_graphs))

    return batched_undirected, batched_bipartite


def bipartitegraph_generation(F, R, gainOverNoisedB, GNN_mode):
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
    match GNN_mode:
        case 'Gains':
            UE_features = th.zeros((F_graph.shape[1], 1), dtype=th.cfloat)
            for idx, k in enumerate(F_graph[0, :]):
                F_graph_adapted[idx, :] = torch.tensor([idx, F_graph[1, idx]])
                UE_features[idx] = db2pow(gainOverNoisedB[F_graph[1, idx], k])

            UE_features = UE_features / db2pow(np.max(gainOverNoisedB))

        case 'CorrMat':
            UE_features = th.zeros((F_graph.shape[1], R.shape[0]*R.shape[1]), dtype=th.cfloat)
            for idx, k in enumerate(F_graph[0, :]):
                F_graph_adapted[idx, :] = torch.tensor([idx, F_graph[1, idx]])
                UE_features[idx, :] = torch.tensor(R[:, :, F_graph[1, idx], k].flatten())

            UE_features = UE_features / db2pow(np.max(gainOverNoisedB))

        case _:
            raise ValueError('GNN_mode must be either "Gains" or "CorrMat"')

    return F_graph_adapted.T, UE_features

