"""
This script implements the training of a Graph Neural Network (GNN) to optimize the activation of Access Points (APs)
"""

import os
import torch as th
import glob


from functionsGraphHandling import SampleBuffer, DualGraphDataset, custom_collate, GNN_CorrMat, GNN_Gains
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

dataConfiguration = 'L_12_N_4_Q_2_T_4_f_5_K_(6_10)_taup_100_NbrSamp_20000'
GNN_mode = 'CorrMat' # 'CorrMat' or 'Gains'

dataset_directory = f'./AP_TrainingData/'+GNN_mode+'/Dataset_'+dataConfiguration
model_directory = f'./AP_TrainingData/'+GNN_mode+'/Model_'+dataConfiguration

# Load the dataset according to selected data configuration
dataset = DualGraphDataset()
try:
    dataset.load(dataset_directory+'.pt')
except:
    print('No stored training data found')

# set a seed for reproducibility
th.manual_seed(0)

# Model, optimizer, and loss
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
match GNN_mode:
    case 'CorrMat':
        model = GNN_CorrMat(UE_feature_size=dataset[0][1].x.shape[1]).to(device)
    case 'Gains':
        model = GNN_Gains(UE_feature_size=dataset[0][1].x.shape[1]).to(device)
    case _:
        raise ValueError('Invalid GNN mode')

optimizer = th.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = th.nn.BCEWithLogitsLoss()

train_size = int(0.9 * len(dataset))  # 90% for training
val_size = len(dataset) - train_size
train_dataset, val_dataset = th.utils.data.random_split(dataset, [train_size, val_size])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

val_loss = model.model_validate(val_loader, loss_fn)
print(f"Before training validation Loss: {val_loss:.4f}")

# Train the model
num_epochs = 30
for epoch in range(num_epochs):

    print(f'Epoch {epoch + 1}/{num_epochs}')

    train_loss = model.model_train(train_loader, optimizer, loss_fn)
    val_loss = model.model_validate(val_loader, loss_fn)
    print(f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

model.save_model(model_directory+f'_Epochs_{num_epochs}_SAGEConv_sum.pt')


