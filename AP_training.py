import os
import torch as th
import glob

from functionsGraphHandling import SampleBuffer, DualGraphDataset, custom_collate, GNN_model
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# Stored training data
try:
    graphs = th.load(f'./AP_TrainingData/AP_training_Dataset_L_9_N_4_Q_3_T_5_f_5_taup_10_NbrSamp_10_20250117_105530.pt')
except:
    graphs = []

# Create the dataset and load the information in the buffers
dataset = DualGraphDataset(graphs)

# Create a list of buffers
buffers = []
# pattern that repeats in the file names
pattern = f'./AP_TRAININGDATA/newData/GraphTrainingBuffer_Comb_MMSE_L_9_N_4_Q_3_T_5_f_5_taup_10_NbrSamp_'
filename = f'{pattern}*.pkl'
# Get the list of files that match the pattern
matching_files = glob.glob(filename)
# Load the buffers from the files
for file in matching_files:
    buffer = SampleBuffer(batch_size=10)
    buffer.load(file)
    buffers.append(buffer)
    os.rename(file, file.replace('newData', 'inDataSet'))

if buffers.__len__() > 0:
    # run over the buffers in the list
    for buffer in buffers:
        # run over the elements in the buffer
        for sample in buffer.storage:
            G_graph = Data(sameCPU_edge = sample[0], full_sameCPU_edge = sample[1], diffCPU_edge = sample[2],
                           y = th.tensor(sample[5], dtype=th.float))
            F_graph = Data(edge_index = sample[3], x = sample[4])
            dataset.add_sample(G_graph, F_graph)
    th.save(dataset.data_list, f'./AP_TrainingData/AP_training_Dataset_L_9_N_4_Q_3_T_5_f_5_taup_10_Samples_'
                               f'{len(dataset.data_list)}.pt')

# set a seed for reproducibility
th.manual_seed(0)

# Model, optimizer, and loss
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
model = GNN_model(UE_feature_size=dataset[0][1].x.shape[1]).to(device)
optimizer = th.optim.Adam(model.parameters(), lr=0.0001)
# loss_fn = th.nn.MSELoss()
loss_fn = th.nn.BCEWithLogitsLoss()

train_size = int(0.9 * len(dataset))  # 80% for training
val_size = len(dataset) - train_size
train_dataset, val_dataset = th.utils.data.random_split(dataset, [train_size, val_size])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

val_loss = model.model_validate(val_loader, loss_fn)
print(f"Before training validation Loss: {val_loss:.4f}")

# Train the model
num_epochs = 3
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    train_loss = model.model_train(train_loader, optimizer, loss_fn)
    val_loss = model.model_validate(val_loader, loss_fn)
    print(f"Epoch {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

model.save_model(f'./AP_TrainingData/AP_trained_model_L_9_N_4_Q_3_T_5_f_5_taup_10_Samples_{len(dataset.data_list)}_nonNorm.pt')




# model.train()
# for epoch in range(nbrOfEpochs):
#     print(f'Epoch {epoch + 1}/{nbrOfEpochs}')
#
#     # Shuffle the dataset
#     dataset = dataset.shuffle()
#
#     for batch in loader:
#         print(f'Batch {counter}/{len(loader)}')
#
#         batched_undirected, batched_bipartite = batch
#         optimizer.zero_grad()
#
#         # Compute the prediction
#         predicted_AP_assignment = model(batched_undirected.full_sameCPU_edge, batched_undirected.diffCPU_edge,
#                                         batched_bipartite.x, batched_bipartite.edge_index)
#
#         loss = loss_fn(predicted_AP_assignment.flatten(), batched_undirected.y)
#         loss.backward()
#         optimizer.step()
#
#         average_loss += loss.item()
#
#         print(f'Average Loss: {average_loss/counter}')
#         counter += 1
#
#         print(f'Loss: {loss.item()}')


#
# # Split the dataset into training and test sets
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = th_data.random_split(dataset, [train_size, test_size])
#
# # Dataloaders
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#
# # Model, optimizer, and loss
# device = th.device('cuda' if th.cuda.is_available() else 'cpu')
# model = SingleLayerGNN(in_channels=dataset.num_features, out_channels=7).to(device)
# optimizer = th.optim.Adam(model.parameters(), lr=0.01)
# loss_fn = th.nn.BCEWithLogitsLoss()
#
# model.train()
# average_loss = 0
# for data in train_loader:
#     data = data.to(device)
#     optimizer.zero_grad()
#
#     embedding = model(data.x, data.edge_index)
#
#     # Compute loss with the graph's vector label
#     loss = loss_fn(embedding, data.y.to(th.float))
#     loss.backward()
#     optimizer.step()
#     average_loss += loss.item()/len(train_loader)
#
#     print(f'Loss: {loss.item()}')
#

print('End of the script')