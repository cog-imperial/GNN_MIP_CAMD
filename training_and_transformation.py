# This file is used to train a GNN and then transform it into ONNX format, run this command:
# python training_and_transformation.py $dataset $seed_gnn
# $dataset is the name of dataset (QM7 or QM9)
# $seed_gnn is the random seed for traing GNN

# Note: please preprocess the dataset before using this file. 

import torch
from torch_geometric.data import Data, Dataset
import os
import os.path as osp
import numpy as np
import sys

class MyOwnDataset(Dataset):
    def __init__(self, root, length, transform=None, pre_transform=None, pre_filter=None):
        self.root = root
        self.length = length
        super().__init__(root, transform, pre_transform, pre_filter)
    
    def len(self):
        return self.length

    def get(self, idx):
        data = torch.load(osp.join(self.root, f'data_{idx}.pt'))
        return data
    
dataset_name = str(sys.argv[1]) # name of dataset
seed_gnn = int(sys.argv[2]) # random seed for training process

# relevant parameters for different datasets
if dataset_name == 'QM7':
    dataset = MyOwnDataset(root = 'QM7/', length = 5822)
    num_train = 5000
    gnn_channels = [16, 32, 16, 4, 1]
elif dataset_name == 'QM9':
    dataset = MyOwnDataset(root = 'QM9/', length = 108723)
    num_train = 80000
    gnn_channels = [32, 64, 16, 4, 1]

print(len(dataset))


print('=============================================================')
print('Information of an example from the dataset')
data = dataset[13]  # Get a graph object.
print(data)
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')
print('=============================================================')
torch.manual_seed(seed_gnn)
dataset_shuffle = dataset.shuffle()

# divide the dataset into training and test part
train_dataset = dataset_shuffle[:num_train]
test_dataset = dataset_shuffle[num_train:]

print('first training data: ', train_dataset[0].smiles)
print('first test data: ', test_dataset[0].smiles)

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')
print('=============================================================')

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_add_pool

# GNN architecture
class SAGE(torch.nn.Module):
    def __init__(self, seed, gnn_channels):
        super(SAGE, self).__init__()
        torch.manual_seed(seed)
        self.conv1 = SAGEConv(dataset.num_node_features, gnn_channels[0], 'sum')
        self.conv2 = SAGEConv(gnn_channels[0], gnn_channels[1], 'sum')
        self.lin1 = Linear(gnn_channels[1], gnn_channels[2])
        self.lin2 = Linear(gnn_channels[2], gnn_channels[3])
        self.lin3 = Linear(gnn_channels[3], gnn_channels[4])

    def forward(self, x, edge_index, batch):
        # SAGE layers for node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        # pooling (or read out) layer
        x = global_add_pool(x, batch)  # [batch_size, hidden_channels]

        # dense layers for final regressor
        x = self.lin1(x)
        x=x.relu()
        x = self.lin2(x)
        x=x.relu()
        x = self.lin3(x)
        
        return x

model = SAGE(seed_gnn, gnn_channels)
print(model)
#for param in model.parameters():
#    print(param)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.L1Loss()
def train():
    model.train()
    for data in train_loader: 
        out = model(data.x, data.edge_index, data.batch).squeeze() 
        loss = criterion(out, data.y)
        loss.backward() 
        optimizer.step()  
        optimizer.zero_grad()  

def test(loader):
    model.eval()
    correct = 0
    for data in loader: 
        out = model(data.x, data.edge_index, data.batch) 
        correct += torch.norm(out.squeeze()-data.y, p=1)
    return correct / len(loader.dataset)

# training the GNN
for epoch in range(1, 101):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# save the GNN
GNN_dir = f'GNN_models/{dataset_name}/'
if not os.path.exists(GNN_dir):
    os.makedirs(GNN_dir)
torch.save(model, osp.join(GNN_dir, f'GNN_{seed_gnn}.pt'))
print('GNN model is saved')

model = torch.load(osp.join(GNN_dir, f'GNN_{seed_gnn}.pt'))
params_gnn = []
for param in model.parameters():
    params_gnn.append(param.detach().numpy())
# print(len(params_gnn))

# transform a SAGE layer to a Dense layer
def SAGE_to_Dense(N, w1, w2, b):
    out_channel, in_channel = w1.shape
    weight = np.zeros((N*out_channel, N*in_channel))
    bias = np.zeros(N*out_channel)
    for u in range(N):
        for v in range(N):
            if u == v:
                weight[u*out_channel:(u+1)*out_channel, v*in_channel:(v+1)*in_channel] = w2
            else:
                weight[u*out_channel:(u+1)*out_channel, v*in_channel:(v+1)*in_channel] = w1
        bias[u*out_channel:(u+1)*out_channel] = b
    return weight, bias

# construct the Dense NN for different N
from omlt.io import write_onnx_model_with_bounds
for N in range(4, 9):
    print('N = ', N)
    F = 16
    L = 6
    layers = ['gnn', 'gnn', 'add_pool', 'dense', 'dense', 'dense']
    activations = [True, True, False, True, True, False]
    params = []
    params_index = 0
    channels = []
    channels.append(N*F)
    for layer in layers:
        if layer == 'gnn':
            w1 = params_gnn[params_index]
            params_index += 1
            b = params_gnn[params_index]
            params_index += 1
            w2 = params_gnn[params_index]
            params_index += 1
            params.append(SAGE_to_Dense(N,w1,w2,b))
            channels.append(w1.shape[0] * N)
        elif layer == 'dense':
            w = params_gnn[params_index]
            params_index += 1
            b = params_gnn[params_index]
            params_index += 1
            params.append((w,b))
            channels.append(w.shape[0])
        elif layer == 'add_pool':
            channels.append(channels[-1] // N)
            w = np.zeros((channels[-1],channels[-2]))
            for i in range(channels[-1]):
                for j in range(N):
                    w[i, i+j*channels[-1]] = 1.
            b = np.zeros(channels[-1])
            params.append((w,b))
    print(channels)
    for param in params:
        print(param[0].shape, param[1].shape)

    import torch 
    import torch.nn as nn
    import torch.nn.functional as F

    class PyTorchModel(nn.Module):
        def __init__(self, L, params, activations):
            super().__init__()
            layers = []
            for l in range(L):
                layers.append(nn.Linear(params[l][0].shape[1], params[l][0].shape[0]))
                layers[-1].weight = nn.Parameter(torch.tensor(params[l][0], dtype=torch.float64))
                layers[-1].bias = nn.Parameter(torch.tensor(params[l][1], dtype=torch.float64))
                if activations[l]:
                    layers.append(nn.ReLU(True))
            self.layer = nn.Sequential(*layers)
            
        def forward(self, x):
            x = self.layer(x)
            return x

    model_dense = PyTorchModel(L, params, activations)
    print(model_dense)

    dummy_input = torch.zeros(channels[0], dtype=torch.float64)
    dummy_input.requires_grad=True
    lb = np.zeros(channels[0])
    ub = np.ones(channels[0])
    input_bounds = [(l, u) for l, u in zip(lb, ub)]

    # save the Dense NN
    Dense_dir = f'Dense_models/{dataset_name}/N={N}'
    if not os.path.exists(Dense_dir):
        os.makedirs(Dense_dir)

    torch.onnx.export(
        model_dense,
        dummy_input,
        osp.join(Dense_dir, f'Dense_{seed_gnn}.onnx'),
        input_names=['input'],
        output_names=['output'],
    )
    write_onnx_model_with_bounds(osp.join(Dense_dir, f'Dense_{seed_gnn}.onnx'), None, input_bounds)
