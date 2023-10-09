# Two datasets (QM7 and QM9) used in this paper are avaiable in package chemprop
# chemprop website: https://chemprop.readthedocs.io/en/latest/tutorial.html#data. 
# We already download and uncompress them here. 

# This file is used to preprocess QM7 dataset, run this command:
# python QM7_dataset_generation.py

import chemprop
from chemprop.data.utils import get_data
from chemprop.features.featurization import MolGraph
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

paths = ['data/qm7.csv', 'data/qm9.csv']

i = 0 # QM7 dataset

# information about QM7 dataset
dir = 'QM7/'
types = 4
atoms = ['C', 'N', 'O', 'S']
orders = [6-1,7-1,8-1,16-1]
covalence = [4, 3, 2, 2]
target_position = 0

It = [0,1,2,3] # indexes for types of atom
In = [4,5,6,7,8] # indexes for numbers of neighbors
Ih = [9,10,11,12,13] # indexes for numbers of hydrogen atoms
Idb = 14 # index for double bond
Itb = 15 # index for triple bond

num = np.zeros(types)
# print(paths[i])
Molecule_Data = get_data(paths[i]) # load the dataset

num_mol = len(Molecule_Data._data) # number of molecules
tag = np.ones(num_mol,dtype=np.int8) # use for checking compatibility of each molecule
num_aromaticity = 0 # number of armatic molecules
num_stereo = 0 # number of molecules with stereo bond

for k, data in enumerate(Molecule_Data._data):
    graph = MolGraph(data.smiles[0])
    f_atoms = np.array(graph.f_atoms, dtype=np.int8) # atom features
    f_bonds = np.array(graph.f_bonds, dtype=np.int8) # bond features
    N = graph.n_atoms # number of atoms
    M = graph.n_bonds # number of bonds
    # check abnormal atoms
    if np.sum(f_atoms[:,orders]) != N:
        # print('Abnormal atoms: ', k, data.smiles[0])
        tag[k] = 0
        continue
    
    # check if there are any abnormal values
    if np.sum(f_atoms[:,[100,107,113,118,124,130]]):
        # print('Abnormal value: ', k, data.smiles[0])
        tag[k] = 0
        continue

    #check formal charge
    if np.sum(f_atoms[:,112]) != N:
        # print('formal charge: ', k, data.smiles[0])
        tag[k] = 0
        continue
    
    # check aromaticity
    if np.sum(f_atoms[:,-2]):
        # print('aromaticity: ', k, data.smiles[0])
        num_aromaticity = num_aromaticity + 1
        tag[k] = 0
        continue

    # check if any bond has stereo
    if M and np.sum(f_bonds[:,[141,142,143,144,145,146]]):
        # print('stereo: ', k, data.smiles[0])
        num_stereo = num_stereo + 1
        tag[k] = 0
        continue
    
    # check if the molecule has non-zero radical electrons
    mol = Chem.MolFromSmiles(data.smiles[0])
    if Descriptors.NumRadicalElectrons(mol):
        # print('non-zero radical electrons: ', k, data.smiles[0])
        tag[k] = 0
        continue
    
    # check the covalance for each type of atom
    flag = np.zeros(types)
    for i in range(N):
        degree = f_atoms[i,120] + 2 * f_atoms[i,121] + 3 * f_atoms[i,122] + 4 * f_atoms[i,123]
        for index in graph.a2b[i]:
            degree = degree + f_bonds[index, 134] + 2 * f_bonds[index, 135] + 3 * f_bonds[index, 136]
        for j in range(types):
            if f_atoms[i,orders[j]] and degree!=covalence[j]:
                flag[j] = 1
        
    for j in range(types):
        num[j] += flag[j]
    
    if np.sum(flag):
        tag[k] = 0

print('num_aromaticity = ', num_aromaticity)
print('num_stereo = ', num_stereo)
print('abonormal atoms: ', num)
print('feasible molecule: ', np.sum(tag))

# re-scale the properties to 0~1
value_list = []
for k, molecule in enumerate(Molecule_Data._data):
    if not tag[k]:
        continue
    value_list.append(molecule.targets[target_position])
values = np.array(value_list)
y_max = np.max(values)
y_min = np.min(values)
y_mean = np.average(values)
y_std = np.std(values)
# print statistical values
print('max: ', y_max)
print('min: ', y_min)
print('avg: ', y_mean)
print('std: ', y_std)

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch 
import os

# extract desired features and save the preprocessed dataset
idx = 0
for k, molecule in enumerate(Molecule_Data._data):
    if not tag[k]:
        continue
    graph = MolGraph(molecule.smiles[0])
    N = graph.n_atoms
    M = graph.n_bonds
    f_atoms = np.array(graph.f_atoms, dtype=np.int8)
    f_bonds = np.array(graph.f_bonds, dtype=np.int8)

    X = np.concatenate((f_atoms[:,orders], np.zeros((N,len(In)), dtype=np.int8), f_atoms[:,119:124], np.zeros((N,2), dtype=np.int8)), axis=1)

    edges = np.zeros((2,M), dtype=np.int8)
    db_tag = np.zeros(M, dtype=np.int8)
    tb_tag = np.zeros(M, dtype=np.int8)
    
    for v in range(N):
        X[v, len(It)+ len(graph.a2b[v])] = 1

    for i in range(M):
        u = graph.b2a[i]
        v = graph.b2a[graph.b2revb[i]]
        edges[0, i] = u
        edges[1, i] = v
        if graph.f_bonds[i][135]:
            X[u, Idb] = 1
            X[v, Idb] = 1
            db_tag[i] = 1
        if graph.f_bonds[i][136]:
            X[u, Itb] = 1
            X[v, Itb] = 1
            tb_tag[i] = 1
            
    # print(X)
    # print(edges)
    # print(db_tag)
    # print(tb_tag)
    # print(molecule.targets[targey_position])
    
    y_original = molecule.targets[target_position]
    y = (y_original - y_min) / (y_max - y_min)
    
    data = Data(x = torch.tensor(X, dtype=torch.float),
                edge_index = torch.tensor(edges, dtype=torch.long),
                y = torch.tensor(y, dtype=torch.float),
                y_original = torch.tensor(y_original, dtype=torch.float),
                db_tag = torch.tensor(db_tag, dtype=torch.long),
                tb_tag = torch.tensor(tb_tag, dtype=torch.long),
                index = torch.tensor(k, dtype=torch.long),
                smiles = molecule.smiles[0],
               )
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(data, os.path.join(dir, f'data_{idx}.pt'))
    idx = idx + 1

print('QM7 dataset is prepared!')