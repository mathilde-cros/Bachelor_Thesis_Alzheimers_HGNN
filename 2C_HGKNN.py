#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import wandb

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

import networkx as nx
from networkx.convert_matrix import from_numpy_array
import hypernetx as hnx

import functions as f
from functions import dict_to_array, normalize_array
import models as m


# In[2]:


# Loading the time series
time_series_list = f.load_time_series()
time_series_df_list = f.list_of_df_of_time_series(time_series_list)


# In[3]:


# Loading the pearson correlation matrices
corr_matrix_list = f.corr_matrix_paths()['pearson']
print(corr_matrix_list)
# Example patient
path = f'ADNI_full/corr_matrices/corr_matrix_pearson/{corr_matrix_list[0]}'
corr_matrix_patient = np.loadtxt(path, delimiter=',')
print(corr_matrix_patient)


# In[4]:


# Loading the diagnostic labels
diagnostic_label_all = np.loadtxt('ADNI_full/diagnostic_label.csv', dtype=str, delimiter=',')


# In[5]:


time_series_folder = 'ADNI_full/time_series'


# In[6]:


def filter_group(group):
    df = pd.read_csv('ADNI_FULL/patient_info.csv')
    labels = df['Research Group']
    label_idx_list = [i for i in range(len(labels)) if labels[i] == group]
    return label_idx_list


# In[7]:


#binary classification
cn = filter_group('CN')
ad = filter_group('AD')
bin_idx = sorted(cn + ad)
corr_matrices = [corr_matrix_list[i] for i in bin_idx]
diagnostic_label = [diagnostic_label_all[i] for i in bin_idx]
for i in range(len(diagnostic_label)):
    if diagnostic_label[i] == 'CN':
        diagnostic_label[i] = 0
    elif diagnostic_label[i] == 'AD':
        diagnostic_label[i] = 1
    else:
        print('Error: incorrect label')


# In[8]:


time_series_folder = 'ADNI_full/time_series'
time_series_folder_binary = []
for i, file_name in enumerate(os.listdir(time_series_folder)):
    if i in bin_idx:
        time_series_folder_binary.append(f'{time_series_folder}/{file_name}')
print(time_series_folder_binary)


# In[9]:


assert len(diagnostic_label) == len(corr_matrices)


# In[10]:


# architecture = 'CK_2Class'
# architecture = 'FC_2Class'
# architecture = 'MC_2Class'
architecture = 'KNN_2Class'


# In[11]:


os.environ['WANDB_NOTEBOOK_NAME']=f'{architecture}.ipynb'


# In[12]:


# Defining a class to preprocess raw data into a format suitable for training Graph Neural Networks (GNNs).
## With the possibility of assigning weight to edges, adding the age feature, sex feature, and matrixe profiling.

class Raw_to_Hypergraph_2Class(InMemoryDataset):
    def __init__(self, root, hg_data_path, method, weight, threshold, age=False, sex=False, transform=None, pre_transform=None):
        self.method = method
        self.weight = weight
        self.threshold = threshold
        self.age = age
        self.sex = sex
        self.hg_data_path = hg_data_path
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    # This function is used to process the raw data into a format suitable for GNNs, by constructing graphs out of the connectivity matrices.
    def process(self):
        # Loading the prebuilt hypergraphs and the correlation matrices
        hg_dict_list = f.load_hg_dict(self.hg_data_path)

        graphs=[]
        for patient_idx, patient_matrix in enumerate(corr_matrices):
            # Create a NetworkX graph from the hypergraph matrix
            patient_hg = hg_dict_list[patient_idx]
            hypergraph = hnx.Hypergraph(patient_hg)

            # Adding the matrix profiling features to the feature array
            path = f'ADNI_full/matrix_profiles/matrix_profile_pearson/{patient_matrix}'
            if patient_matrix.endswith('.DS_Store'):
                continue  # Skip hidden system files like .DS_Store
            with open(path, "rb") as fl:
                patient_dict = pkl.load(fl)
            # combine dimensions
            features = np.array(patient_dict['mp']).reshape(len(patient_dict['mp']),-1)
            features = features.astype(np.float32)

            # Concatenate the degree, participation coefficient, betweenness centrality, local efficiency, and ratio of local to global efficiency arrays to form a single feature vector
            x = torch.tensor(features, dtype=torch.float)

            # Create a Pytorch Geometric Data object
            edge_index0 = []
            edge_index1 = []
            i = 0
            for hyperedge, nodes in hypergraph.incidence_dict.items():
                edge_index0 = np.concatenate((edge_index0, nodes), axis=0)
                for j in range(len(nodes)):
                    edge_index1.append(i)
                i += 1
            edge_index = np.stack([[int(x) for x in edge_index0], edge_index1], axis=0)
            y = torch.tensor(float(diagnostic_label[patient_idx]))
            hg_data = Data(x=x, edge_index=torch.tensor(edge_index, dtype=torch.long), y=y)
            graphs.append(hg_data)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


# In[13]:


assert len(corr_matrices) == len(time_series_folder_binary)


# In[14]:


corr_matrices[0]


# In[23]:


# Building the Hypergraphs (choose one method cell above)
# method_list = ['maximal_clique', 'knn']
method_list = ['knn']
m.save_all_hypergraphs(method_list, time_series_folder_binary, corr_matrices, diagnostic_label)


# In[25]:


# Doing some parameter gridsearch to find the best hyperparameters
from sklearn.model_selection import ParameterGrid

# Building the graphs
threshold = 0.5
k_neighbors = 3
age = False
sex = False
method = 'knn'
if method == 'knn':
    threshold = f'{k_neighbors}neighbors'
weight = False

hg_data_path = f'Hypergraphs/{method}/thresh_{threshold}'
root = f'Raw_to_hypergraph_2Class/ADNI_T_{threshold}_M_{method}_W{weight}_A{age}_S{sex}_MPTrue'
dataset = Raw_to_Hypergraph_2Class(root=root, hg_data_path=hg_data_path, method=method, weight=weight, threshold=threshold, age=age, sex=sex)

# Creating the train, validation and test sets
train_loader, valid_loader, test_loader, nbr_classes = f.create_train_test_valid(dataset)

# param_grid = {
#     'learning_rate': [0.001, 0.0001],
#     'hidden_channels': [128, 64],
#     'num_layers': [3, 2, 1],
#     'dropout_rate': [0.2, 0.1, 0.0],
#     'weight_decay': [0.001, 0.0001]
# }
param_grid = {
    'learning_rate': [0.0001, 0.001],
    'hidden_channels': [64, 128],
    'num_layers': [1, 2, 3],
    'dropout_rate': [0.0, 0.1, 0.2],
    'weight_decay': [0.0001, 0.001]
}

# Create combinations of hyperparameters
param_combinations = ParameterGrid(param_grid)
architecture = architecture
n_epochs = 800
in_channels = dataset.num_features
# Train using each combination
for params in param_combinations:
    filename = f'2Class_Models/HGConv_Models_MP/threshold_{threshold}/method_{method}/lr{params["learning_rate"]}_hc{params["hidden_channels"]}_nl{params["num_layers"]}_d{params["dropout_rate"]}_epochs{n_epochs}_wdecay{params["weight_decay"]}_w{weight}.png'
    if os.path.exists(filename):
        pass
    else:
        parameters = [params['learning_rate'], params['hidden_channels'], params['num_layers'], params['dropout_rate']]
        model = m.HGConv(in_channels=in_channels, hidden_channels=parameters[1], out_channels=nbr_classes, num_layers=parameters[2], dropout=parameters[3], nbr_classes=nbr_classes)
        criterion = torch.nn.CrossEntropyLoss() 
        if 'weight_decay' not in params.keys():
            w_decay = 0
        else:
            w_decay = params['weight_decay']
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters[0], weight_decay=w_decay)
        train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy, test_accuracy = m.train_HGConv(model, optimizer, criterion, w_decay, threshold, train_loader, valid_loader, parameters, architecture, test_loader, testing=True, n_epochs=800)

