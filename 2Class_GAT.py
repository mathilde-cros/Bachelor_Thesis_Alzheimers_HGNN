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

import networkx as nx
from networkx.convert_matrix import from_numpy_array

import functions as f
from functions import dict_to_array, normalize_array
import models as m


# In[2]:


os.environ['WANDB_NOTEBOOK_NAME']="GAT_2Class.ipynb"


# In[3]:


# Loading the time series
time_series_list = f.load_time_series()
time_series_df_list = f.list_of_df_of_time_series(time_series_list)


# In[4]:


# Loading the pearson correlation matrices
corr_matrix_list = f.corr_matrix_paths()['pearson']
print(corr_matrix_list)
# Example patient
path = f'ADNI_full/corr_matrices/corr_matrix_pearson/{corr_matrix_list[0]}'
corr_matrix_patient = np.loadtxt(path, delimiter=',')
print(corr_matrix_patient)


# In[5]:


# Loading the diagnostic labels
diagnostic_label_all = np.loadtxt('ADNI_full/diagnostic_label.csv', dtype=str, delimiter=',')


# In[6]:


def filter_group(group):
    df = pd.read_csv('ADNI_full/patient_info.csv')
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


assert len(diagnostic_label) == len(corr_matrices)


# In[9]:


# Defining a class to preprocess raw data into a format suitable for training Graph Neural Networks (GNNs).
## With the possibility of assigning weight to edges, adding the age feature, sex feature, and matrixe profiling.

class Raw_to_Graph_2Class(InMemoryDataset):
    def __init__(self, root, threshold, method, weight=False, age=False, sex=False, matrixprofile=False, transform=None, pre_transform=None):
        self.threshold = threshold
        self.method = method
        self.weight = weight
        self.age = age
        self.sex = sex
        self.matrixprofile = matrixprofile
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    # This function is used to process the raw data into a format suitable for GNNs, by constructing graphs out of the connectivity matrices.
    def process(self):
        graphs=[]
        for patient_idx, patient_matrix in enumerate(corr_matrices):
            path = f'ADNI_full/corr_matrices/corr_matrix_{self.method}/{patient_matrix}'
            corr_matrix = np.loadtxt(path, delimiter=',')
            # Here ROIs stands for Regions of Interest
            nbr_ROIs = corr_matrix.shape[0]
            edge_matrix = np.zeros((nbr_ROIs,nbr_ROIs))
            for j in range(nbr_ROIs):
                for k in range(nbr_ROIs):
                    # Here we are using the absolute value of each element of the correlation matrix, as the corr coeff is in the range [-1,1].
                    if np.abs(corr_matrix[j,k]) < self.threshold:
                        edge_matrix[j,k] = 0
                    else:
                        if self.weight:
                            # Here we assign the absolute value of the correlation coefficient as the edge weight.
                            edge_matrix[j,k] = corr_matrix[j,k]
                        else:
                            # Here we assign 1 as the edge weight, i.e. regardless of the the absolute value of the correlation coefficient.
                            edge_matrix[j,k] = 1

            # Create a NetworkX graph from the edge matrix
            NetworkX_graph = from_numpy_array(edge_matrix)

            # Compute the degree, betweenness centrality, clustering coefficient, local efficiency for each node of the graph and the global efficiency of the graph
            degree_dict = dict(NetworkX_graph.degree())
            between_central_dict = nx.betweenness_centrality(NetworkX_graph)
            cluster_coeff_dict = nx.clustering(NetworkX_graph)
            global_eff = nx.global_efficiency(NetworkX_graph)
            local_eff_dict = {}
            for node in NetworkX_graph.nodes():
                subgraph_neighb = NetworkX_graph.subgraph(NetworkX_graph.neighbors(node))
                if subgraph_neighb.number_of_nodes() > 1:
                    efficiency = nx.global_efficiency(subgraph_neighb)
                else:
                    efficiency = 0.0
                local_eff_dict[node] = efficiency

            # Convert the degree, betweenness centrality, local efficiency, clustering coefficient and ratio of local to global efficiency dictionaries to NumPy arrays then normalize them
            degree_array = dict_to_array(degree_dict)
            degree_array_norm = normalize_array(degree_array)

            between_central_array = dict_to_array(between_central_dict)
            between_central_array_norm = normalize_array(between_central_array)

            local_efficiency_array = dict_to_array(local_eff_dict)
            local_eff_array_norm = normalize_array(local_efficiency_array)

            ratio_local_global_array = dict_to_array(local_eff_dict) / global_eff
            ratio_local_global_array_norm = normalize_array(ratio_local_global_array)

            cluster_coeff_array = dict_to_array(cluster_coeff_dict)
            cluster_coeff_array_norm = normalize_array(cluster_coeff_array)

            # Initializing an array for the graph features
            x_array = np.stack([degree_array_norm, between_central_array_norm, local_eff_array_norm, cluster_coeff_array_norm, ratio_local_global_array_norm], axis=-1)
            x_array = x_array.astype(np.float32)
            
            if self.matrixprofile:
                path = f'ADNI_full/matrix_profiles/matrix_profile_{self.method}/{patient_matrix}'
                with open(path, "rb") as fl:
                  patient_dict = pkl.load(fl)
                # combine dimensions
                features = np.array(patient_dict['mp']).reshape(len(patient_dict['mp']),-1)
                features = features.astype(np.float32)
                x_array = np.concatenate((x_array, features), axis=-1)

            # Concatenate the degree, participation coefficient, betweenness centrality, local efficiency, and ratio of local to global efficiency arrays to form a single feature vector
            x = torch.tensor(x_array, dtype=torch.float)

            # Create a Pytorch Geometric Data object from the NetworkX
            graph_data = from_networkx(NetworkX_graph)
            ## The feature matrix of the graph is the degree, betweenness centrality, local efficiency, clustering coefficient and ratio of local to global efficiency of each node
            graph_data.x = x
            ## The target/output variable that we want to predict is the diagnostic label of the patient
            graph_data.y = float(diagnostic_label[patient_idx])
            graphs.append(graph_data)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


# In[10]:


# Doing some parameter gridsearch to find the best hyperparameters
from sklearn.model_selection import ParameterGrid

# Building the graphs
threshold = 0.5
age = False
sex = False
matrixprofile = True
weight = True
if matrixprofile:
    in_channels = 461 + int(age) + int(sex)
else:
    in_channels = 5 + int(age) + int(sex)
method = 'pearson'

root = f'Raw_to_graph_2Class/ADNI_T_{threshold}_M_{method}_W{weight}_A{age}_S{sex}_MP{matrixprofile}'
dataset = Raw_to_Graph_2Class(root=root, threshold=threshold, method=method, weight=weight, sex=sex, age=age, matrixprofile=matrixprofile)
f.dataset_features_and_stats(dataset, diagnostic_label)

# Creating the train, validation and test sets
train_loader, valid_loader, test_loader, nbr_classes = f.create_train_test_valid(dataset)

param_grid = {
    'learning_rate': [0.001, 0.0001],
    'hidden_channels': [128, 64],
    'num_layers': [3, 2, 1],
    'dropout_rate': [0.2, 0.1, 0.0],
    'weight_decay': [0.001, 0.0001],
    'heads': [4, 3]
}
# param_grid = {
#     'learning_rate': [0.0001, 0.001],
#     'hidden_channels': [64, 128],
#     'num_layers': [1, 2, 3],
#     'dropout_rate': [0.0, 0.1, 0.2],
#     'weight_decay': [0.0001, 0.001],
#     'heads': [3, 4]
# }

# Create combinations of hyperparameters
param_combinations = ParameterGrid(param_grid)
n_epochs = 500
# Train using each combination
for params in param_combinations:
    if matrixprofile:
        filename = f'2Class_Models/GAT_Models_MP/threshold_{threshold}/lr{params["learning_rate"]}_hc{params["hidden_channels"]}_nl{params["num_layers"]}_d{params["dropout_rate"]}_epochs{n_epochs}_heads{params["heads"]}_wdecay{params["weight_decay"]}_w{weight}.png'
    else:
        filename = f'2Class_Models/GAT_Models/threshold_{threshold}/lr{params["learning_rate"]}_hc{params["hidden_channels"]}_nl{params["num_layers"]}_d{params["dropout_rate"]}_epochs{n_epochs}_heads{params["heads"]}_wdecay{params["weight_decay"]}_w{weight}.png'
    if os.path.exists(filename):
        pass
    else:
        parameters = [params['learning_rate'], params['hidden_channels'], params['num_layers'], params['dropout_rate'], params['heads']]
        model = m.GAT(in_channels=in_channels, hidden_channels=parameters[1], out_channels=nbr_classes, num_layers=parameters[2], dropout=parameters[3], heads=parameters[4], nbr_classes=nbr_classes)
        criterion = torch.nn.CrossEntropyLoss()
        if 'weight_decay' not in params.keys():
            w_decay = 0
        else:
            w_decay = params['weight_decay']
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters[0], weight_decay=w_decay)
        train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy, test_accuracy = m.train_GAT(model, optimizer, criterion, w_decay, threshold, train_loader, valid_loader, parameters, test_loader, testing=True, n_epochs=800)

