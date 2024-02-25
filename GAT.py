#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, ELU, Sequential, ModuleList
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_mean_pool

from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split


# In[2]:


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, heads, nbr_classes):
        super(GAT, self).__init__()
        self.nbr_classes = nbr_classes
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        self.mlp = torch.nn.Sequential(torch.nn.Linear(hidden_channels * heads, hidden_channels), torch.nn.ReLU(), torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)  # Using ELU activation for GAT
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.mlp(x)
        return x


# In[3]:


# Some cells from the Graph_Building ipynb because I didn't manage to save the dataset in a way that I could use it in another notebook

from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx

import networkx as nx
from networkx.convert_matrix import from_numpy_array

# Creating a dictionary of lists of paths to the correlation matrices for each method. Each list in the dictionary represents a different method.
methods = ['pearson', 'spearman', 'kendall', 'partial']
full_corr_path_lists = {}
for method in methods:
    method_dir = f'ADNI_full/corr_matrices/corr_matrix_{method}/'
    full_corr_path_lists[method] = []
    for file in os.listdir(method_dir):
        full_corr_path_lists[method].append(file)
# Generating the diagnostic file from the diagnostic_label.csv file
diagnostic_label = np.loadtxt('ADNI_full/diagnostic_label.csv', dtype=str, delimiter=',')

# Combining the 'EMCI', 'LMCI' and 'MCI' diagnostics into a single 'MCI' label for simplicity, then one-hot encoding the diagnostics
for patient in range(len(diagnostic_label)):
    if diagnostic_label[patient] == 'CN':
        diagnostic_label[patient] = 0
    elif diagnostic_label[patient] == 'SMC':
        diagnostic_label[patient] = 1
    elif diagnostic_label[patient] == 'EMCI' or diagnostic_label[patient] == 'LMCI' or diagnostic_label[patient] == 'MCI':
        diagnostic_label[patient] = 2
    elif diagnostic_label[patient] == 'AD':
        diagnostic_label[patient] = 3
    else:
        print('Error: Diagnostic label not recognised')
        break

# Loading the age feature of patients to use as a node feature
ages = np.loadtxt('ADNI_full/age.csv', delimiter=',')
min_age = np.min(ages)
max_age = np.max(ages)

# Prepocessing the sex feature of patients to use as a node feature. Here, 0 represents male patients and 1 represents female patients
sex = np.loadtxt('ADNI_full/sex.csv', dtype=str, delimiter=',')
for patient in range(len(sex)):
    if sex[patient] == 'M':
        sex[patient] = 0
    else:
        sex[patient] = 1

# Defining functions to simplify the code in the class Raw_to_Graph.
# To convert a dictionnary into a numpy array
def dict_to_array(dict):
    array = np.array(list(dict.values()))
    return array

# To normalize an array
def normalize_array(array):
    norm_array = (array - np.mean(array)) / np.std(array)
    return norm_array

# Functions to implement the matrix profile algorithm using STUMPY++. 
## This is commented out as my local kernel dies when running this cell because of memory issues with matrix profiling. I acquired the data by running it on colab and saving it in the repo here. Only uncomment if needed.

# # Computing the matrix profile using STUMPY++
# def compute_matrix_profile(time_series, m=3):
#     # Here the m parameter is the window size, i.e. the length of the subsequence to compute the matrix profile for. 
#     # Are we looking for short-lived neural activations (then small m), or are we interested in detecting longer-term changes or anomalies in brain activity (large m).
#     matrix_profile, idx = stump(time_series, m)
#     return matrix_profile, idx

# # Finding the motif and discord in the matrix profile
# def find_motif_discord(matrix_profile, idx):
#     motif_idx = np.argmin(matrix_profile)
#     discord_idx = np.argmax(matrix_profile)
#     motif = idx[motif_idx]
#     discord = idx[discord_idx]
#     return motif, discord

# # Function to compute the matrix profile, motifs and discords for a correlation matrix
# def matrix_profile(corr_matrix):
#     matrix_profiles = []
#     motifs = []
#     discords = []
#     for row in corr_matrix:
#         mp, idx = compute_matrix_profile(row)
#         matrix_profiles.append(mp)
#         motif, discord = find_motif_discord(mp, idx)
#         motifs.append(motif)
#         discords.append(discord)
#     motifs_array = np.array(motifs)
#     discords_array = np.array(discords)
#     return matrix_profiles, motifs_array, discords_array

# # Computing the matrix profiles for all the correlation matrices we have and saving them in a folder
# for method in methods:
#     method_dir = f'ADNI_full/corr_matrices/corr_matrix_{method}/'
#     for file in os.listdir(method_dir):
#         corr_matrix = np.loadtxt(method_dir + file, delimiter=',')
#         matrix_profiles, motifs, discords = matrix_profile(corr_matrix)
#         np.savetxt(f'ADNI_full/matrix_profiles/matrix_profile_{method}/{file}', matrix_profiles, delimiter=',')
#         np.savetxt(f'ADNI_full/motifs/motifs_{method}/{file}', motifs, delimiter=',')
#         np.savetxt(f'ADNI_full/discords/discords_{method}/{file}', discords, delimiter=',')

# Defining a class to preprocess raw data into a format suitable for training Graph Neural Networks (GNNs).
## With the possibility of assigning weight to edges, adding the age feature, sex feature, and matrixe profiling.

class Raw_to_Graph(InMemoryDataset):
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
        corr_matrices = full_corr_path_lists[self.method]
        for patient_idx, patient_matrix in enumerate(corr_matrices):
            path = f'ADNI_full/corr_matrices/corr_matrix_{self.method}/{patient_matrix}'
            corr_matrix = pd.read_csv(path, header=None).values
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
            x_array = [degree_array_norm, between_central_array_norm, local_eff_array_norm, cluster_coeff_array_norm, ratio_local_global_array_norm]
            
            if self.age:
                # Extracting the age feature of the patient
                patient_age = ages[patient_idx]
                age_norm = (patient_age - min_age) / (max_age - min_age)
                # Making the age array the same size as the other arrays
                age_array = np.full((nbr_ROIs,), age_norm)
                x_array.append(age_array)
            if self.sex:
                # Extracting the sex feature of the patient
                patient_sex = int(sex[patient_idx])
                # Making the sex array the same size as the other arrays
                sex_array = np.full((nbr_ROIs,), patient_sex)
                x_array.append(sex_array)

            if self.matrixprofile:
                motifs_array = np.loadtxt(f'ADNI_full/motifs/motifs_{method}/{patient_matrix}', delimiter=',')
                discords_array = np.loadtxt(f'ADNI_full/discords/discords_{method}/{patient_matrix}', delimiter=',')
                x_array.append(motifs_array)
                x_array.append(discords_array)

            # Concatenate the degree, participation coefficient, betweenness centrality, local efficiency, and ratio of local to global efficiency arrays to form a single feature vector
            x_conc = torch.tensor(np.concatenate(x_array), dtype=torch.float)
            # Determining the number of features concatenated to reshape with the correct dimensions
            nbr_features = len(x_array)
            x = torch.reshape(x_conc, (nbr_features, nbr_ROIs)).T

            # Create a Pytorch Geometric Data object from the NetworkX 
            graph_data = from_networkx(NetworkX_graph)
            ## The feature matrix of the graph is the degree, betweenness centrality, local efficiency, clustering coefficient and ratio of local to global efficiency of each node
            graph_data.x = x
            ## The target/output variable that we want to predict is the diagnostic label of the patient
            graph_data.y = diagnostic_label[patient_idx]
            graphs.append(graph_data)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

def dataset_features_and_stats(dataset):
    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Weighted: {dataset.weight}')
    print(f'Threshold: {dataset.threshold}')
    print(f'Correlation Method: {dataset.method}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {len(np.unique(diagnostic_label))}')

    # Getting the first graph object in the dataset.
    data = dataset[0]

    print()
    print(data)
    print('=============================================================')

    # Some statistics about the first graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')

# Testing the class Raw_to_Graph with one example and saving it
threshold = 0.4
weight = False
age = False
sex = False
matrixprofile = False
nbr_features = 5 + int(age) + int(sex) + int(matrixprofile)
method = 'pearson'

root = f'Raw_to_graph/ADNI_T_{threshold}_M_{method}_W{weight}_A{age}_S{sex}_MP{matrixprofile}'
dataset = Raw_to_Graph(root=root, threshold=threshold, method=method, weight=weight)
dataset_features_and_stats(dataset)


# In[4]:


# Creating the train, validation and test sets
X = dataset
y = dataset.data.y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.125, random_state=42)
nbr_classes = len(np.unique(y))

print(f'Number of training graphs: {len(X_train)}')
print(f'Number of validation graphs: {len(X_valid)}')
print(f'Number of test graphs: {len(X_test)}')
print(f'Number of classes: {nbr_classes}')

train_loader = DataLoader(X_train, batch_size=16, shuffle=True)
valid_loader = DataLoader(X_valid, batch_size=len(X_valid), shuffle=True)
test_loader = DataLoader(X_test, batch_size=len(X_test), shuffle=False)


# In[5]:


# The function we are using to compute the accuracy of our model
def quick_accuracy(y_hat, y):
  """
  Args :
    y_hat : logits predicted by model [n, num_classes]
    y : ground trutch labels [n]
  returns :
    average accuracy
  """
  n = y.shape[0]
  y_hat = torch.argmax(y_hat, dim=-1)
  accuracy = (y_hat==y).sum().data.item()
  return accuracy/n


# In[9]:


# Training the model
def train(model, optimizer, criterion, train_loader, valid_loader, parameters, test_loader=False, testing=False, n_epochs=100):
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0
        for data in train_loader:
            # Converting each element of data.y to a float
            target = torch.tensor([float(y) for y in data.y], dtype=torch.long)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.batch)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            train_loss += criterion(out, target)
            train_accuracy += quick_accuracy(out, target)

        train_losses.append(train_loss.detach().numpy()/len(train_loader))
        train_accuracies.append(train_accuracy/len(train_loader))

        model.eval()
        valid_loss = 0
        valid_accuracy = 0
        with torch.no_grad():
            for data in valid_loader:
                # Converting each element of data.y to a float
                target = torch.tensor([float(y) for y in data.y], dtype=torch.long)
                out = model(data.x, data.edge_index, data.batch)
                valid_loss += criterion(out, target)
                valid_accuracy += quick_accuracy(out, target)

            valid_losses.append(valid_loss.detach().numpy()/len(valid_loader))
            valid_accuracies.append(valid_accuracy/len(valid_loader))

            if testing:
                test_loss = 0
                test_accuracy = 0
                for data in test_loader:
                    # Converting each element of data.y to a long
                    target = torch.tensor([float(y) for y in data.y], dtype=torch.long)
                    out = model(data.x, data.edge_index, data.batch)
                    test_loss += criterion(out, target)
                    test_accuracy += quick_accuracy(out, target)

                test_losses.append(test_loss.detach().numpy()/len(test_loader.dataset))
                test_accuracies.append(test_accuracy/len(test_loader.dataset))

        if testing:
            print(f'Epoch {epoch+1}/{n_epochs}')
            print(f'Train Loss: {train_losses[-1]:.4f}, Validation Loss: {valid_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')
            print(f'Train Accuracy: {train_accuracies[-1]:.4f}, Validation Accuracy: {valid_accuracies[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}')
        else:
            print(f'Epoch {epoch+1}/{n_epochs}')
            print(f'Train Loss: {train_losses[-1]:.4f}, Validation Loss: {valid_losses[-1]:.4f}')
            print(f'Train Accuracy: {train_accuracies[-1]:.4f}, Validation Accuracy: {valid_accuracies[-1]:.4f}')
    
    plt.figure(figsize=(12, 5))

    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label=f'Train Loss')
    plt.plot(valid_losses, label=f'Validation Loss')
    if testing:
        plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label=f'Train Accuracy')
    plt.plot(valid_accuracies, label=f'Validation Accuracy')
    if testing:
        plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save the plot
    lr = parameters[0]
    hidden_channels = parameters[1]
    num_layers = parameters[2]
    dropout = parameters[3]
    heads = parameters[4]
    filename = f'GAT_Models/lr{lr}_hc{hidden_channels}_nl{num_layers}_d{dropout}_epochs{n_epochs}_heads{heads}.png'
    plt.savefig(filename)
    plt.show()

    if testing:
        return train_losses, train_accuracies, valid_losses, valid_accuracies, test_losses, test_accuracies
    else:
        return train_losses, train_accuracies, valid_losses, valid_accuracies


# In[8]:


# Defining the model, optimizer and loss function
lr=0.00001
hidden_channels=32
dropout=0.2
num_layers=3
heads=2
parameters = [lr, hidden_channels, num_layers, dropout, heads]

model = GAT(in_channels=nbr_features, hidden_channels=parameters[1], out_channels=nbr_classes, num_layers=parameters[2], dropout=parameters[3], heads=parameters[4], nbr_classes=nbr_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=parameters[0])
criterion = torch.nn.CrossEntropyLoss()

# Printing the model architecture
print(model)

# Running the training
train_losses, train_accuracies, valid_losses, valid_accuracies = train(model, optimizer, criterion, train_loader, valid_loader, parameters, n_epochs=1000)


# In[11]:


# Doing some parameter gridsearch to find the best hyperparameters
from sklearn.model_selection import ParameterGrid

param_grid = {
    'learning_rate': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
    'hidden_channels': [32, 64, 128],
    'num_layers': [1, 2, 3],
    'dropout_rate': [0.0, 0.1, 0.2, 0.3],
    'heads': [2, 3, 4, 5]
}


# Create combinations of hyperparameters
param_combinations = ParameterGrid(param_grid)
n_epochs = 500
# Train using each combination
for params in param_combinations:
    filename = f'GAT_Models/lr{params["learning_rate"]}_hc{params["hidden_channels"]}_nl{params["num_layers"]}_d{params["dropout_rate"]}_epochs{n_epochs}_heads{params["heads"]}.png'
    if os.path.exists(filename):
        pass
    else:
        parameters = [params['learning_rate'], params['hidden_channels'], params['num_layers'], params['dropout_rate'], params['heads']]
        model = GAT(in_channels=nbr_features, hidden_channels=parameters[1], out_channels=nbr_classes, num_layers=parameters[2], dropout=parameters[3], heads=parameters[4], nbr_classes=nbr_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters[0])
        criterion = torch.nn.CrossEntropyLoss()
        train_losses, train_accuracies, valid_losses, valid_accuracies = train(model, optimizer, criterion, train_loader, valid_loader, parameters, n_epochs=n_epochs)

