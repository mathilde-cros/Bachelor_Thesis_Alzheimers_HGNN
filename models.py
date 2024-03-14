import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import pdb
import wandb

import torch
import torch.nn.functional as F
from torch.nn import ReLU, Linear, Sequential, ModuleList
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.nn import HypergraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, degree

from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

import hypernetx as hnx
import networkx as nx
from networkx.convert_matrix import from_numpy_array

from sklearn.neighbors import NearestNeighbors

import functions as f
from functions import dict_to_array, normalize_array

weight = True
matrixprofile = True

# Raw_to_graph class but for 2-class classification
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

# Convolutional Graph Neural Network
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, nbr_classes):
        super(GCN, self).__init__()
        self.nbr_classes = nbr_classes
        self.convs = ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        # self.double_mlp = Sequential(Linear(hidden_channels * heads, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.mlp = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # xs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
                # xs.append(x)
        x = global_mean_pool(x, batch)
        # x = torch.cat(xs, dim=1)
        # x = self.double_mlp(x)
        x = self.mlp(x)
        return x
    
# Training the GCN model
def train_GCN(model, optimizer, criterion, w_decay, threshold, train_loader, valid_loader, parameters, test_loader=False, testing=False, n_epochs=100):
    test_loader = test_loader
    testing = testing
    n_epochs = n_epochs

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    max_valid_accuracy = 0
    test_accuracy = 0

    # start a new wandb run to track this script
    run = wandb.init(
        # set the wandb project where this run will be logged
        project = "Alzheimers_GNN",
        # track hyperparameters and run metadata
        config = {
        "architecture": "GCN_2Class",
        "weights": weight,
        "weight_decay": w_decay,
        "threshold": threshold,
        "matrix profiling": matrixprofile,
        "learning_rate": parameters[0],
        "hidden_channels": parameters[1],
        "num_layers": parameters[2],
        "dropout": parameters[3],
        "epochs": n_epochs},)

    for epoch in range(n_epochs):
        if testing:
            train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy, test_accuracy = f.epochs_training(model, optimizer, criterion, train_loader, valid_loader, test_loader, testing, test_accuracy, train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy)
            wandb.log({"Train Loss": train_losses[-1], "Train Accuracy": train_accuracies[-1], "Validation Loss": valid_losses[-1], "Validation Accuracy": valid_accuracies[-1], "Max Valid Accuracy": max_valid_accuracy, "Test Accuracy": test_accuracy})
        else:
            train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy = f.epochs_training(model, optimizer, criterion, train_loader, valid_loader, test_loader, testing, test_accuracy, train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy)
            wandb.log({"Train Loss": train_losses[-1], "Train Accuracy": train_accuracies[-1], "Validation Loss": valid_losses[-1], "Validation Accuracy": valid_accuracies[-1], "Max Valid Accuracy": max_valid_accuracy})
        print(f'Epoch {epoch+1}/{n_epochs}')
        print(f'Train Loss: {train_losses[-1]:.4f}, Validation Loss: {valid_losses[-1]:.4f}')
        print(f'Train Accuracy: {train_accuracies[-1]:.4f}, Validation Accuracy: {valid_accuracies[-1]:.4f}')
        print(f'Max Validation Accuracy: {max_valid_accuracy:.4f}')

    if testing:
        print('Test Accuracy:', test_accuracy)

    plt.figure(figsize=(12, 5))

    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label=f'Train Loss')
    plt.plot(valid_losses, label=f'Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label=f'Train Accuracy')
    plt.plot(valid_accuracies, label=f'Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Save the plot
    lr = parameters[0]
    hidden_channels = parameters[1]
    num_layers = parameters[2]
    dropout = parameters[3]
    if matrixprofile:
        filename = f'2Class_Models/GCN_Models_MP/threshold_{threshold}/lr{lr}_hc{hidden_channels}_nl{num_layers}_d{dropout}_epochs{n_epochs}_wdecay{w_decay}_w{weight}.png'
    else:
        filename = f'2Class_Models/GCN_Models/threshold_{threshold}/lr{lr}_hc{hidden_channels}_nl{num_layers}_d{dropout}_epochs{n_epochs}_wdecay{w_decay}_w{weight}.png'
    plt.savefig(filename)
    if testing:
        plt.title(f'Test Accuracy: {test_accuracy}')
    plt.show()

    # wandb.finish()

    if testing:
        return train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy, test_accuracy
    else:
        return train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy

# Graph Attention Network
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, heads, nbr_classes):
        super(GAT, self).__init__()
        self.nbr_classes = nbr_classes
        self.convs = ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads))
        # self.double_mlp = Sequential(Linear(hidden_channels * heads, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.mlp = Sequential(Linear(hidden_channels * heads, hidden_channels), ReLU(), Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # xs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.elu(x)  # Using ELU activation for GAT
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
                # xs.append(x)
        x = global_mean_pool(x, batch)
        # x = torch.cat(xs, dim=1)
        # x = self.double_mlp(x)
        x = self.mlp(x)
        return x
    
# Training the GAT model
def train_GAT(model, optimizer, criterion, w_decay, threshold, train_loader, valid_loader, parameters, test_loader=False, testing=False, n_epochs=100):
    test_loader = test_loader
    testing = testing
    n_epochs = n_epochs

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    max_valid_accuracy = 0
    test_accuracy = 0

    # start a new wandb run to track this script
    run = wandb.init(
        # set the wandb project where this run will be logged
        project = "Alzheimers_GNN",
        # track hyperparameters and run metadata
        config = {
        "architecture": "GAT_2Class",
        "weights": weight,
        "weight_decay": w_decay,
        "threshold": threshold,
        "matrix profiling": matrixprofile,
        "learning_rate": parameters[0],
        "hidden_channels": parameters[1],
        "num_layers": parameters[2],
        "dropout": parameters[3],
        "heads": parameters[4],
        "epochs": n_epochs},)

    for epoch in range(n_epochs):
        if testing:
            train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy, test_accuracy = f.epochs_training(model, optimizer, criterion, train_loader, valid_loader, test_loader, testing, test_accuracy, train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy)
            wandb.log({"Train Loss": train_losses[-1], "Train Accuracy": train_accuracies[-1], "Validation Loss": valid_losses[-1], "Validation Accuracy": valid_accuracies[-1], "Max Valid Accuracy": max_valid_accuracy, "Test Accuracy": test_accuracy})
        else:
            train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy = f.epochs_training(model, optimizer, criterion, train_loader, valid_loader, test_loader, testing, test_accuracy, train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy)
            wandb.log({"Train Loss": train_losses[-1], "Train Accuracy": train_accuracies[-1], "Validation Loss": valid_losses[-1], "Validation Accuracy": valid_accuracies[-1], "Max Valid Accuracy": max_valid_accuracy})
        print(f'Epoch {epoch+1}/{n_epochs}')
        print(f'Train Loss: {train_losses[-1]:.4f}, Validation Loss: {valid_losses[-1]:.4f}')
        print(f'Train Accuracy: {train_accuracies[-1]:.4f}, Validation Accuracy: {valid_accuracies[-1]:.4f}')
        print(f'Max Validation Accuracy: {max_valid_accuracy:.4f}')

    if testing:
        print('Test Accuracy:', test_accuracy)

    plt.figure(figsize=(12, 5))

    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label=f'Train Loss')
    plt.plot(valid_losses, label=f'Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label=f'Train Accuracy')
    plt.plot(valid_accuracies, label=f'Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Save the plot
    lr = parameters[0]
    hidden_channels = parameters[1]
    num_layers = parameters[2]
    dropout = parameters[3]
    heads = parameters[4]
    if matrixprofile:
        filename = f'2Class_Models/GAT_Models_MP/threshold_{threshold}/lr{lr}_hc{hidden_channels}_nl{num_layers}_d{dropout}_epochs{n_epochs}_heads{heads}_wdecay{w_decay}_w{weight}.png'
    else:
        filename = f'2Class_Models/GAT_Models/threshold_{threshold}/lr{lr}_hc{hidden_channels}_nl{num_layers}_d{dropout}_epochs{n_epochs}_heads{heads}_wdecay{w_decay}_w{weight}.png'
    plt.savefig(filename)
    if testing:
        plt.title(f'Test Accuracy: {test_accuracy}')
    plt.show()

    wandb.finish()

    if testing:
        return train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy, test_accuracy
    else:
        return train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy

# Message Passing Neural Network
class MPNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, nbr_classes):
        super(MPNN, self).__init__()
        self.nbr_classes = nbr_classes
        self.convs = ModuleList()
        self.convs.append(MPNNLayer(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(MPNNLayer(hidden_channels, hidden_channels))
        # self.double_mlp = Sequential(Linear(hidden_channels * heads, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.mlp = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # xs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
                # xs.append(x)
        x = global_mean_pool(x, batch)
        # x = torch.cat(xs, dim=1)
        # x = self.double_mlp(x)
        x = self.mlp(x)
        return x

class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNLayer, self).__init__(aggr='mean')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out
    
# Training the MPNN model
def train_MPNN(model, optimizer, criterion, w_decay, threshold, train_loader, valid_loader, parameters, test_loader=False, testing=False, n_epochs=100):
    test_loader = test_loader
    testing = testing
    n_epochs = n_epochs

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    max_valid_accuracy = 0
    test_accuracy = 0

    # start a new wandb run to track this script
    run = wandb.init(
        # set the wandb project where this run will be logged
        project = "Alzheimers_GNN",
        # track hyperparameters and run metadata
        config = {
        "architecture": "MPNN_2Class",
        "weights": weight,
        "weight_decay": w_decay,
        "threshold": threshold,
        "matrix profiling": matrixprofile,
        "learning_rate": parameters[0],
        "hidden_channels": parameters[1],
        "num_layers": parameters[2],
        "dropout": parameters[3],
        "epochs": n_epochs},)

    for epoch in range(n_epochs):
        if testing:
            train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy, test_accuracy = f.epochs_training(model, optimizer, criterion, train_loader, valid_loader, test_loader, testing, test_accuracy, train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy)
            wandb.log({"Train Loss": train_losses[-1], "Train Accuracy": train_accuracies[-1], "Validation Loss": valid_losses[-1], "Validation Accuracy": valid_accuracies[-1], "Max Valid Accuracy": max_valid_accuracy, "Test Accuracy": test_accuracy})
        else:
            train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy = f.epochs_training(model, optimizer, criterion, train_loader, valid_loader, test_loader, testing, test_accuracy, train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy)
            wandb.log({"Train Loss": train_losses[-1], "Train Accuracy": train_accuracies[-1], "Validation Loss": valid_losses[-1], "Validation Accuracy": valid_accuracies[-1], "Max Valid Accuracy": max_valid_accuracy})
        print(f'Epoch {epoch+1}/{n_epochs}')
        print(f'Train Loss: {train_losses[-1]:.4f}, Validation Loss: {valid_losses[-1]:.4f}')
        print(f'Train Accuracy: {train_accuracies[-1]:.4f}, Validation Accuracy: {valid_accuracies[-1]:.4f}')
        print(f'Max Validation Accuracy: {max_valid_accuracy:.4f}')

    if testing:
        print('Test Accuracy:', test_accuracy)

    plt.figure(figsize=(12, 5))

    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label=f'Train Loss')
    plt.plot(valid_losses, label=f'Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label=f'Train Accuracy')
    plt.plot(valid_accuracies, label=f'Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Save the plot
    lr = parameters[0]
    hidden_channels = parameters[1]
    num_layers = parameters[2]
    dropout = parameters[3]
    if matrixprofile:
        filename = f'2Class_Models/MPNN_Models_MP/threshold_{threshold}/lr{lr}_hc{hidden_channels}_nl{num_layers}_d{dropout}_epochs{n_epochs}_wdecay{w_decay}_w{weight}.png'
    else:
        filename = f'2Class_Models/MPNN_Models/threshold_{threshold}/lr{lr}_hc{hidden_channels}_nl{num_layers}_d{dropout}_epochs{n_epochs}_wdecay{w_decay}_w{weight}.png'
    plt.savefig(filename)
    if testing:
        plt.title(f'Test Accuracy: {test_accuracy}')
    plt.show()

    wandb.finish()

    if testing:
        return train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy, test_accuracy
    else:
        return train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy

# Convolutional Hypergraph Networks
class HGConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, nbr_classes):
        super(HGConv, self).__init__()
        self.nbr_classes = nbr_classes
        self.convs = ModuleList()
        self.convs.append(f.HypergraphConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(f.HypergraphConv(hidden_channels, hidden_channels))
        # self.double_mlp = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.mlp = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # xs = []
        for conv in self.convs:
            print(edge_index)
            x = conv(x, edge_index)
            x = F.relu(x)
            if self.dropout > 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
                # xs.append(x)
        x = global_mean_pool(x, batch)
        # x = torch.cat(xs, dim=1)
        # x = self.double_mlp(x)
        x = self.mlp(x)
        return x

# Training the HGCN model
def train_HGConv(model, optimizer, criterion, w_decay, threshold, train_loader, valid_loader, parameters, architecture, test_loader=False, testing=False, n_epochs=100):
    test_loader = test_loader
    testing = testing
    n_epochs = n_epochs

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    test_losses = []
    test_accuracies = []
    max_valid_accuracy = 0

    # start a new wandb run to track this script
    run = wandb.init(
        # set the wandb project where this run will be logged
        project = "Alzheimers_GNN",
        # track hyperparameters and run metadata
        config = {
        "architecture": architecture,
        "weights": weight,
        "weight_decay": w_decay,
        "threshold": threshold,
        "matrix profiling": True,
        "learning_rate": parameters[0],
        "hidden_channels": parameters[1],
        "num_layers": parameters[2],
        "dropout": parameters[3],
        "epochs": n_epochs},)

    for epoch in range(n_epochs):
        if testing:
            train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy, test_losses, test_accuracies = f.epochs_training(model, optimizer, criterion, train_loader, valid_loader, test_loader, testing, train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy, test_losses, test_accuracies)
            print(f'Epoch {epoch+1}/{n_epochs}')
            print(f'Train Loss: {train_losses[-1]:.4f}, Validation Loss: {valid_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}')
            print(f'Train Accuracy: {train_accuracies[-1]:.4f}, Validation Accuracy: {valid_accuracies[-1]:.4f}, Test Accuracy: {test_accuracies[-1]:.4f}')
            print(f'Max Validation Accuracy: {max_valid_accuracy:.4f}')
            wandb.log({"Train Loss": train_losses[-1], "Train Accuracy": train_accuracies[-1], "Validation Loss": valid_losses[-1], "Validation Accuracy": valid_accuracies[-1], "Max Valid Accuracy": max_valid_accuracy, "Test Loss": test_losses[-1], "Test Accuracy": test_accuracies[-1]})
        else:
            train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy = f.epochs_training(model, optimizer, criterion, train_loader, valid_loader, test_loader, testing, train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy)
            print(f'Epoch {epoch+1}/{n_epochs}')
            print(f'Train Loss: {train_losses[-1]:.4f}, Validation Loss: {valid_losses[-1]:.4f}')
            print(f'Train Accuracy: {train_accuracies[-1]:.4f}, Validation Accuracy: {valid_accuracies[-1]:.4f}')
            print(f'Max Validation Accuracy: {max_valid_accuracy:.4f}')
            wandb.log({"Train Loss": train_losses[-1], "Train Accuracy": train_accuracies[-1], "Validation Loss": valid_losses[-1], "Validation Accuracy": valid_accuracies[-1], "Max Valid Accuracy": max_valid_accuracy})

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
    filename = f'2Class_Models/HGConv_Models_MP/threshold_{threshold}/lr{lr}_hc{hidden_channels}_nl{num_layers}_d{dropout}_epochs{n_epochs}_wdecay{w_decay}_w{weight}.png'
    plt.savefig(filename)
    plt.show()

    wandb.finish()

    if testing:
        return train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy, test_losses, test_accuracies
    else:
        return train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy

# Hypergraph Building
def display_hypergraph(hypergraph):
    # Displaying the hypergraph
    n_nodes = len(hypergraph.nodes)
    n_edges = len(hypergraph.edges)
    print("Hypergraph number of nodes:", n_nodes)
    print("Hypergraph number of edges:", n_edges)
    print(hypergraph.shape)

    plt.subplots(figsize=(10,10))
    hnx.draw(hypergraph, with_edge_labels=False)
    plt.show()

def remove_duplicate_hyperedges(hypergraph_dict):
    unique_hyperedges = {}
    for hyperedge, vertices in hypergraph_dict.items():
        sorted_vertices = sorted(vertices)
        sorted_vertices_tuple = tuple(sorted_vertices)
        if sorted_vertices_tuple not in unique_hyperedges.values():
            unique_hyperedges[hyperedge] = sorted_vertices_tuple
    # print("Dictionary representation of the hypergraph:")
    # print(unique_hyperedges)
    return unique_hyperedges

# Fourier Clustering
def adjacency_matrix_to_hypergraph_dict(adjacency_matrix):
    hypergraph_dict = {}
    for i, row in enumerate(adjacency_matrix):
        hyperedge = []
        for j, val in enumerate(row):
            if val == 1:
                hyperedge.append(j)
        hypergraph_dict[i] = hyperedge
    return hypergraph_dict

def generate_adjacency_from_cluster_indices(cluster_indices):
    n_ROIs = len(cluster_indices)
    adjacency = np.zeros((n_ROIs, n_ROIs))
    for row_idx in range(n_ROIs):
        for col_idx in range(n_ROIs):
            if cluster_indices[row_idx] == cluster_indices[col_idx]:
                adjacency[row_idx, col_idx] = 1
    return adjacency

def generate_hypergraph_from_cluster(correlation_matrix, threshold, display=False):
    # Transforming correlation matrix into dissimilarity matrix
    dissimilarity_matrix = 1 - np.abs(correlation_matrix)
    dissimilarity_matrix[np.isnan(dissimilarity_matrix)] = 0
    # Performing Fourier clustering
    linkage_matrix = linkage(squareform(dissimilarity_matrix), method='complete')
    cluster_indices = fcluster(linkage_matrix, threshold, criterion='distance')
    # Generating hypergraph from the cluster indices
    adjacency = generate_adjacency_from_cluster_indices(cluster_indices)
    hg_dict = adjacency_matrix_to_hypergraph_dict(adjacency)
    # Reducing the amount of hyperedges by removing duplicated hyperedges
    hg_dict = remove_duplicate_hyperedges(hg_dict)
    # Creating HyperNetX hypergraph
    hg = hnx.Hypergraph(hg_dict)
    if display:
        display_hypergraph(hg)
    return hg, hg_dict

# Maximal Cliques
def max_cliques(graph):
    cliques = list(nx.find_cliques(graph))
    max_clique_size = max(len(c) for c in cliques)
    summ = sum(1 for c in cliques)
    print(f"Number of cliques: {summ}")
    print(f"Maximum clique size: {max_clique_size}")
    return cliques

def graph_to_hypergraph_max_cliques(graph, display=False):
    cliques = max_cliques(graph)
    hg_dict = {i: set(clique) for i, clique in enumerate(cliques) if len(clique) > 1}
    # hg_dict = {i: set(clique) for i, clique in enumerate(cliques)}
    hg_dict = remove_duplicate_hyperedges(hg_dict)
    hg = hnx.Hypergraph(hg_dict)
    if display:
        display_hypergraph(hg)
    return hg, hg_dict

# Coskewness Cube
def adjacency_cube_to_hypergraph_dict(adjacency_cube):
    hg_dict = {}
    hyperedge_nbr = 0
    for i, row in enumerate(adjacency_cube):
        for j, col in enumerate(row):
            for k, val in enumerate(col):
                if val == 1:
                    if (i != j) and (i != k) and (j != k):
                        hg_dict[hyperedge_nbr] = [i, j, k]
                        hyperedge_nbr += 1
    return hg_dict

def coskewness_cube_to_hypergraph(coskewness_cube, threshold, display=False):
    adjacency_cube = np.zeros_like(coskewness_cube)
    for i in range(len(coskewness_cube)):
        for j in range(len(coskewness_cube[i])):
            for k in range(len(coskewness_cube[i][j])):
                if np.abs(coskewness_cube[i][j][k]) > threshold:
                    adjacency_cube[i][j][k] = 1
    hg_dict = adjacency_cube_to_hypergraph_dict(adjacency_cube)
    hg_dict = remove_duplicate_hyperedges(hg_dict)
    hg = hnx.Hypergraph(hg_dict)
    if display:
        display_hypergraph(hg)
    return hg, hg_dict

# K-Nearest Neighbours Clustering
def generate_adjacency_from_knn(X, k_neighbors=3):
    # Finding k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    n_samples = X.shape[0]
    adjacency = np.zeros((n_samples, n_samples))
    # Constructing adjacency matrix based on k-nearest neighbors
    for i in range(n_samples):
        for j in indices[i]:
            adjacency[i, j] = 1
            adjacency[j, i] = 1
    return adjacency

def adjacency_matrix_to_hypergraph_dict(adjacency_matrix):
    hypergraph_dict = {}
    for i, row in enumerate(adjacency_matrix):
        hyperedge = []
        for j, val in enumerate(row):
            if val == 1:
                hyperedge.append(j)
        hypergraph_dict[i] = hyperedge
    return hypergraph_dict

def generate_hypergraph_from_knn(data_matrix, k_neighbors, display=False):
    # Generating adjacency matrix from k-nearest neighbors
    adjacency = generate_adjacency_from_knn(data_matrix, k_neighbors)
    # Generating hypergraph from the adjacency matrix
    hg_dict = adjacency_matrix_to_hypergraph_dict(adjacency)
    # Removing duplicate hyperedges (optional)
    hg_dict = remove_duplicate_hyperedges(hg_dict)
    # Creating HyperNetX hypergraph
    hg = hnx.Hypergraph(hg_dict)
    if display:
        display_hypergraph(hg)
    return hg, hg_dict

# Defining a function to save hypergraphs
def save_hypergraph(hg_dict, directory, method, threshold, id):
    dir = f'{directory}/{method}/thresh_{threshold}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(f'{dir}/{id}_{method}_{threshold}.pkl', 'wb') as f:
        pkl.dump(hg_dict, f)
    return

# Saving all the built hypergraphs
def save_all_hypergraphs(method_list, time_series_folder, corr_matrix_list):
    for i, file_name in enumerate(os.listdir(time_series_folder)):
        patient_id = file_name[3:13]
        print(f'Processing patient {patient_id}')
        for method in method_list:
            if method == 'fourier_cluster':
                patient_matrix = corr_matrix_list[i]
                _, hg_dict = generate_hypergraph_from_cluster(patient_matrix, threshold=threshold)
            elif method == 'maximal_clique':
                method_corr = 'pearson'
                root = f'Raw_to_graph_2Class/ADNI_T_{threshold}_M_{method_corr}_WFalse_AFalse_SFalse_MPTrue'
                dataset = Raw_to_Graph_2Class(root=root, threshold=threshold, method=method_corr, weight=False, age=False, sex=False, matrixprofile=True)
                graph = f.r2g_to_nx(dataset[i])
                _, hg_dict = graph_to_hypergraph_max_cliques(graph)
            elif method == 'coskewness':
                filename = f'ADNI_full/coskew_matrices/patient_{patient_id}.pkl'
                coskew_cube = pkl.load(open(filename, 'rb'))
                _, hg_dict = coskewness_cube_to_hypergraph(coskew_cube, threshold=threshold)
            elif method == 'knn':
                k_neighbors = 3
                patient_matrix = corr_matrix_list[i]
                _, hg_dict = generate_hypergraph_from_knn(patient_matrix, k_neighbors)
                threshold = f'{k_neighbors}neighbors'
            save_hypergraph(hg_dict, 'Hypergraphs', method, threshold, patient_id)
            print(f'Patient {patient_id} processed and saved for the {method} method with threshold {threshold}')


