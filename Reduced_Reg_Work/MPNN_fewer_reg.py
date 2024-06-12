#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import wandb

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_mean_pool

import functions as f


# In[2]:


os.environ['WANDB_NOTEBOOK_NAME']="MPNN_fewer_reg.ipynb"


# - MPNNLayer class defines a single message passing layer, inheriting from torch_geometric.nn.MessagePassing. It applies a linear transformation to the input node features and then performs message passing.
# - forward method of MPNNLayer adds self-loops to the adjacency matrix, linearly transforms the input node features, and then starts propagating messages using the propagate method.
# - message method of MPNNLayer normalizes node features and multiplies them with the corresponding adjacency weights.
# - update method of MPNNLayer returns the new node embeddings after aggregation.
# - MPNN class serves as a container for multiple MPNNLayer instances, similar to the GCN class. It iterates over the layers, applies them sequentially, and then applies a final MLP for prediction.

# In[3]:


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


# In[4]:


# Testing the class Raw_to_Graph_reduced_reg with one example and saving it
threshold = 0.5
weight = False
age = False
sex = False
matrixprofile = True
if matrixprofile:
    in_channels = 461 + int(age) + int(sex)
else:
    in_channels = 5 + int(age) + int(sex)
method = 'pearson'

root = f'Raw_to_graph_reduced_reg/ADNI_T_{threshold}_M_{method}_W{weight}_A{age}_S{sex}_MP{matrixprofile}'
dataset = f.Raw_to_Graph_reduced_reg(root=root, threshold=threshold, method=method, weight=weight, sex=sex, age=age, matrixprofile=matrixprofile)
f.dataset_features_and_stats(dataset)


# # In[5]:


# Creating the train, validation and test sets
train_loader, valid_loader, test_loader, nbr_classes, y_train = f.create_train_test_valid(dataset)


# In[6]:

stratify = True
# Training the model
def train(model, optimizer, criterion, w_decay, threshold, train_loader, valid_loader, parameters, test_loader=False, testing=False, n_epochs=100):
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
        "architecture": "MPNN_fewer_reg",
        "strat + w loss": stratify,
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
        filename = f'Reduced_reg_models/MPNN_Models_MP/threshold_{threshold}/lr{lr}_hc{hidden_channels}_nl{num_layers}_d{dropout}_epochs{n_epochs}_wdecay{w_decay}_w{weight}.png'
    else:
        filename = f'Reduced_reg_models/MPNN_Models/threshold_{threshold}/lr{lr}_hc{hidden_channels}_nl{num_layers}_d{dropout}_epochs{n_epochs}_wdecay{w_decay}_w{weight}.png'
    plt.savefig(filename)
    if testing:
        plt.title(f'Test Accuracy: {test_accuracy}')
    plt.show()

    wandb.finish()

    if testing:
        return train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy, test_accuracy
    else:
        return train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy


# In[7]:

# Example Model

threshold = 0.6
age = False
sex = False
matrixprofile = True
weight = False
if matrixprofile:
    in_channels = 461 + int(age) + int(sex)
else:
    in_channels = 5 + int(age) + int(sex)
method = 'pearson'

root = f'Raw_to_graph_reduced_reg/ADNI_T_{threshold}_M_{method}_W{weight}_A{age}_S{sex}_MP{matrixprofile}'
dataset = f.Raw_to_Graph_reduced_reg(root=root, threshold=threshold, method=method, weight=weight, sex=sex, age=age, matrixprofile=matrixprofile)
f.dataset_features_and_stats(dataset)
# Creating the train, validation and test sets
train_loader, valid_loader, test_loader, nbr_classes, y_train = f.create_train_test_valid(dataset)

# Defining the model, optimizer and loss function
lr=0.00001
hidden_channels=32
num_layers=3
dropout=0.2
w_decay=0
parameters = [lr, hidden_channels, num_layers, dropout]

model = MPNN(in_channels=in_channels, hidden_channels=parameters[1], out_channels=nbr_classes, num_layers=parameters[2], dropout=parameters[3], nbr_classes=nbr_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=parameters[0], weight_decay=w_decay)
criterion = torch.nn.CrossEntropyLoss()

# Printing the model architecture
print(model)

# Running the training
train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy = train(model, optimizer, criterion, w_decay, threshold, train_loader, valid_loader, parameters, n_epochs=2750)


# In[8]:


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

root = f'Raw_to_graph_reduced_reg/ADNI_T_{threshold}_M_{method}_W{weight}_A{age}_S{sex}_MP{matrixprofile}'
dataset = f.Raw_to_Graph_reduced_reg(root=root, threshold=threshold, method=method, weight=weight, sex=sex, age=age, matrixprofile=matrixprofile)
f.dataset_features_and_stats(dataset)

# Creating the train, validation and test sets
train_loader, valid_loader, test_loader, nbr_classes, y_train = f.create_train_test_valid(dataset, stratify)

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
n_epochs = 1000
testing = True
# Train using each combination
for params in param_combinations:
    if matrixprofile:
        filename = f'Reduced_reg_models/MPNN_Models_MP/threshold_{threshold}/lr{params["learning_rate"]}_hc{params["hidden_channels"]}_nl{params["num_layers"]}_d{params["dropout_rate"]}_epochs{n_epochs}_wdecay{params["weight_decay"]}_w{weight}.png'
    else:
        filename = f'Reduced_reg_models/MPNN_Models/threshold_{threshold}/lr{params["learning_rate"]}_hc{params["hidden_channels"]}_nl{params["num_layers"]}_d{params["dropout_rate"]}_epochs{n_epochs}_wdecay{params["weight_decay"]}_w{weight}.png'
    if os.path.exists(filename):
        pass
    else:
        parameters = [params['learning_rate'], params['hidden_channels'], params['num_layers'], params['dropout_rate']]
        model = MPNN(in_channels=in_channels, hidden_channels=parameters[1], out_channels=nbr_classes, num_layers=parameters[2], dropout=parameters[3], nbr_classes=nbr_classes)
        if stratify:
            diag_lab = [0 , 1 , 2, 3]
            class_freq = []
            for i in diag_lab:
                class_freq.append(np.count_nonzero(torch.Tensor(y_train) == i))
            class_freq = torch.FloatTensor(class_freq)
            class_weights = 1 / class_freq
            class_weights /= class_weights.sum()
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = torch.nn.CrossEntropyLoss() 
        if 'weight_decay' not in params.keys():
            w_decay = 0
        else:
            w_decay = params['weight_decay']
        optimizer = torch.optim.Adam(model.parameters(), lr=parameters[0], weight_decay=w_decay)
        train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy, test_accuracy = train(model, optimizer, criterion, w_decay, threshold, train_loader, valid_loader, parameters, test_loader, testing, n_epochs=2750)
