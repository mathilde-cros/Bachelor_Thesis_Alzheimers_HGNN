import torch
import numpy as np
import pandas as pd
import pickle as pkl
import os

from nilearn import datasets

from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split

from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx

import networkx as nx
from networkx.convert_matrix import from_numpy_array

# Generating the brain atlas
def gen_atlas_labels():
    atlas = datasets.fetch_atlas_aal()
    atlas_filename = atlas.maps
    atlas_labels = atlas.labels
    n_ROIs = len(atlas_labels)
    return atlas, atlas_filename, atlas_labels, n_ROIs

# Loading the time series
def load_time_series():
    root = 'ADNI_full/time_series'
    time_series_list = os.listdir(root)
    time_series=[]
    for i in time_series_list:
        time_series_sub = np.loadtxt(os.path.join(root, i), delimiter=',')
        time_series.append(time_series_sub)
    return time_series

# Make the list of time series into a list of df
def list_of_df_of_time_series(time_series_list):
    time_series_df_list = []
    for ts in time_series_list:
        df = pd.DataFrame(ts)
        time_series_df_list.append(df)
    return time_series_df_list

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

# Creating the train, validation and test sets
def create_train_test_valid(dataset):
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
    valid_loader = DataLoader(X_valid, batch_size=len(X_valid), shuffle=False)
    test_loader = DataLoader(X_test, batch_size=len(X_test), shuffle=False)

    return train_loader, valid_loader, test_loader, nbr_classes

# To convert a dictionnary into a numpy array
def dict_to_array(dict):
    array = np.array(list(dict.values()))
    return array

# To normalize an array
def normalize_array(array):
    norm_array = (array - np.mean(array)) / np.std(array)
    return norm_array

# printing a few features and statistics about the dataset
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

# Creating a dictionary of lists of paths to the correlation matrices for each method. Each list in the dictionary represents a different method.
def corr_matrix_paths():
    methods = ['pearson', 'spearman', 'kendall', 'partial']
    full_corr_path_lists = {}
    for method in methods:
        method_dir = f'ADNI_full/corr_matrices/corr_matrix_{method}/'
        full_corr_path_lists[method] = []
        for file in os.listdir(method_dir):
            full_corr_path_lists[method].append(file)
    return full_corr_path_lists

# Generating the diagnostic file from the diagnostic_label.csv file
diagnostic_label = np.loadtxt('ADNI_full/diagnostic_label.csv', dtype=str, delimiter=',')

# Combining the 'EMCI', 'LMCI' and 'MCI' diagnostics into a single 'MCI' label for simplicity, then one-hot encoding the diagnostics
def combine_diag_labels(diagnostic_label):
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
        full_corr_path_lists = corr_matrix_paths()
        diagnostic_label = combine_diag_labels(diagnostic_label)
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
            x_array = np.stack([degree_array_norm, between_central_array_norm, local_eff_array_norm, cluster_coeff_array_norm, ratio_local_global_array_norm], axis=-1)
            x_array = x_array.astype(np.float32)
            print(x_array.shape)

            if self.age:
                # Extracting the age feature of the patient
                patient_age = ages[patient_idx]
                age_norm = (patient_age - min_age) / (max_age - min_age)
                # Making the age array the same size as the other arrays
                age_array = np.full((nbr_ROIs,), age_norm)
                x_array = np.concatenate((x_array, age_array), axis=-1)
            if self.sex:
                # Extracting the sex feature of the patient
                patient_sex = int(sex[patient_idx])
                # Making the sex array the same size as the other arrays
                sex_array = np.full((nbr_ROIs,), patient_sex)
                x_array = np.concatenate((x_array, sex_array), axis=-1)
            
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
            print(x.shape)

            # Create a Pytorch Geometric Data object from the NetworkX
            graph_data = from_networkx(NetworkX_graph)
            ## The feature matrix of the graph is the degree, betweenness centrality, local efficiency, clustering coefficient and ratio of local to global efficiency of each node
            graph_data.x = x
            ## The target/output variable that we want to predict is the diagnostic label of the patient
            graph_data.y = float(diagnostic_label[patient_idx])
            graphs.append(graph_data)

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

# Taken from https://github.com/pyg-team/pytorch_geometric/blob/af0f5f44d4dfa8accd898df3e8523d06b67940b0/torch_geometric/nn/conv/hypergraph_conv.py#L15     
from typing import Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax

class HypergraphConv(MessagePassing):
    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{D}^{-1} \mathbf{H} \mathbf{W}
        \mathbf{B}^{-1} \mathbf{H}^{\top} \mathbf{X} \mathbf{\Theta}

    where :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` is the incidence
    matrix, :math:`\mathbf{W} \in \mathbb{R}^M` is the diagonal hyperedge
    weight matrix, and
    :math:`\mathbf{D}` and :math:`\mathbf{B}` are the corresponding degree
    matrices.

    For example, in the hypergraph scenario
    :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` with
    :math:`\mathcal{V} = \{ 0, 1, 2, 3 \}` and
    :math:`\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3 \} \}`, the
    :obj:`hyperedge_index` is represented as:

    .. code-block:: python

        hyperedge_index = torch.tensor([
            [0, 1, 2, 1, 2, 3],
            [0, 0, 0, 1, 1, 1],
        ])

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          hyperedge indices :math:`(|\mathcal{V}|, |\mathcal{E}|)`,
          hyperedge weights :math:`(|\mathcal{E}|)` *(optional)*
          hyperedge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(in_channels, out_channels, bias=False,
                              weight_initializer='glorot')

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (torch.Tensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (torch.Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (torch.Tensor, optional): Hyperedge feature matrix
                in :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D,
                             alpha=alpha, size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out
    

# Training the models base function
def epochs_training(model, optimizer, criterion, train_loader, valid_loader, test_loader, testing, train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy,test_losses=None, test_accuracies=None):

    model.train()
    train_loss = 0
    train_accuracy = 0
    for data in train_loader:
        target = data.y.clone().detach().long()
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
            target = data.y.clone().detach().long()
            out = model(data.x, data.edge_index, data.batch)
            valid_loss += criterion(out, target)
            valid_accuracy += quick_accuracy(out, target)

        valid_losses.append(valid_loss.detach().numpy()/len(valid_loader))
        valid_accuracies.append(valid_accuracy/len(valid_loader))
        if valid_accuracies[-1] > max_valid_accuracy:
            max_valid_accuracy = valid_accuracies[-1]
            
        if testing:
            test_loss = 0
            test_accuracy = 0
            for data in test_loader:
                target = data.y.clone().detach().long()
                out = model(data.x, data.edge_index, data.batch)
                test_loss += criterion(out, target)
                test_accuracy += quick_accuracy(out, target)

            test_losses.append(test_loss.detach().numpy()/len(test_loader.dataset))
            test_accuracies.append(test_accuracy/len(test_loader.dataset))
            return train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy, test_losses, test_accuracies
        else:
            return train_losses, train_accuracies, valid_losses, valid_accuracies, max_valid_accuracy