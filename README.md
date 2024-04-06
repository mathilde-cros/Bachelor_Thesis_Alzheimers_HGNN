# Bachelor-Thesis-Alzheimers-HGNN
In this repository, you will find the entirety of my code contribution for my Bachelor's Thesis Project, in the Computer Science and Technology Department of the University of Cambridge, on Graph and Hyper-graph Neural Networks for Alzheimer’s Disease Detection.

## Setting up the project's environment
The environment.yml and requirements.txt files are provided in order to set up a working environment for the project. 

More detail about the folders/files in this project:

## The ADNI_full folder
You will find the time series data for each patient as well as csv fils of their information. 
Folders with the computer correlation matrices, matrix profiles and coskewness matrices can also be found in this folder. 

## The Corr_and_df.ipynb file
This notebook file focuses on preprocessing time series data, computing various correlation measures between brain regions, and exploring differences in correlation matrices based on different correlation methods.

## The Graph_Building.ipynb and HyperGraph_Building.ipynb files
This notebook file focuses on transforming the preprocessed neuroimaging data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset (from notebook "Corr_and_df.ipynb") into graphs, suitable for training Graph Neural Networks (GNNs). The notebook includes code for converting correlation matrices into graphs, extracting relevant features, and organizing the data for training. Respectively for the HyperGraph_Building.ipynb file with hypergraphs and the appropriate methods implemented.
In this notebook, we save all the generated graphs in the "Raw_to_Graph", "Raw_to_hypergraph" and "hypergraphs" folders respectively in the following way or similarly: 'ADNI_T_{threshold}_W_{weight}_M_{method}'. The Graph_Building_MP_Colab.ipynb notebook has the relevant code to compute the matrix profiles on Google Colab (for resources) and saving them for local use on our graphs and hypergraphs.

## The GCN, MPNN, GAT and 2C_GCN, 2C_GAT, 2C_MPNN python and .ipynb files
These files are the implementation of the graph neural networks for 4 class and 2 class (2C) classification. Slight version updates differences might occur between the ipynb files (old version) and the python files (newer version that we ran and modified over ssh).
These results are then saved in their respective names' folders.

## The HGCK, HGFC, HGKNN, HGMC, HGConv and 2CHGCK, 2C_HGFC, 2C_HGKNN, 2C_HGMC python and .ipynb files
These files are the implementation of the hypergraph neural networks for 4 class and 2 class (2C) classification. CK = Coskewness method; FC = Fourier Cluster method, KNN = knn method, MC = Maximal Clique method. Slight version updates differences might occur between the ipynb files (old version) and the python files (newer version that we ran and modified over ssh).
These results are then saved in their respective names' folders.
NOTE: Such folders and the saved raw_to_hypergraph/graph data from the ran code aren't saved here for memory purposes so these folders are mostly empty on purpose and the code needs to be re-ran.

## The Fake_Data and Fake_Data4C notebooks
Such notebooks contain the information in the discussion part of the report where we built a new fake dataset for both 2-way and 4-way classification of Alzheimer's disease. After running our models, they were again saved in their respective names' folder with the prefixe "Fake" in order to distinguish our results.

## The functions.py and models.py files
These files gather commonly used functions accross all notebooks and python files (functions.py), as well as training, display and definition functions for the models (models.py).
