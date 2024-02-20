# Bachelor-Thesis-Alzheimers-HGNN
In this repository, you will find the entirety of my code contribution for my Bachelor's Thesis Project, in the Computer Science and Technology Department of the University of Cambridge, on Graph and Hyper-graph Neural Networks for Alzheimer’s Disease Detection.

## In the ADNI_full folder
You will find:
- A "time_series" folder, where you will find the time_series data for each patient. Each .csv file represents 1 patient, and each line is a brain region. This data was generated using the brain atlas and fMRI images from the Alzheimer's Disease Neuroimaging Initiative (ADNI). The ADNI data were downloaded from the ADNI database (adni.loni.usc.edu). This was done previously to my work.
- A "corr_matrices" folder, in which subfolders of correlation matrices can be found, depending on the correlation computation used (Pearson, Spearman Rank, Kendall Rank, Partial). Similarly as in the "time_series" folder, each .csv file represents 1 patient, and each line/column of the matrix is a brain region.
- A patient_info.csv file, where the Subject ID, Sex, Research Group, Age, Modality and Description features for each patient can be found. Each line represents a patient. The variable "Research Group" is the variable that we want to predict using our graph and hypergraph neural networks. This data was also acquired previously to my work. The different "Research Group" labels that can be associated to a patient are:
    - CN: Cognitively Normal
    - SMC: Subjective Memory Complaint
    - MCI: Mild Cognitive Impairment; that can also be subdivided into:
        - EMCI: Early Mild Cognitive Impairment
        - LMCI: Late Mild Cognitive Impairment
    - AD: Alzheimer's disease
- A sex.csv file where the column for the patient's sex/gender is extracted from the patient_info.csv file.
- A age.csv file where the column for the patient's age is extracted from the patient_info.csv file.
- A diagnostic_label.csv file where the column for the patient's diagnostic (Research Group) is extracted from the patient_info.csv file.

## In the Corr_and_df.ipynb file
This notebook file focuses on preprocessing time series data, computing various correlation measures between brain regions, and exploring differences in correlation matrices based on different correlation methods.
Before running the notebook, make sure that 'os', 'numpy', 'pandas', 'matplotlib', 'nilearn' and 'seaborn' are installed.
This notebook is useful for:
- Data Preparation: fetching a predefined brain atlas from Nilearn and loading time series data from multiple patients from the ADNI_full folder.
- Preprocessing: converting the time series data into a dataframes for easier manipulation.
- Correlation Computation: implementing 4 correlation methods: Pearson correlation, Spearman rank correlation, Kendall rank correlation, and partial correlation. Then correlation matrices are generated for each method.
- Visualization: plotting and display of heatmaps of some correlation matrices for each method.
- Comparison: quantification of differences between correlation matrices using the Frobenius norm. Then the dissimilarity score between matrices was printed for comparison.
- Saving Results: saving all the correlation matrices as CSV files for each method.
- Dataframe Creation: constructing a dataframe containing the headers 'Patient_id', 'Age', 'Sex', 'Time_series', 'Corr_matrix_pearson', 'Corr_matrix_spearman', 'Corr_matrix_kendall', 'Corr_matrix_partial', 'Diagnostic_label'. This is built combining the information from the patient_info.csv file (the Research Group was renamed Diagnostic_label for simplicity), the correlation matrices computed (for each method) and the time series data acquired.
- Data Exploration: display of basic features and distributions (diagnostic, age and sex) of the dataframe and an example time series data for the first patient is shown.

## In the Graph_Building.ipynb file
This notebook file focuses on transforming the preprocessed neuroimaging data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset (from notebook "Corr_and_df.ipynb") into graphs, suitable for training Graph Neural Networks (GNNs). The notebook includes code for converting correlation matrices into graphs, extracting relevant features, and organizing the data for training.
Before running the notebook, make sure that 'os', 'numpy', 'pandas', 'torch', 'torch_geometric' and 'networkx' are installed.
This notebook is useful for:
- Data Preparation: building upon the data files provided in the "Corr_and_df.ipnyb" file, such as the corr_matrices folder, the age.csv, sex.csv and diagnostic_label.csv files.
- Preprocessing: Combining diagnostic labels into simplified categories, loading additional patient features (age and sex), and defining functions for data normalization and conversion.
- Graph Construction: Definition of the Raw_to_Graph class to convert correlation matrices into graph representations. These graphs are PyTorch Geometric Data objects. Various parameters such as threshold, correlation method (see "Corr_and_df.ipynb"), and edge weighting can be adjusted to customize graph construction. The graphs also have the common node features: degree, betweenness centrality, local efficiency, clustering coefficient, and ratio of local to global efficiency.
- Dataset Analysis: Analyzing the characteristics of the generated graph dataset, including the number of graphs, features, classes, and basic statistics of individual graphs.
- Parameter Exploration: Exploring combinations of parameters (thresholds, correlation methods, and edge weighting) using loops for generating multiple datasets.
- Saving Results: Saving all the generated graphs in the "Raw_to_Graph" folder (see section below).

## In the Raw_to_Graph folder 
Contains subfolders where the graphs made in the "Graph_Building.ipynb" notebook are saved according to the different parameters for their construction.
The folders are named in the following way: 'ADNI_T_{threshold}_W_{weight}_M_{method}'.

## In the Matrix_Profiling.ipynb file
This notebook file focuses on the same task as the "Graph_Building.ipynb" file except that it incorporates additional time series features using the Matrix Profile technique.
Before running the notebook, make sure that all the modules necessary for the "Graph_Building.ipynb" file are installed, and furthermore 'matrixprofile' and 'stumpy'.
The only differences between this notebook and "Graph_Building.ipynb" are:
- The Raw_to_Graph_MatrixProfile Class, which additionally to the Raw_to_Graph class, incorporates time series features using the Matrix Profile technique.
- The compute_matrix_profile function, which computes the matrix profile for each row (time series) of the correlation matrix.
- The find_motif_discord function, which finds the motif and discord in the matrix profile.
- The process function also has slight modifications to adjust for the incorporation of time series features.
- All the generated graphs are saved in the "Raw_to_Graph_MatrixProfile" folder, named in the same way as in the Raw_to_Graph folder (see section above).
