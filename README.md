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
This notebook file focuses on