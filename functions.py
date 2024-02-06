import os
from nilearn import datasets, plotting, image
from nilearn.maskers import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.interfaces.fmriprep import load_confounds
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
def filter_file(data_path, file_type):
    preprocessed_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file_type in file:
                preprocessed_list.append(os.path.join(root, file))
    preprocessed_list.sort()
    return preprocessed_list

def filter_SMC_patient_info():
    df          = pd.read_csv('/Users/georgepulickal/Documents/ADNI_FULL/patient_info.csv')
    labels      = df['Research Group']
    label_idx_list = [i for i in range(len(labels)) if labels[i] != 'SMC']
    return label_idx_list

def load_patient_info(data):
    df          = pd.read_csv('/Users/georgepulickal/Documents/ADNI_FULL/patient_info.csv')
    info        = df[data]
    info_arr    = info.to_numpy()
    return info_arr

def filter_group(group):
    df = pd.read_csv('/Users/georgepulickal/Documents/ADNI_FULL/patient_info.csv')
    labels = df['Research Group']
    label_idx_list = [i for i in range(len(labels)) if labels[i] == group]
    return label_idx_list


def corr_gsr(time_series_gsr):
    correlation_measure = ConnectivityMeasure(kind='correlation')
    corr_gsr = []
    for i in range(len(time_series_gsr)):
        corr_gsr_sub = correlation_measure.fit_transform([time_series_gsr[i]])[0]
        np.fill_diagonal(corr_gsr_sub, 0)
        corr_gsr.append(corr_gsr_sub)
    return corr_gsr

def plot_corr(corr_mtrx):
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_mtrx, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

def filter_list(full_list):
    idx = filter_SMC_patient_info()
    new_list = full_list[idx]
    return new_list

def save_mtrx(mtrx_list , id_list, dir, tag):
    for i in range(len(mtrx_list)):
        np.savetxt(f'{dir}/{tag}_{id_list[i]}.csv', mtrx_list[i], delimiter=',')

def load_time_series():
    root = 'ADNI_full/time_series'
    time_series_list = sorted(os.listdir(root))
    time_series_gsr=[]
    for i in range(len(time_series_list)):
        time_series_sub = np.loadtxt(os.path.join(root, time_series_list[i]), delimiter=',')
        time_series_gsr.append(time_series_sub)
    return time_series_gsr

def df_pcorr(time_series_gsr):
    corr_gsr = []
    for i in range(len(time_series_gsr)):
        df = pd.DataFrame(time_series_gsr[i])
        corr_df = df.corr(method='pearson')
        corr_arr = corr_df.to_numpy()
        corr_arr = np.nan_to_num(corr_arr, nan=0)
        np.fill_diagonal(corr_arr, 0)
        corr_gsr.append(corr_arr)
    return corr_gsr

def header_t_r(img):
    n1_img = nib.load(img)
    header = n1_img.header
    t_r=header['pixdim'][4]
    return t_r

def load_pcorr():
    root = 'ADNI_gsr_full/pcorr_matrices'
    time_series_list = sorted(os.listdir(root))
    time_series_gsr=[]
    for i in range(len(time_series_list)):
        time_series_sub = np.loadtxt(os.path.join(root, time_series_list[i]), delimiter=',')
        time_series_gsr.append(time_series_sub)
    return time_series_gsr

def plot_loss_acc(loss , acc , epoche_no):
    x = np.arange(0 , epoche_no, 1)