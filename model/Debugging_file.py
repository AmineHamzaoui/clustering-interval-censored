
import matplotlib.pyplot as plt

import numpy as np
import pickle

def clean_plot():
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)    
    
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()   
    plt.grid()

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (10,6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

import os, sys
project_root = os.path.abspath(os.path.dirname(os.getcwd()))
sys.path.append(project_root)
import sys
import torch

sys.path.append('.')
from data.load import chf
from data.data_utils import parse_data
from data.synthetic_data import load_piecewise_synthetic_data


sys.path.append('./model')
from models import Sublign
from run_experiments import get_hyperparameters
import sys
import os

# Go up one directory from 'model/' to the project root
root_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path, 'data'))
sys.path.append(os.path.join(root_path, 'model'))
sys.path.append(os.path.join(root_path, 'cross_validation'))
import numpy as np
import torch
import sys, os
from data.load import sigmoid, quadratic, load_data_format, parkinsons
from data.load import chf as load_chf
from data.data_utils import parse_data
from model.models import Sublign
import pickle

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

"""
import pandas as pd 
data= pd.read_csv('C:/Users/nss_1/clustering-interval-censored/model/data/result_4_long_format.csv',sep=';')
required_visits = ["V04", "V06", "V08"]
cog_cols = ["MCATOT", "NP1RTOT", "NP2PTOT", "NP3TOT", "SDMTOTAL"]
filtered_df = data[data["EVENT_ID"].isin(required_visits)]

# Group by PATNO and filter out any patients with missing values in those columns
def is_patient_valid(group):
    # Ensure all 3 visits are there
    if len(group) != 3:
        return False
    # Check for any NaNs in the target columns
    return not group[cog_cols].isnull().any().any()

# Apply the filter
filtered_df = data.groupby("PATNO").filter(is_patient_valid)
selected_columns = ["PATNO","AGE_AT_VISIT", "FINAL_SEX_ENCODED", "MCATOT","NP1RTOT" ,"NP2PTOT", "NP3TOT", "SDMTOTAL"]
filtered_df = data[data["COHORT"].isin(["PD", "Prodromal"])][selected_columns]
filtered_df['subtype'] = data['COHORT'].apply(lambda x: 1 if x in ['PD', 'Prodromal'] else 0)
filtered_df = filtered_df.sort_values(['PATNO', 'AGE_AT_VISIT']).reset_index(drop=True)
# Your list of required visits and cognitive columns



filtered_df['obs_time'] = pd.Series(filtered_df.groupby('PATNO').cumcount().values + 1, index=filtered_df.index)
cols_to_rescale = ["MCATOT", "NP1RTOT", "NP2PTOT", "NP3TOT", "SDMTOTAL"]

for col in cols_to_rescale:
    max_val = filtered_df[col].max()
    if pd.notnull(max_val) and max_val != 0:
        filtered_df[col] = filtered_df[col] / max_val
        
from run_experiments import get_hyperparameters_ppmi
b_vae, C, d_s, d_h, d_rnn, reg_type, lr = get_hyperparameters_ppmi()
filtered_df = filtered_df.groupby('PATNO').filter(lambda x: len(x) == 9)
filtered_df = filtered_df[[
    "MCATOT", "NP1RTOT", "NP2PTOT", "NP3TOT", "SDMTOTAL",  # features
    "subtype",                  # -4
    "AGE_AT_VISIT",            # time
    "PATNO",                   # patient ID
    "obs_time"                 # relative time
]]
data       = filtered_df"""
import pandas as pd 

# Step 1: Load data
data = pd.read_csv('C:/Users/nss_1/clustering-interval-censored/model/data/result_4_long_format.csv', sep=';')

# Step 2: Filter to only V04, V06, V08
required_visits = ["V04", "V06", "V08"]
cog_cols = ["MCATOT", "NP1RTOT", "NP2PTOT", "NP3TOT", "SDMTOTAL"]
selected_columns = ["PATNO", "AGE_AT_VISIT", "FINAL_SEX_ENCODED", "COHORT"] + cog_cols + ["EVENT_ID"]

df = data[data["EVENT_ID"].isin(required_visits)][selected_columns]

# Step 3: Keep only 'PD' and 'Prodromal'
df = df[df["COHORT"].isin(["PD", "Healthy Control"])]
# First, find all IDs with any NaN in cog_cols
ids_with_nan = df[df[cog_cols].isna().any(axis=1)]['PATNO'].unique()

# Then, remove all rows for those IDs
df = df[~df['PATNO'].isin(ids_with_nan)]

# Step 4: Drop any PATNO who has missing values in any of those columns
#def is_patient_valid(group):
    #return (len(group) == 3) and (not group[cog_cols].isnull().any().any())

#df = df.groupby("PATNO").filter(is_patient_valid)

# Step 5: Add subtype column
df['subtype'] = df['COHORT'].apply(lambda x: 0 if x == 'PD' else 1 if x == 'Healthy Control' else np.nan)

# Step 6: Sort and add obs_time
df = df.sort_values(['PATNO', 'AGE_AT_VISIT']).reset_index(drop=True)
df['obs_time'] = df.groupby('PATNO').cumcount() + 1
# Step 1: Count occurrences of each ID
id_counts = df['PATNO'].value_counts()

# Step 2: Filter out IDs that occur exactly 1 or 2 times
ids_to_drop = id_counts[id_counts.isin([1, 2])].index

# Step 3: Drop all rows where ID is in the list
df = df[~df['PATNO'].isin(ids_to_drop)]
df.to_csv('C:/Users/nss_1/clustering-interval-censored/model/data/filtered_data.csv', sep=';', index=False)
# Step 7: Normalize
for col in cog_cols:
    max_val = df[col].max()
    if pd.notnull(max_val) and max_val != 0:
        df[col] = df[col] / max_val

# Final formatting
filtered_df = df[[
    "MCATOT", "NP1RTOT", "NP2PTOT", "NP3TOT", "SDMTOTAL",
    "subtype",
    "AGE_AT_VISIT",
    "PATNO",
    "obs_time"
]]
print(filtered_df["subtype"].value_counts())
# Save or pass to model
df_filtered = filtered_df
