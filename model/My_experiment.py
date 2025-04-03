
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
df = df[df["COHORT"].isin(["PD", "Prodromal"])]

# Step 4: Drop any PATNO who has missing values in any of those columns
def is_patient_valid(group):
    return (len(group) == 3) and (not group[cog_cols].isnull().any().any())

df = df.groupby("PATNO").filter(is_patient_valid)

# Step 5: Add subtype column
df['subtype'] = df['COHORT'].apply(lambda x: 0 if x == 'PD' else 1 if x == 'Prodromal' else np.nan)

# Step 6: Sort and add obs_time
df = df.sort_values(['PATNO', 'AGE_AT_VISIT']).reset_index(drop=True)
df['obs_time'] = df.groupby('PATNO').cumcount() + 1

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
data = filtered_df

max_visits = 3
shuffle    = True
num_output_dims = data.shape[1] - 4
train_data_loader, train_data_dict, _, _, test_data_loader, test_data_dict, valid_pid, test_pid, unique_pid = parse_data(
            data.values, max_visits=max_visits, test_per=0.2, valid_per=0.2, shuffle=shuffle, device='cpu')
data_loader, collect_dict, unique_pid = parse_data(
            data.values, max_visits=max_visits, device="cpu")

b_vae, C, d_s, d_h, d_rnn, reg_type, lr = 0.01, 0.0, 10, 10, 20, 'l1', 0.1
epochs = 50000
device = 'cpu'
max_delta = 5.
learn_time = True


model = Sublign(d_s, d_h, d_rnn, C=C, dim_biomarkers=num_output_dims,
                sigmoid=True, reg_type=reg_type, auto_delta=True,
                max_delta=max_delta, learn_time=learn_time)

# Fit model
model.fit(
    train_data_loader,
    test_data_loader,
    epochs,
    lr=lr,
    verbose=True,
    fname='C:/Users/nss_1/clustering-interval-censored/model/runs/ppmi.pt',
    eval_freq=25
)


# Evaluate model
results = model.score(train_data_dict, test_data_dict)
print('PPMI Test ARI: %.3f' % results['ari'])

# Extract results
subtypes = model.get_subtypes_datadict(collect_dict)
labels = model.get_labels(collect_dict)
deltas = model.get_deltas(collect_dict)


# Save
pickle.dump((labels, deltas, subtypes), open('C:/Users/nss_1/clustering-interval-censored/model/runs/ppmi_icml.pk', 'wb'))