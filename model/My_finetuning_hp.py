
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
df = pd.read_csv('C:/Users/nss_1/clustering-interval-censored/model/data/result_4_long_format.csv', sep=';')

# Step 2: Filter to only V04, V06, V08
required_visits = ["V04", "V06", "V08"]
cog_cols = ["MCATOT", "NP1RTOT", "NP2PTOT", "NP3TOT", "SDMTOTAL"]
selected_columns = ["PATNO", "AGE_AT_VISIT", "FINAL_SEX_ENCODED", "COHORT", "subtype", "obs_time"] + cog_cols + ["EVENT_ID"]
import numpy as np

# Suppose cog_cols = ['A','B',…]
df[cog_cols] = df[cog_cols].replace(0, np.nan)
df = df.dropna(subset=cog_cols)
df = df[df["COHORT"].isin(["PD", "Prodromal"])]
df['subtype'] = df['COHORT'].apply(lambda x: 0 if x == 'PD' else 1 if x == 'Prodromal' else np.nan)
visit_encoding = {visit: i+1 for i, visit in enumerate(required_visits)}

# Assign obs_time using the mapping
df['obs_time'] = df['EVENT_ID'].map(visit_encoding)
df = df.sort_values(['PATNO', 'AGE_AT_VISIT']).reset_index(drop=True)
event_counts = df.groupby("PATNO")["EVENT_ID"].apply(set)

# Keep only IDs where all required visits are present
valid_ids = event_counts[event_counts.apply(lambda x: set(required_visits).issubset(x))].index

# Filter the DataFrame to keep only those IDs and the selected columns
df_filtered = df[df["PATNO"].isin(valid_ids) & df["EVENT_ID"].isin(required_visits)][selected_columns]


# Step 4: Drop any PATNO who has missing values in any of those columns
#def is_patient_valid(group):
    #return (len(group) == 3) and (not group[cog_cols].isnull().any().any())

#df = df.groupby("PATNO").filter(is_patient_valid)


# Step 7: Normalize
for col in cog_cols:
    max_val = df_filtered[col].max()
    if pd.notnull(max_val) and max_val != 0:
        df_filtered[col] = df_filtered[col] / max_val

# Final formatting
filtered_df = df_filtered[[
    "MCATOT", "NP1RTOT", "NP2PTOT", "NP3TOT", "SDMTOTAL",
    "subtype",
    "AGE_AT_VISIT",
    "PATNO",
    "obs_time"
]]
print(filtered_df["subtype"].value_counts())

def run_model(MODEL, hp, lr, train_loader, train_data_dict, valid_loader, valid_data_dict, device, epochs = 1000, eval_freq = 25, ppmi=True, chf=False, fname=None, search='nelbo', anneal=False):
#     print ('Running for ',epochs, ' epochs w/ evfreq',eval_freq)
    model = MODEL(**hp)
    model.to(device)
    
#     import pdb; pdb.set_trace()
    model.fit(train_loader, valid_loader, epochs, lr, anneal=anneal, eval_freq=eval_freq, fname=fname)
    results = model.score(train_data_dict,valid_data_dict, K=2)
    
    sublign_mse = results['mse']
    sublign_ari = results['ari']
    #sublign_swaps = results['swaps']
    #sublign_pear = results['pear']
    
    train_Y = train_data_dict['Y_collect']
    train_X = train_data_dict['obs_t_collect']
    
    test_Y = valid_data_dict['Y_collect']
    test_S = valid_data_dict['s_collect']
    test_X = valid_data_dict['obs_t_collect']
    test_M = valid_data_dict['mask_collect']
    
    test_X_torch = torch.tensor(test_X).to(device)
    test_Y_torch = torch.tensor(test_Y).to(device)
    test_M_torch = torch.tensor(test_M).to(device)
        
    train_z, _ = model.get_mu(train_X, train_Y)
    test_z, _  = model.get_mu(test_X, test_Y)
    
    #align_metrics = np.mean([1-sublign_swaps, sublign_pear])
    (nelbo, nll, kl), reg = model.forward(test_Y_torch, None, test_X_torch, test_M_torch, None)
    
    nelbo_metric  = nelbo
    #cheat_metric  = - np.mean([sublign_ari, align_metrics])
    
    # We are MINIMIZING over metrics
    if search == 'mse':
        final_metric = sublign_mse
    #elif search == 'cheat':
        #final_metric = cheat_metric
    elif search == 'nelbo':
        final_metric = nelbo_metric
    
    all_metrics = {
        'nelbo': nelbo_metric,
        #'cheat': cheat_metric,
        'nll': nll,
        'kl': kl,
        'ari': sublign_ari,
        'mse': sublign_mse,
        #'pear': sublign_pear,
        #'swaps': sublign_swaps
    }
    #all_metrics = (nll, nelbo_metric, kl)#, kmeans_metric) #cheat_metric)
    return final_metric, all_metrics


def get_unsup_results(train_loader, train_data_dict, valid_loader, test_data_dict, device, model, epochs, sigmoid, ppmi=True, chf=False, fname=None, search='nelbo'):
    """
    We are looking for the LOWEST metric over all the parameter searches. Make sure metrics are tuned accordingly!
    
    """
    hp = {}
    best_perf, best_mse, best_ari= np.inf, np.inf, np.inf
    best_config = None
    all_results = {}
    
    MODEL = None
    if model == 'sublign':
        MODEL = Sublign
    else:
        NotImplemented()
    print (MODEL)
                 
        
#     anneal, beta, C, ds, dh, drnn, reg_type, lr = True, 0.001, 0.0, 5, 200, 200, 'l2', 0.001
        
    if model =='sublign':
        for beta in [1.]:
            for anneal in [False]:    
                for C in [0., 0.1, 1.]:
                    for ds in [10]:
                        for dim_h in [50]:
                            for dim_rnn in [200]:
                                for reg_type in ['l1']:
                                    for lr in [1e-2, 1e-1]:                                        
                                        hp['C']              = C
                                        hp['beta']           = beta
                                        hp['dim_hidden']     = dim_h
                                        hp['dim_stochastic'] = ds
                                        hp['dim_rnn']        = dim_rnn
                                        hp['reg_type']       = reg_type
                                        hp['sigmoid']        = sigmoid
#                                         hp['eval_freq']      = 10
                                        hp['dim_biomarkers'] = 3 if sigmoid else 1
                                        if ppmi:
                                            hp['dim_biomarkers'] = 5
                                            
#                                         try:
                                        perf, all_metrics = run_model(MODEL, hp, lr, train_loader, train_data_dict, valid_loader, 
                                                                          test_data_dict, device, epochs = epochs, ppmi=ppmi, 
                                                                          eval_freq=1, fname=fname, search=search, anneal=anneal)
#                                         print('MSE %.3f, ARI: %.3f' % (all_metrics['mse'], all_metrics['ari']))
#                                         except:
#                                             import pdb; pdb.set_trace()
#                                         perf, all_metrics, mse, ari, swaps, pear = 10000., None, 0., 0., 0., 0.
                                        print((anneal, C, beta, dim_h, ds, dim_rnn, reg_type, lr))
                                        all_results[(anneal, beta, C, dim_h, ds, dim_rnn, reg_type, lr)] = all_metrics
                                        if perf<best_perf:

#                                             try:
                                            best_perf = perf; best_ari = all_metrics['ari']; best_config = (anneal, beta, C, dim_h, ds, dim_rnn, reg_type, lr)
# #                                             except:
#                                                 import pdb; pdb.set_trace()
    return best_perf, best_ari, best_config, all_results

def run_cv(model='sublign', dataset='sigmoid', epochs=10, ppmi=True, search='nelbo'):
    import pickle
    import torch

    chf = False  # Fix missing variable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on:', device)

    # Load dataset
    if ppmi:
        use_sigmoid = True
        data = filtered_df  # You manually prepared this earlier
        max_visits = 3
        train_loader, train_data_dict, valid_loader, valid_dict, p_ids, full_p_ids = parse_data(
            data.values, max_visits=max_visits, test_per=0.2
        )
        fname = 'runs/ppmi_hptune.pt'
    else:
        raise ValueError("Currently only ppmi=True is supported.")

    # Run hyperparameter search
    best_perf, best_ari, best_config, all_results = get_unsup_results(
        train_loader, train_data_dict, valid_loader, valid_dict,
        device, model, epochs=epochs, sigmoid=use_sigmoid,
        ppmi=ppmi, chf=chf, fname=fname, search=search
    )

    print(best_config, 'Best ARI: %.3f' % best_ari)

    # Save results
    fname = f"runs/{model}_ppmi.pkl"
    with open(fname, 'wb') as f:
        print('dumped pickle')
        pickle.dump(all_results, f)
        
run_cv(model = 'sublign', dataset='sigmoid', epochs=1000, ppmi=True, search='nelbo') #, ppmi=True, search='nelbo')