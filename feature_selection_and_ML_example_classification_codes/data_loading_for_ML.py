#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 12:19:07 2025

@author: Ali Daher
"""

import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind
import pickle as pkl

train_path = "train_z_scores.ods"
test_path  = "test_z_scores.ods"
mean_std_path = "train_means_sds.ods"

train_sheets = pd.ExcelFile(train_path, engine="odf").sheet_names
test_sheets  = pd.ExcelFile(test_path,  engine="odf").sheet_names
print("Train sheets:", train_sheets)
print("Test sheets:", test_sheets)

metadata = pd.read_csv("metadata.csv")   # adjust filename
# Make sure `sample` column matches column names in train/test matrices
metadata.head()


def select_features(z_scored_tr_data, mean_vector, std_vector, conditions_label, FDR_threshold= 0.1, K = 1000):
    
    conditions = conditions_label.unique()
    features_selected = set()
    
    un_z_scored_data = z_scored_tr_data.mul(std_vector, axis=1).add(mean_vector, axis=1)
    un_z_scored_data.index
    for cond in conditions:
        condition_samples = conditions_label[conditions_label==cond].index
        non_condition_samples = conditions_label[conditions_label!=cond].index
               
        p_values = []
        log2fc_values = []
        for gene in un_z_scored_data.columns:
            cond_values = un_z_scored_data.loc[condition_samples, gene]
            noncond_values = un_z_scored_data.loc[non_condition_samples, gene]
            diff = cond_values.mean() - noncond_values.mean() #approximately log2FC
            stat, pval = ttest_ind(cond_values, noncond_values, equal_var=False)
            log2fc_values.append(diff)
            p_values.append(pval)
        reject, pvals_corrected, _, _ = multipletests(p_values, alpha=FDR_threshold, method='fdr_bh') #across genes
        filtered_genes = un_z_scored_data.columns[reject]
        filtered_log2fc = np.array(log2fc_values)[reject]

        if len(filtered_genes) == 0:
            continue

        # Get indices of top K genes by absolute effect size
        top_k_indices = np.argsort(np.abs(filtered_log2fc))[-K:]
        top_genes = filtered_genes[top_k_indices]

        features_selected.update(top_genes)

    return list(features_selected)
    
        
        
    
    

folds = {}

for b in train_sheets:

    print(f"Loading LOSO fold: {b}")

    # ---- Load TRAIN data
    train_df = pd.read_excel(train_path, sheet_name=b, engine="odf")
    mean_std = pd.read_excel(mean_std_path, sheet_name=b, engine="odf").set_index("gene")
    
    mean = mean_std['mean']
    std  = mean_std['sd']

    # genes × samples  →  samples × genes
    train_df = train_df.set_index("gene").T

    # align metadata with train samples
    meta_tr = metadata.set_index("Sample").loc[train_df.index]
    y_train = meta_tr["condition"]
    batch_train = meta_tr["batch"]

    # ---- Feature selection
    features_selected = select_features(train_df, mean, std, y_train)

    # Final training matrix
    X_train = train_df[features_selected]

    # ---- Load TEST data
    test_df = pd.read_excel(test_path, sheet_name=b, engine="odf")
    test_df = test_df.set_index("gene").T   # samples × genes
    X_test  = test_df[features_selected]

    # Ensure same gene order in train and test
    X_test = X_test[X_train.columns]

    # Align metadata for test samples
    meta_te = metadata.set_index("Sample").loc[X_test.index]
    y_test  = meta_te["condition"]

    # ---- Store
    folds[b] = {
        "X_train": X_train,
        "y_train": y_train,
        "batch_train": batch_train,
        "X_test": X_test,
        "y_test": y_test,
        "features_selected": features_selected
    }

print("✅ All folds loaded with metadata merged.")



with open("nested_dict_1000_0.1.pkl", "wb") as f:
    pkl.dump(folds, f)
