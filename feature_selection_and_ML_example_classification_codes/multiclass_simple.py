#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 12:49:15 2025

@author: raluca
"""
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, GroupKFold
import numpy as np
import pickle as pkl
import pandas as pd

with open("nested_dict_1000_0.1.pkl",'rb') as f:
    dict_data = pkl.load(f)
    
results_summary = []
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
l1_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
#num_cv_batches = len(dict_data)-1

results = {}
accuracy_list = []
F1_score_macro_list = []
F1_score_micro_list = []
F1_score_weighted_list = []

for b, data in dict_data.items():
    print(f"Training LOSO fold: held out batch {b}")

    X_train = data["X_train"]
    y_train = data["y_train"]
    batch_train = data["batch_train"]  # needed for group-aware CV
    features_selected = data['features_selected']
    X_test = data["X_test"]
    y_test = data["y_test"]

 # group-Kfold to ensure that during tuning, I never split one batch
    num_cv_batches = len(np.unique(batch_train))
    cv = GroupKFold(n_splits=num_cv_batches)

    model = LogisticRegression(
     penalty="elasticnet",
     solver="saga",
     max_iter=int(1e5),
     #multi_class="multinomial",
     random_state=12
 )

    param_grid = {
     "C": C_values,
     "l1_ratio": l1_ratios
 }

    clf = GridSearchCV(
     estimator=model,
     param_grid=param_grid,
     cv=cv.split(X_train, y_train, groups=batch_train),
     scoring="f1_micro",
     n_jobs=-1
 )

    clf.fit(X_train, y_train)

 # pick model with best hyperparameters
    best_model = clf.best_estimator_

 # Evaluate on held-out test

    preds = best_model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro")
    f1_micro = f1_score(y_test, preds, average="micro")
    f1_weighted = f1_score(y_test, preds, average="weighted")

    print(f"→ Best params: {clf.best_params_}")
    print(f"→ Test ACC = {acc:.3f}, Test Macro-F1 = {f1_macro:.3f}, Test Micro-F1 = {f1_micro:.3f},\
          Test weighted-F1 = {f1_weighted:.3f}",)
    accuracy_list.append(acc)
    F1_score_macro_list.append(f1_macro)
    F1_score_micro_list.append(f1_micro)
    F1_score_weighted_list.append(f1_weighted)
 
    results[b] = {
       "best_model": best_model,
       "best_params": clf.best_params_,
       "test_preds": best_model.predict(X_test),
       "true_labels": y_test,
       "coefficients": best_model.coef_,
       "features": features_selected
   }
    
# with open('multi_regress_dict.pkl', 'wb') as f:
#     pkl.dump(results, f)
    
# metrics_dict = {
#     'accuracy': accuracy_list,
#     'f1_macro': F1_score_macro_list,
#     'f1_micro': F1_score_micro_list,
#     'f1_weighted': F1_score_weighted_list
# }

# with open('metrics_summary.pkl', 'wb') as f:
#     pkl.dump(metrics_dict, f)
        
