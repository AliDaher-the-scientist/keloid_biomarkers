#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 12:29:43 2025

@author: raluca
"""

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import seaborn as sns
import plotly.io as pio
pio.renderers.default = "browser"

# Load batch-corrected matrix (genes × samples)
expr = pd.read_csv("normalized_batch_corrected.csv", index_col=0)
pre_batch = pd.read_csv("vst_matrix_uncorrected.csv", index_col=0)
# Load metadata
meta = pd.read_csv("metadata.csv", index_col=0)

# Make sure columns in expr match metadata rows
expr = expr.loc[:, meta.index]
print(expr.shape, meta.shape)

# Transpose: samples × genes
X = expr.T

pca = PCA(n_components=3)
pca_coords = pca.fit_transform(X)

pca_df = pd.DataFrame(pca_coords, columns=['PC1', 'PC2', 'PC3'], index=X.index)
pca_df = pca_df.join(meta)

print(f"Explained variance: {pca.explained_variance_ratio_ * 100}")

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

colors = sns.color_palette("tab10", n_colors=pca_df["batch"].nunique())

for i, batch in enumerate(pca_df["batch"].unique()):
    subset = pca_df[pca_df["batch"] == batch]
    ax.scatter(subset["PC1"], subset["PC2"], subset["PC3"],
               color=colors[i], label=batch, s=60, alpha=0.8)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% var)")
ax.set_title("3D PCA: Colored by Batch (After ComBat)")
ax.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
fig.show()


fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

colors = sns.color_palette("Set2", n_colors=pca_df["condition"].nunique())

for i, cond in enumerate(pca_df["condition"].unique()):
    subset = pca_df[pca_df["condition"] == cond]
    ax.scatter(subset["PC1"], subset["PC2"], subset["PC3"],
               color=colors[i], label=cond, s=60, alpha=0.8)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% var)")
ax.set_title("3D PCA: Colored by Condition (After ComBat)")
ax.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()




X_before = pre_batch.T
pca_before = PCA(n_components=3).fit_transform(X_before)
pca_before_df = pd.DataFrame(pca_before, columns=['PC1', 'PC2', 'PC3'], index=X_before.index)
pca_before_df = pca_before_df.join(meta)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

colors = sns.color_palette("tab10", n_colors=pca_before_df["batch"].nunique())

for i, batch in enumerate(pca_before_df["batch"].unique()):
    subset = pca_before_df[pca_before_df["batch"] == batch]
    ax.scatter(subset["PC1"], subset["PC2"], subset["PC3"],
               color=colors[i], label=batch, s=60, alpha=0.8)

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}% var)")
ax.set_title("3D PCA: Colored by Batch (Before ComBat)")
ax.legend(bbox_to_anchor=(1.05, 1))
plt.tight_layout()
fig.show()