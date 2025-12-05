#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 09:41:26 2025

@author: raluca
"""

import pandas as pd
from functools import reduce


files = [
    "pseudo_bulk_GSE156326 hypertrophic scar 1.csv",
    "pseudo_bulk_GSE156326 hypertrophic scar 2.csv",
    "pseudo_bulk_GSE156326 hypertrophic scar 3.csv",
    "pseudo_bulk_GSE156326 normal skin 1.csv",
    "pseudo_bulk_GSE156326 normal skin 2.csv",
    "pseudo_bulk_GSE156326 normal skin 3.csv",
    "pseudo_bulk_GSE163973 KF1_matrix.csv",
    "pseudo_bulk_GSE163973 KF2_matrix.csv",
    "pseudo_bulk_GSE163973 KF3_matrix.csv",
    "pseudo_bulk_GSE163973 NF1_matrix.csv",
    "pseudo_bulk_GSE163973 NF2_matrix.csv",
    "pseudo_bulk_GSE163973 NF3_matrix.csv",
    "pseudo_bulk_GSE181297 Keloid 1.csv",
    "pseudo_bulk_GSE181297 keloid 2.csv",
    "pseudo_bulk_GSE243716 hypertrophic.csv",
    "pseudo_bulk_GSE243716 keloid.csv",
    "pseudo_bulk_GSE266334 keloid 1.csv",
    "pseudo_bulk_GSE266334 keloid 2.csv",
    "pseudo_bulk_GSE266334 keloid 3.csv",
    "pseudo_bulk_GSE266334 normal skin 1.csv",
    "pseudo_bulk_GSE270438 keloid scar 1.csv",
    "pseudo_bulk_GSE270438 keloid scar 2.csv",
    "pseudo_bulk_GSE270438 keloid scar 3.csv",
    "pseudo_bulk_GSE270438 normal scar 1.csv",
    "pseudo_bulk_GSE270438 normal scar 2.csv",
    "pseudo_bulk_GSE270438 normal scar 3.csv",
    "pseudo_bulk_GSE303592 keloid.csv",
    "pseudo_bulk_GSE303592 normal.csv",
    "pseudo_bulk_GSE307504_keloid.csv",
    "pseudo_bulk_GSE_188952_Hypertrophic scar_2.csv",
    "pseudo_bulk_GSE_188952_Hypertrophic scar_3.csv",
    "pseudo_bulk_GSE_188952_Hypertrophic scar_4.csv",
    "pseudo_bulk_GSE_188952_Hypertrophic_scar_1.csv",
    "pseudo_bulk_GSE_188952_hypertrophic scar_5.csv",
    "pseudo_bulk_GSE_188952_keloid scar_1.csv",
    "pseudo_bulk_GSE_188952_keloid scar_2.csv",
    "pseudo_bulk_GSE_188952_keloid scar_3.csv",
    "pseudo_bulk_GSE_188952_keloid scar_4.csv",
    "pseudo_bulk_GSE_188952_normotropic_scar_1.csv",
    "pseudo_bulk_GSE_188952_normotropic_scar_2.csv",
    "pseudo_bulk_GSE_188952_normotropic_scar_3.csv",
    "pseudo_bulk_GSE307504_hypertrophic.csv",
]




dfs = []
for f in files:
    df = pd.read_csv(f)
    df.columns = ['gene', f.split('.')[0]]
    # Deduplicate gene entries — keep the first occurrence
    df = df.drop_duplicates(subset='gene')
    # Strip whitespace and unify case (important!)
    df['gene'] = df['gene'].str.strip().str.upper()
    dfs.append(df)
    
for f, df in zip(files, dfs):
    print(f"{f}: {df.shape[0]} unique genes")

# Merge on common genes
merged = reduce(lambda left, right: pd.merge(left, right, on='gene', how='inner'), dfs)

#save
merged.to_csv("merged_counts_common_genes.csv", index=False)
print("Merged matrix shape:", merged.shape)

