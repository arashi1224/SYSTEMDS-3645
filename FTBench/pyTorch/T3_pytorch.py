import torch
import pandas as pd
import numpy as np
import time

# Methods was taken from T8:sk.py
def readNprep(nRows):
    if nRows == 1:
        criteo = pd.read_csv("../..datasets/criteo_day21_1M", delimiter=",", header=None)
    if nRows == 10:
        criteo = pd.read_csv("../..datasets/criteo_day21_10M", delimiter=",", header=None)
    
    criteo = criteo.fillna(method="ffill").fillna(method="bfill")
    print(criteo.head())
    print(criteo.info())
    return criteo

def transform_pytorch(df):
    base = df.copy(deep=True)
    # Columns 0-12 are Numerical (I1-I13)
    # Columns 13-38 are Categorical (C1-C26)
    num_cols = list(range(0, 13))
    cat_cols = list(range(13, 39))

    coords_rows = []
    coords_cols = []
    coords_vals = []
    current_col_offset = 0

    N = len(base)

    passthrough_col = []
    for col in base.columns:
        if col not in cat_cols:
            passthrough_col.append(col)

    for col in cat_cols:
        raw_vals = df[col].astype(str).values
        uniques = np.unique(raw_vals)
        token_ids = np.searchsorted(uniques, raw_vals) # same as bucketise, but for string
        token_ids = torch.from_numpy(token_ids)

        coords_rows.append(torch.arange(N))
        coords_cols.append(token_ids + current_col_offset)
        coords_vals.append(torch.ones(N))
        
        current_col_offset += len(uniques)
    
    # Combine from all cols
    final_rows = torch.cat(coords_rows).long()
    final_cols = torch.cat(coords_cols).long()
    final_vals = torch.cat(coords_vals).float()

    # Create Sparse Matrix (COO)
    sparse_tensor = torch.sparse_coo_tensor(
        torch.stack([final_rows, final_cols]),
        final_vals,
        size=(N, current_col_offset)
    )

    return sparse_tensor.to_sparse_csr()

if __name__ == '__main__':
    criteo_cat = readNprep()

    t1 = time.time()
    X_transformed = transform_pytorch(criteo_cat)
    print(f"Elapsed time for transform = {(time.time() - t1) *1000} millisec")
    
    print(f"\tOriginal shape: {criteo_cat.shape}")
    print(f"\tTransformed shape: {X_transformed.shape}")