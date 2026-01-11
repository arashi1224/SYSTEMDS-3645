import torch
import pandas as pd
import numpy as np
import time

# Methods was taken from T8:sk.py
def readNprep():
    train = pd.read_csv("../../datasets/homeCreditTrain.csv", delimiter=",", header=None)
    test = pd.read_csv("../../datasets/homeCreditTest.csv", delimiter=",", header=None)
    train = train.iloc[1:,:] #remove header
    train.drop(1, axis=1, inplace=True); #remove target
    train.columns = [*range(0,121)] #rename header from 0 to 120
    test = test.iloc[1:,:]
    home = pd.concat([train, test])
    # Replace NaNs with before/after entries
    home.fillna(method='pad', inplace=True)
    home.fillna(method='bfill', inplace=True)
    print(home.head())
    print(home.info())
    return home

def transform_pytorch(df):
    base = df.copy(deep=True)
    dummy_col=[1,2,3,4,10,11,12,13,14,27,31,39,85,86,88,89]

    coords_rows = []
    coords_cols = []
    coords_vals = []
    current_col_offset = 0

    N = len(base)

    passthrough_col = []
    for col in base.columns:
        if col not in dummy_col:
            passthrough_col.append(col)

    for col in dummy_col:
        raw_vals = base[col].astype(str).values
        uniques = np.unique(raw_vals)
        token_ids = np.searchsorted(uniques, raw_vals) # same as bucketise, but for string
        token_ids = torch.from_numpy(token_ids)

        coords_rows.append(torch.arange(N))
        coords_cols.append(token_ids + current_col_offset)
        coords_vals.append(torch.ones(N))
        
        current_col_offset += len(uniques)

    for col in passthrough_col:
        raw_data = pd.to_numeric(base[col], errors='coerce').fillna(0).values
        tensor_vals = torch.from_numpy(raw_data).float()

        coords_rows.append(torch.arange(N))
        coords_cols.append(tensor_vals + current_col_offset)
        coords_vals.append(torch.ones(N))
        
        current_col_offset += 1
    
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
    home = readNprep()

    t1 = time.time()
    X_transformed = transform_pytorch(home)
    print(f"Elapsed time for transform = {(time.time() - t1) *1000} millisec")
    
    print(f"\tOriginal shape: {home.shape}")
    print(f"\tTransformed shape: {X_transformed.shape}")