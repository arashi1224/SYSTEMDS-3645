import torch
import pandas as pd
import numpy as np
import time

def readNprep():
    # Read and isolate target and training data
    train = pd.read_csv("../../datasets/catindattrain.csv", delimiter=",", header=None).fillna("missing").astype(str)
    train = train.iloc[1:,:] #remove header
    train.drop(24, axis=1, inplace=True); #remove target
    print(train.head())
    print(train.info())
    return train

def transform_pytorch_t9(df):
    base= df.copy(deep=True)

    cols = base.columns
    N = len(base)
    HASH_K = 1000
    
    coords_rows = []
    coords_cols = []
    coords_vals = []
    
    current_col_offset = 0

    for col in cols:
        raw_vals = base[col].values
        
        # hash instead of unique
        hashed_vals = pd.util.hash_array(raw_vals, encoding='utf8')
        
        bucket_ids = (np.abs(hashed_vals) % HASH_K).astype(np.int64)
        token_ids = torch.from_numpy(bucket_ids)
        coords_rows.append(torch.arange(N))
        coords_cols.append(token_ids + current_col_offset)
        coords_vals.append(torch.ones(N))
        
        current_col_offset += HASH_K

    final_rows = torch.cat(coords_rows).long()
    final_cols = torch.cat(coords_cols).long()
    final_vals = torch.cat(coords_vals).float()

    sparse_tensor = torch.sparse_coo_tensor(
        torch.stack([final_rows, final_cols]),
        final_vals,
        size=(N, current_col_offset)
    )

    return sparse_tensor.to_sparse_csr()

if __name__ == '__main__':
    cat_dat = readNprep()

    t1 = time.time()
    X_transformed = transform_pytorch_t9(cat_dat)
    
    print(f"Elapsed time for transform = {(time.time() - t1) * 1000} millisec")
    
    print(f"\tOriginal shape: {cat_dat.shape}")
    print(f"\tTransformed shape: {X_transformed.shape}")