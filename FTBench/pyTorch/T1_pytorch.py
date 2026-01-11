import torch
import pandas as pd
import numpy as np
import time

def readNprep():
    adult = pd.read_csv("../../datasets/adult.data", delimiter=",", header=None, skipinitialspace=True)
    print(adult.head())
    print(adult.info())
    return adult

def transform_python(df):
    
    bin_cols = [0, 2, 10, 11, 12]
    cat_cols = [1, 3, 5, 6, 7, 8, 9, 13, 14]
    passthrough_cols = [4]

    N = len(df)

    coords_rows = []
    coords_cols = []
    coords_vals = []
    current_col_offset = 0
    
    for col in bin_cols:
        raw_vals = pd.to_numeric(df[col], errors='coerce').dropna().values
        tensor_vals = torch.from_numpy(raw_vals).float()

        if N > 0:
            min_val, max_val = raw_vals.min(), raw_vals.max()
            # 5 bins = 6 edges
            boundaries = torch.linspace(min_val, max_val, steps=6)
            boundaries[0] -= 0.001
            boundaries[-1] += 0.001
            
        bin_indices = torch.bucketize(tensor_vals, boundaries) - 1
        bin_indices = torch.clamp(bin_indices, 0, 4)
        
        coords_rows.append(torch.arange(N))
        coords_cols.append(bin_indices + current_col_offset)
        coords_vals.append(torch.ones(N))
        
        current_col_offset += 5
    
    for col in cat_cols:
        raw_vals = df[col].astype(str).values
        uniques = np.unique(raw_vals)
        token_ids = np.searchsorted(uniques, raw_vals) # same as bucketise, but for string
        token_ids = torch.from_numpy(token_ids)

        coords_rows.append(torch.arange(N))
        coords_cols.append(token_ids + current_col_offset)
        coords_vals.append(torch.ones(N))
        
        current_col_offset += len(uniques)

    for col in passthrough_cols:
        raw_data = pd.to_numeric(df[col], errors='coerce').fillna(0).values
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
    adult = readNprep()
    print(adult)

    t1 = time.time()
    X_transformed = transform_python(adult)
    print(f"Elapsed time for transform = {(time.time() - t1) *1000:.2f} millisec")
    
    print(f"\tOriginal shape: {adult.shape}")
    print(f"\tTransformed shape: {X_transformed.shape}")