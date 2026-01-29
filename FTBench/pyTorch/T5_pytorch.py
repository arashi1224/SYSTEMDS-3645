import torch
import pandas as pd
import numpy as np
import time

def readNprep():
    # Read and isolate target and training data
    train = pd.read_csv("../../datasets/santander.csv", delimiter=",", header=None)
    santander = train.iloc[1:,2:]
    santander.columns = [*range(0,200)] #rename header from 0 to 199
    print(santander.head())
    print(santander.info())
    return santander

def transform_pytorch(df):
    base = df.copy(deep=True)

    bin_cols = base.columns
    N = len(df)

    coords_rows = []
    coords_cols = []
    coords_vals = []
    current_col_offset = 0
    
    for col in bin_cols:
        raw_vals = pd.to_numeric(df[col], errors='coerce').dropna().values
        tensor_vals = torch.from_numpy(raw_vals).float()

        min_val, max_val = raw_vals.min(), raw_vals.max()
        # 10 bins = 11 edges
        boundaries = torch.linspace(min_val, max_val, steps=11)
        boundaries[0] -= 0.001
        boundaries[-1] += 0.001
            
        bin_indices = torch.bucketize(tensor_vals, boundaries) - 1
        bin_indices = torch.clamp(bin_indices, 0, 4)
        
        coords_rows.append(torch.arange(N))
        coords_cols.append(bin_indices + current_col_offset)
        coords_vals.append(torch.ones(N))
        
        current_col_offset += 10

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
    santander = readNprep()
    timers = np.zeros(3)

    for i in range(3):
        t1 = time.time()
        X_transformed = transform_pytorch(santander)

        timers[i] = round(time.time() - t1,1)
        print(f"Elapsed time for transform = {(time.time() - t1) * 1000} millisec")

        print(f"\tOriginal shape: {santander.shape}")
        print(f"\tTransformed shape: {X_transformed.shape}")
    np.savetxt("santander_pytorch.dat", [round(np.mean(timers),1)], delimiter="\t", fmt='%f')