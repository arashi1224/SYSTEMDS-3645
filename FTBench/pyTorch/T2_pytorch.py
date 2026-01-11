import torch
import pandas as pd
import numpy as np
import time

# taken from keras T2
def readNprep():
    # Read and isolate target and training data
    kdd = pd.read_csv("../../datasets/KDD98.csv", delimiter=",", header=None)
    print(kdd.head())
    print(kdd.shape)
    kddX = kdd.iloc[:,0:469]
    kddX = kddX.drop([0], axis=0)
    kddX = kddX.replace(r'\s+',np.nan,regex=True).replace('',np.nan)
    # Replace NAs with before/after entries
    kddX = kddX.fillna(method="ffill").fillna(method="bfill")

    # Cast categorical columns to str, 
    #st = [23,24,*range(28,42),195,196,197,*range(362,384),*range(412,434)]
    #kddX[st] = kddX[st].astype(str)

    # Set dtype float for numeric columns
    # The default dtype for all columns is object at this point 
    fl = [4,7,16,26,*range(43,50),53,*range(75,195),*range(198,361),407,409,410,411,*range(434,469)]
    kddX[fl] = kddX[fl].astype(float)
    cat = kddX.select_dtypes(exclude=np.float64).columns
    kddX[cat] = kddX[cat].astype(str)
    print("head: \n", kddX.head())
    print("head: \n", kddX.info())
    return kddX

def transform_pytorch(df):
    base = df.copy(deep=True)
    
    fl = base.select_dtypes(include=np.float64).columns
    cat = base.select_dtypes(exclude=np.float64).columns

    coords_rows = []
    coords_cols = []
    coords_vals = []
    current_col_offset = 0

    N = len(df)

    for col in fl:
        raw_vals = pd.to_numeric(base[col], errors='coerce').dropna().values
        tensor_vals = torch.from_numpy(raw_vals).float()

        min_val, max_val = raw_vals.min(), raw_vals.max()
        # 5 bins = 6 edges
        boundaries = torch.linspace(min_val, max_val, steps=6)
        boundaries[0] -= 0.001
        boundaries[-1] += 0.001
            
        bin_indices = torch.bucketize(tensor_vals, boundaries) - 1
        bin_indices = torch.clamp(bin_indices, 0, 4)

        counts = torch.bincount(bin_indices, minlength=5).float()
        probs = counts / N
        # standard deviation of a binary variable = sqrt(p(1-p))
        stds = torch.sqrt(probs * (1 - probs)) + 1e-8
        scaling_factors = 1.0 / stds

        values = scaling_factors[bin_indices]
        
        coords_rows.append(torch.arange(N))
        coords_cols.append(bin_indices + current_col_offset)
        coords_vals.append(values)
        
        current_col_offset += 5
    print("cols for binning: ", current_col_offset)

    for col in cat:
        raw_vals = base[col].astype(str).values
        uniques, counts = np.unique(raw_vals, return_counts = True)
        token_ids = np.searchsorted(uniques, raw_vals) # same as bucketise, but for string
        token_ids = torch.from_numpy(token_ids)

        probs = torch.tensor(counts / N, dtype=torch.float32)
        stds = torch.sqrt(probs * (1 - probs)) + 1e-8
        scaling_factors = 1.0 / stds
        
        values = scaling_factors[token_ids]

        coords_rows.append(torch.arange(N))
        coords_cols.append(token_ids + current_col_offset)
        coords_vals.append(values)
        
        current_col_offset += len(uniques)
    
    
    print("total cols: ", current_col_offset)
    
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
    kdd = readNprep()

    t1 = time.time()
    X_transformed = transform_pytorch(kdd)
    print(f"Elapsed time for transform = {(time.time() - t1) *1000} millisec")
    
    print(f"\tOriginal shape: {kdd.shape}")
    print(f"\tTransformed shape: {X_transformed.shape}")