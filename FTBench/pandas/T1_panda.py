import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix

def readNprep():
    adult = pd.read_csv("../../datasets/adult.data", delimiter=",", header=None, skipinitialspace=True)
    print(adult.head())
    print(adult.info())
    return adult

def transform_pandas(df):
    result = df.copy()
    
    bin_cols = [0, 2, 10, 11, 12]
    cat_cols = [1, 3, 5, 6, 7, 8, 9, 13, 14]
    passthrough_cols = [4]

    binned_dfs = []
    cat_dfs = []
    
    for col in bin_cols:
        result[col] = pd.to_numeric(result[col], errors='coerce')
        binned = pd.cut(result[col], bins=5, labels=range(5), duplicates='drop')

        # dummy code for int data
        dummies = pd.get_dummies(binned, prefix=f'col_{col}_bin', dtype=int)
        binned_dfs.append(dummies)
    
    for col in cat_cols:
        dummies = pd.get_dummies(result[col], prefix=f'col_{col}', dtype=int)
        cat_dfs.append(dummies)

    for col in passthrough_cols:
        passthrough = pd.DataFrame(
            result[col].astype(float), 
            columns=[f'passthrough_{col}']
        )
        binned_dfs.append(passthrough)

    all_features = binned_dfs + cat_dfs
    transformed = pd.concat(all_features, axis=1)
    return csr_matrix(transformed.astype(float).values)

if __name__ == '__main__':
    adult = readNprep()

    t1 = time.time()
    X_transformed = transform_pandas(adult)
    print(f"Elapsed time for transform = {(time.time() - t1) *1000:.2f} millisec")
    
    print(f"\tOriginal shape: {adult.shape}")
    print(f"\tTransformed shape: {X_transformed.shape}")