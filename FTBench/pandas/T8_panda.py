import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix

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

def transform_panda(X):
    base = X.copy(deep=True)
    dummy_col=[1,2,3,4,10,11,12,13,14,27,31,39,85,86,88,89]

    passthrough_col = []
    for col in base.columns:
        if col not in dummy_col:
            passthrough_col.append(col)

    result = []

    for col in passthrough_col:
        passthrough = pd.DataFrame(
            pd.to_numeric(base[col], errors='coerce'), 
            columns=[f'passthrough_{col}']
        )
        result.append(passthrough)
    for col in dummy_col:
        dummies = pd.get_dummies(base[col], prefix=f'col_{col}')
        result.append(dummies)

    final_df = pd.concat(result, axis=1)

    return csr_matrix(final_df.astype(float).values)

if __name__ == '__main__':
    home = readNprep()

    t1 = time.time()
    X_transformed = transform_panda(home)
    print(f"Elapsed time for transform = {(time.time() - t1) *1000} millisec")
    
    print(f"\tOriginal shape: {home.shape}")
    print(f"\tTransformed shape: {X_transformed.shape}")