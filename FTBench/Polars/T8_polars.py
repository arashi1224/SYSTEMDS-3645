import polars as pl
import numpy as np
import time
import json
from scipy.sparse import csr_matrix

# Methods was taken from T8:sk.py
def readNprep():
    df_train = pl.read_csv("Polars/data/application_train.csv", has_header=False)
    df_test = pl.read_csv("Polars/data/application_test.csv", has_header=False)

    # remove header row and target
    df_train = df_train[1:, :]
    df_train = df_train.drop(df_train.columns[1])
    # rename columns for concat
    df_train.columns = [str(i) for i in range(len(df_train.columns))]

    df_test = df_test[1:, :]
    # rename test columns for concat
    df_test.columns = [str(i) for i in range(len(df_test.columns))]

    # concat test train
    df_home = pl.concat([df_train, df_test])

    # replace NaNs
    df_home = df_home.fill_null(strategy="forward")
    df_home = df_home.fill_null(strategy="backward")
    # open and read .json spec file
    spec_file = open("polars/specs/homecredit_spec1.json", "r")
    spec = json.loads(spec_file.read())
    dummy_cols = [i - 1 for i in spec['dummycode']]

    return df_home, dummy_cols

def transform_panda(df, dummy_cols):
    dummy_col_names = [df.columns[i] for i in dummy_cols]
    passthrough_col_names = [col for col in df.columns if col not in dummy_col_names]  

    result = []
    for col in dummy_col_names:
        # dummy code
        dummies = df.select([
            pl.col(col)
        ]).to_dummies(columns=[col], separator='_')

        # rename with prefix
        dummies = dummies.rename({
            old_name: f'col_{col}_{old_name.split("_")[-1]}'
            for old_name in dummies.columns
        })
        
        result.append(dummies)
    
    final_df = pl.concat(result, how='horizontal')
    
    # convert to sparse matrix (copied from T8_pandas)
    return csr_matrix(final_df.to_numpy().astype(float))


if __name__ == '__main__':
    df_home, dummy_cols = readNprep()

    t1 = time.time()
    X_transformed = transform_panda(df_home, dummy_cols=dummy_cols)
    print(f"Elapsed time for transform = {(time.time() - t1) *1000} millisec")
    
    print(f"\tOriginal shape: {df_home.shape}")
    print(f"\tTransformed shape: {X_transformed.shape}")