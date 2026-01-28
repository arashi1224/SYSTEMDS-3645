import pandas as pd
import numpy as np
import json
import time

def read_and_prep(specfile_name, datafile_name):
    # read and prep
    df = pd.read_csv(datafile_name, delimiter=",", header=None)
    df = df.iloc[1:,:]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='pad', inplace=True)
    df.fillna(method='bfill', inplace=True)

    # open and read .json spec file
    spec_file = open(specfile_name, "r")
    spec = json.loads(spec_file.read())
    bins = []
    nbins = []

    # getting specs
    for bin in spec['bin']:
        bins.append(bin.get('id'))
        nbins.append(bin.get('numbins'))

    # number of bins
    nbins = nbins[0]
    # index starting at 0
    bins = [i-1 for i in bins]

    # Cast bin columns to float
    df.iloc[:, bins] = df.iloc[:, bins].astype(float)

    return df, bins, nbins

# bin-transformation on all indexed columns
def bin_columns(df, columns, num_bins):
    for col in columns:
        df.iloc[:, col] = pd.qcut(df.iloc[:, col], q=num_bins, labels=False, duplicates='drop')
    return df

def benchmark_t7(df, bins, nbins):
    timeres = np.zeros(3)
    # benchmark transformation
    for i in range(3):
        df_copy = df.copy()
        t1 = time.time()
        df_transformed = bin_columns(df_copy, bins, nbins)
        timeres[i] = (time.time() - t1) * 1000

    # print and save
    print(np.shape(df_transformed))
    print("Elapsed time for transformations using pandas.cut in millisec")
    print(timeres)

    return df_transformed


if __name__ == '__main__':
    df, bins, nbins = read_and_prep(specfile_name = "pandas/specs/crypto_spec2.json", datafile_name = "pandas/data/train.csv")
    df_transformed = benchmark_t7(df, bins, nbins)