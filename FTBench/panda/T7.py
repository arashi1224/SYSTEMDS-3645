import pandas as pd
import numpy as np
import json
import time

# read and prep
train = pd.read_csv("train.csv", delimiter=",", header=None)
crypto = train.iloc[1:,:]
crypto.replace([np.inf, -np.inf], np.nan, inplace=True)
crypto.fillna(method='pad', inplace=True)
crypto.fillna(method='bfill', inplace=True)

# open and read .json spec file
spec_file = open("specs/crypto_spec2.json", "r")
spec = json.loads(spec_file.read())
timeres = np.zeros(3)
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
crypto.iloc[:, bins] = crypto.iloc[:, bins].astype(float)

# bin-transformation on all indexed columns
def bin_columns(df, columns, num_bins):
    for col in columns:
        df.iloc[:, col] = pd.qcut(df.iloc[:, col], q=num_bins, labels=False, duplicates='drop')
    return df

# benchmark transformation
for i in range(3):
    df_copy = crypto.copy()
    t1 = time.time()
    df_transformed = bin_columns(df_copy, bins, nbins)
    timeres[i] = (time.time() - t1) * 1000

# print and save
print(np.shape(df_transformed))
print("Elapsed time for transformations using pandas.cut in millisec")
print(timeres)
