import polars as pl
import numpy as np
import json
import time

def read_and_prep(specfile_name, datafile_name):
    # read and prep
    df = pl.read_csv(datafile_name)
    df = df[1:, :]
    # Cast all columns to float first to handle inf values
    df = df.with_columns([pl.all().cast(pl.Float64, strict=False)])

    df = df.with_columns([pl.all().replace([float('inf'), float('-inf')], None)])
    df = df.fill_null(strategy="forward")
    df = df.fill_null(strategy="backward")

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
    df = df.with_columns([pl.col(df.columns[i]).cast(pl.Float64) for i in bins])

    return df, bins, nbins

# bin-transformation on all indexed columns
def bin_columns(df, columns, num_bins):
    bin_expressions = []
    
    for col_idx in columns:
        col_name = df.columns[col_idx]
        
        bin_expr = (
            pl.col(col_name)
            .qcut(quantiles=num_bins, allow_duplicates=True, include_breaks=False)
            .cast(pl.UInt32)
            .alias(col_name)
        )
        
        bin_expressions.append(bin_expr)
    
    return df.with_columns(bin_expressions)

def benchmark_t7(df, bins, nbins):    
    timeres = np.zeros(3)
    # benchmark transformation
    for i in range(3):
        df_copy = df.clone()
        t1 = time.time()
        df_transformed = bin_columns(df_copy, bins, nbins)
        timeres[i] = (time.time() - t1) * 1000

    # print and save
    print(np.shape(df_transformed))
    print("Elapsed time for transformations using pandas.cut in millisec")
    print(timeres)

    return df_transformed


if __name__ == '__main__':
    df, bins, nbins = read_and_prep(specfile_name = "polars/specs/crypto_spec1.json", datafile_name = "polars/data/train.csv")
    df_transformed = benchmark_t7(df, bins, nbins)