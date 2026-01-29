from time import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from scipy import sparse

# 1) Load dataset (ensure dataset is already downloaded)
df = pd.read_csv("SYSTEMDS-3645\\FTBench\\panda\\T5_Dataset.csv")
df = df.drop(columns=["target", "ID_code"])

# Ensure only feature columns are used
# (drop id/target if present)
X = df.select_dtypes(include=["number"])

t1 = time.time()

# 2) Equi-height binning (quantile binning)
binning = KBinsDiscretizer(
    n_bins=10,
    encode="ordinal",
    strategy="quantile"  # equi-height
)

X_binned = binning.fit_transform(X)

# 3) Dummy-coding/one-hot encoding
encoder = OneHotEncoder(
    sparse_output=True,
    handle_unknown="ignore"
)

X_onehot = encoder.fit_transform(X_binned)
timers = (time.time() - t1) * 1000
np.savetxt("kdd_pandas.dat", [np.mean(timers)], delimiter="\t", fmt='%f')

# 4) Result: sparse matrix matching T5
print(X.shape)       #(200000, 200)
print(X_onehot.shape)  #≈(200000, 200 * 10)
print(isinstance(X_onehot, sparse.spmatrix))  #True
