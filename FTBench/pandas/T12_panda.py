import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# 1. Generate synthetic dataset for T12 (100K x 100)
def generate_dataset():
    np.random.seed(42)
    rows = 100000
    num = 50
    cat = 50

    # Numerical features: continuous uniform random values [0, 100]
    num_data = np.random.uniform(low=0, high=100, size=(rows, num))

    # Categorical features: random categories (10 distinct per column)
    cat_data = np.random.randint(0, 10, size=(rows, cat)).astype(str)

    # Combine into DataFrame
    num_cols = [f"num_{i}" for i in range(num)]
    cat_cols = [f"cat_{i}" for i in range(cat)]
    
    df_num = pd.DataFrame(num_data, columns=num_cols)
    df_cat = pd.DataFrame(cat_data, columns=cat_cols)

    df = pd.concat([df_num, df_cat], axis=1)
    print(f"Generated T12 dataset: {df.shape}")
    return df, num_cols, cat_cols

# 2. Build transformation pipeline: Bin(50) + RC(50)
def build_transformer(num_cols, cat_cols):
    # Equi-width binning (10 bins) for 50 numerical columns
    num_pipeline = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    
    # Recoding (ordinal encoding) for 50 categorical columns
    cat_pipeline = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    transformer = ColumnTransformer(
        transformers=[
            ("num_bin", num_pipeline, num_cols), # Bin(50)
            ("cat_rc", cat_pipeline, cat_cols) # RC(50)
        ],
        remainder='drop'
    )
    return transformer

# 3. Full transformation with timing and saving
def run_t12_full(df, num_cols, cat_cols):
    print("Building and fitting T12 transformer..")
    transformer = build_transformer(num_cols, cat_cols)
    
    print("Running transformation..")
    t1 = time.time()
    X_transformed = transformer.fit_transform(df)
    t_total = time.time() - t1
    
    # Convert to DataFrame for CSV saving
    df_transformed = pd.DataFrame(X_transformed)
    
    print(f"Completed Transformation: {X_transformed.shape}")
    print(f"Time: {round(t_total, 3)} seconds")
    
    return df_transformed, t_total, transformer

# Main execution
if __name__ == "__main__":
    # Generate dataset
    print("FTBench T12: Generating Synthetic Dataset")
    df_t12, num_cols, cat_cols = generate_dataset()
    
    # Save RAW dataset
    print("\nSaving raw dataset..")
    df_t12.to_csv("SYSTEMDS-3645\\FTBench\\pandas\\T12_synthetic_raw.csv", index=False)
    print("Saved T12 raw dataset (100000 x 100)")
    
    # Run transformation
    print("\nT12 Transformation: Bin(50) + RC(50)")
    df_transformed, t_total, transformer = run_t12_full(df_t12, num_cols, cat_cols)
    
    # Save transformed dataset
    print("\nSaving transformed dataset..")
    df_transformed.to_csv("SYSTEMDS-3645\\FTBench\\pandas\\T12_synthetic_transformed.csv", index=False)
    print("Saved T12 transformed dataset (100000 x 100)")
    
    # Save timing
    with open("SYSTEMDS-3645\\FTBench\\pandas\\T12_FTBench_timing.dat", "w") as f:
        f.write(str(round(t_total, 3)))
    print("Saved timing to T12_FTBench_timing.dat")
    
    print("\nT12 Complete")
    print(f"Raw shape:      {df_t12.shape}")
    print(f"Transformed:    {df_transformed.shape}")
    print(f"Transformation time: {round(t_total, 3)}s")
