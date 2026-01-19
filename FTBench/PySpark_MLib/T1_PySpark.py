import os
import urllib.request
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import Bucketizer, StringIndexer, OneHotEncoder, VectorAssembler

def main():
    #Download dataset locally if it doesn't exist
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    local_file = "adult.data"

    if not os.path.exists(local_file):
        print("Downloading dataset..")
        urllib.request.urlretrieve(url, local_file)
        print("Download complete")

    spark = SparkSession.builder \
        .appName("Adult_T1_Transformation") \
        .getOrCreate()

    #Load dataset from local file
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "income"
    ]

    df = spark.read.csv(local_file, header=False, inferSchema=True)
    df = df.toDF(*column_names)

    #Column groups
    numerical_cols = [
        "age", "fnlwgt", "education-num",
        "capital-gain", "capital-loss", "hours-per-week"
    ]

    categorical_cols = [
        "workclass", "education", "marital-status",
        "occupation", "relationship", "race",
        "sex", "native-country"
    ]

    #Bin numerical columns (Bin + DC(5))
    n_bins = 5
    binned_cols = []

    for col in numerical_cols:
        stats = df.selectExpr(f"min(`{col}`) as min", f"max(`{col}`) as max").collect()[0]
        splits = [stats["min"] + i*(stats["max"]-stats["min"])/n_bins for i in range(n_bins+1)]
        splits[0] -= 1e-5  #include min in first bin

        bucket_col = col + "_binned"
        bucketizer = Bucketizer(splits=splits, inputCol=col, outputCol=bucket_col)
        df = bucketizer.transform(df)
        binned_cols.append(bucket_col)

    #Limit categorical columns to top 9 categories (DC(9))
    top_n = 9
    for col in categorical_cols:
        top_categories = [row[col] for row in df.groupBy(col).count().orderBy(F.desc("count")).limit(top_n).collect()]
        df = df.withColumn(col, F.when(F.col(col).isin(top_categories), F.col(col)).otherwise("Other"))

    #Index + One-hot encode categorical columns
    indexed_cols = []
    for col in categorical_cols:
        idx_col = col + "_idx"
        ohe_col = col + "_ohe"

        indexer = StringIndexer(inputCol=col, outputCol=idx_col, handleInvalid="keep")
        df = indexer.fit(df).transform(df)

        encoder = OneHotEncoder(inputCol=idx_col, outputCol=ohe_col)
        df = encoder.fit(df).transform(df)

        indexed_cols.append(ohe_col)

    #Pass-through column: encode income numerically (PT(1))
    income_indexer = StringIndexer(inputCol="income", outputCol="income_idx")
    df = income_indexer.fit(df).transform(df)
    pass_through_cols = ["income_idx"]

    #Assemble all features into a single vector
    assembler = VectorAssembler(
        inputCols=binned_cols + indexed_cols + pass_through_cols,
        outputCol="features"
    )
    df = assembler.transform(df)

    df.select("features").show(5, truncate=False)
    spark.stop()


if __name__ == "__main__":
    main()
