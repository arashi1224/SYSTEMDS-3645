from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)

spark = SparkSession.builder \
    .appName("T8-HomeCredit-DC16") \
    .getOrCreate()

df = spark.read.csv(
    "T8_Dataset.csv",
    header=True,
    inferSchema=True
)

categorical_cols = [
    "NAME_CONTRACT_TYPE",
    "CODE_GENDER",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
    "ORGANIZATION_TYPE",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE"
]

#StringIndexers (build dictionaries)
indexers = [
    StringIndexer(
        inputCol=col,
        outputCol=f"{col}_idx",
        handleInvalid="keep"
    )
    for col in categorical_cols
]

#OneHotEncoders (dummy coding)
encoders = [
    OneHotEncoder(
        inputCol=f"{col}_idx",
        outputCol=f"{col}_ohe",
        dropLast=True
    )
    for col in categorical_cols
]

#Assemble all dummy-coded columns into one feature vector
assembler = VectorAssembler(
    inputCols=[f"{col}_ohe" for col in categorical_cols],
    outputCol="features"
)

#Build pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler])

model = pipeline.fit(df)
df_t8 = model.transform(df)

df_t8.select("features").show(truncate=False)

