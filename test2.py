import pandas as pd

df = pd.read_parquet("image_features.parquet")

print(df)

print(df.columns)

print(df.shape)

print(df.info())

print(df.describe())