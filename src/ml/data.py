import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/raw/creditcard.csv")
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Class"])

train.to_parquet("data/processed/train.parquet")
test.to_parquet("data/processed/test.parquet")

# Reference sample for future drift detection
reference = train.sample(10000, random_state=42)
reference.to_parquet("data/processed/reference.parquet")

print("Data prepared!")
