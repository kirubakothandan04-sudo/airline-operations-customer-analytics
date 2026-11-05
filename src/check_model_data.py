
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

train = pd.read_csv(
    BASE_DIR / "data/processed/train_cleaned.csv"
)

test = pd.read_csv(
    BASE_DIR / "data/processed/test_cleaned.csv"
)

# Exclude target and ID
X_train = train.drop(columns=["satisfaction", "id"])

numeric = X_train.select_dtypes(include=["number"])

print("=" * 60)
print("NUMERICAL DATA DIAGNOSTICS")
print("=" * 60)

print("\n1. Checking missing values:")
print(numeric.isnull().sum().sum())

print("\n2. Checking infinite values:")
print(np.isinf(numeric.to_numpy()).sum())

print("\n3. Checking numerical ranges:")
print(numeric.describe().T[
    ["min", "max", "mean", "std"]
].to_string())

print("\n4. Checking columns with zero standard deviation:")
print(numeric.columns[numeric.std() == 0].tolist())

print("\n5. Checking extremely large values:")
for column in numeric.columns:
    maximum = numeric[column].abs().max()

    if maximum > 10000:
        print(column, maximum)

print("\nDATA DIAGNOSTICS COMPLETED")
