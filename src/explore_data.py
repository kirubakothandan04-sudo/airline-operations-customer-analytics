import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset location
DATA_DIR = BASE_DIR / "data" / "archive-2"

# Load datasets
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

print("=" * 60)
print("AIRLINE PASSENGER SATISFACTION ANALYSIS")
print("=" * 60)

# Dataset dimensions
print("\n1. DATASET SIZE")
print("Training:", train.shape)
print("Testing:", test.shape)

# Columns
print("\n2. DATASET COLUMNS")
print(train.columns.tolist())

# First records
print("\n3. FIRST FIVE RECORDS")
print(train.head())

# Missing values
print("\n4. MISSING VALUES")
print(train.isnull().sum())

# Duplicate records
print("\n5. DUPLICATE RECORDS")
print(train.duplicated().sum())

# Satisfaction distribution
print("\n6. SATISFACTION DISTRIBUTION")
print(train["satisfaction"].value_counts())

# Statistical summary
print("\n7. STATISTICAL SUMMARY")
print(train.describe())

print("\nDATA EXPLORATION COMPLETED!")