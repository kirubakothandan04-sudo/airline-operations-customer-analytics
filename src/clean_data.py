
import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "archive-2"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load original datasets
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")

print("=" * 60)
print("AIRLINE DATA CLEANING AND VALIDATION")
print("=" * 60)

# 1. Remove unnecessary exported index column
for df in [train, test]:
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

print("\n1. Removed unnecessary index column.")

# 2. Check dataset consistency
if set(train.columns) != set(test.columns):
    raise ValueError("Training and testing columns do not match.")

test = test[train.columns]

# 3. Record missing values before cleaning
print("\n2. MISSING VALUES BEFORE CLEANING")
print("Training:", train.isnull().sum().sum())
print("Testing:", test.isnull().sum().sum())

# 4. Learn imputation values from training data only
imputation_values = {}

for column in train.columns:
    if train[column].isnull().any():
        if pd.api.types.is_numeric_dtype(train[column]):
            imputation_values[column] = train[column].median()
        else:
            imputation_values[column] = train[column].mode()[0]

# 5. Apply the same training-derived values to both datasets
train.fillna(imputation_values, inplace=True)
test.fillna(imputation_values, inplace=True)

print("\n3. MISSING VALUES AFTER CLEANING")
print("Training:", train.isnull().sum().sum())
print("Testing:", test.isnull().sum().sum())

# 6. Check duplicates
print("\n4. DUPLICATE RECORDS")
print("Training:", train.duplicated().sum())
print("Testing:", test.duplicated().sum())

# 7. Remove exact duplicate rows if present
train.drop_duplicates(inplace=True)
test.drop_duplicates(inplace=True)

# 8. Final validation
assert train.isnull().sum().sum() == 0
assert test.isnull().sum().sum() == 0

# 9. Save processed datasets
train.to_csv(OUTPUT_DIR / "train_cleaned.csv", index=False)
test.to_csv(OUTPUT_DIR / "test_cleaned.csv", index=False)

print("\n5. FINAL DATASET SHAPES")
print("Training:", train.shape)
print("Testing:", test.shape)

print("\n6. SAVED FILES")
print("data/processed/train_cleaned.csv")
print("data/processed/test_cleaned.csv")

print("\nDATA CLEANING COMPLETED SUCCESSFULLY!")
