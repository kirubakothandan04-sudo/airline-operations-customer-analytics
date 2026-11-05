
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent

train = pd.read_csv(
    BASE_DIR / "data/processed/train_cleaned.csv"
)

X_train = train.drop(columns=["satisfaction", "id"])

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ],
    sparse_threshold=0
)

print("Transforming training data...")

X_transformed = preprocessor.fit_transform(X_train)

print("\nTRANSFORMED DATA DIAGNOSTICS")
print("Shape:", X_transformed.shape)
print("Data type:", X_transformed.dtype)

print(
    "NaN values:",
    np.isnan(X_transformed).sum()
)

print(
    "Infinite values:",
    np.isinf(X_transformed).sum()
)

print(
    "Maximum absolute value:",
    np.max(np.abs(X_transformed))
)

print(
    "Mean absolute value:",
    np.mean(np.abs(X_transformed))
)

print("\nTRANSFORMED FEATURE CHECK COMPLETED")
