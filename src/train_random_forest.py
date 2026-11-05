
import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent

train = pd.read_csv(
    BASE_DIR / "data/processed/train_cleaned.csv"
)

test = pd.read_csv(
    BASE_DIR / "data/processed/test_cleaned.csv"
)

X_train = train.drop(columns=["satisfaction", "id"])
y_train = train["satisfaction"]

X_test = test.drop(columns=["satisfaction", "id"])
y_test = test["satisfaction"]

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
        )
    ],
    remainder="passthrough"
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

print("Training Random Forest...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nRANDOM FOREST RESULTS")

print("Accuracy:",
      round(accuracy_score(y_test, predictions), 4))

print("Precision:",
      round(precision_score(
          y_test, predictions,
          pos_label="satisfied",
          zero_division=0
      ), 4))

print("Recall:",
      round(recall_score(
          y_test, predictions,
          pos_label="satisfied",
          zero_division=0
      ), 4))

print("F1 Score:",
      round(f1_score(
          y_test, predictions,
          pos_label="satisfied",
          zero_division=0
      ), 4))

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

print("\nCONFUSION MATRIX")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["neutral or dissatisfied", "satisfied"]
))

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

joblib.dump(
    model,
    MODEL_DIR / "airline_random_forest.pkl"
)

print("\nRandom Forest model saved successfully!")
