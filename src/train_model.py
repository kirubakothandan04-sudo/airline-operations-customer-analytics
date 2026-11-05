
import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = BASE_DIR / "data/processed/train_cleaned.csv"
TEST_PATH = BASE_DIR / "data/processed/test_cleaned.csv"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

# Load cleaned datasets
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("=" * 55)
print("AIRLINE PASSENGER SATISFACTION PREDICTION")
print("=" * 55)

# Separate features and target
X_train = train.drop(columns=["satisfaction", "id"])
y_train = train["satisfaction"]

X_test = test.drop(columns=["satisfaction", "id"])
y_test = test["satisfaction"]

# Identify categorical and numerical columns
categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))
print("Categorical features:", len(categorical_features))
print("Numerical features:", len(numerical_features))

# Preprocessing
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

# Baseline model
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

# Logistic Regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        C=1.0,
        solver="lbfgs",
        random_state=42
    ))
])

print("\nTraining Logistic Regression...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Evaluation
def evaluate_model(name, actual, predicted):
    return {
        "Model": name,
        "Accuracy": accuracy_score(actual, predicted),

        "Precision": precision_score(
            actual,
            predicted,
            pos_label="satisfied",
            zero_division=0
        ),

        "Recall": recall_score(
            actual,
            predicted,
            pos_label="satisfied",
            zero_division=0
        ),

        "F1 Score": f1_score(
            actual,
            predicted,
            pos_label="satisfied",
            zero_division=0
        )
    }


results = pd.DataFrame([
    evaluate_model("Dummy Classifier", y_test, baseline_predictions),
    evaluate_model("Logistic Regression", y_test, predictions)
])

print("\nMODEL COMPARISON")
print(results.round(4).to_string(index=False))

print("\nCLASSIFICATION REPORT")
report = classification_report(y_test, predictions)
print(report)

print("\nCONFUSION MATRIX")
matrix = confusion_matrix(
    y_test,
    predictions,
    labels=["neutral or dissatisfied", "satisfied"]
)
print(matrix)

# Save model and reports
joblib.dump(model, MODEL_DIR / "airline_satisfaction_model.pkl")

results.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

with open(REPORT_DIR / "classification_report.txt", "w") as file:
    file.write(report)

print("\nModel saved successfully!")
print("Evaluation reports saved successfully!")
print("\nPREDICTIVE ANALYSIS COMPLETED!")
