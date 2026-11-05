
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

results = pd.DataFrame([
    {
        "Model": "Dummy Classifier",
        "Accuracy": 0.5610,
        "Precision": 0.0000,
        "Recall": 0.0000,
        "F1 Score": 0.0000
    },
    {
        "Model": "Logistic Regression",
        "Accuracy": 0.8718,
        "Precision": 0.8694,
        "Recall": 0.8333,
        "F1 Score": 0.8509
    },
    {
        "Model": "Random Forest",
        "Accuracy": 0.9624,
        "Precision": 0.9699,
        "Recall": 0.9437,
        "F1 Score": 0.9566
    }
])

print("\nMODEL PERFORMANCE COMPARISON")
print(results.to_string(index=False))

output_path = BASE_DIR / "reports/model_comparison.csv"
results.to_csv(output_path, index=False)

print("\nComparison report saved:", output_path)
