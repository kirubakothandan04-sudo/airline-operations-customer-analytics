
import pandas as pd
import joblib

from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models/airline_random_forest.pkl"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(exist_ok=True)
FIGURE_DIR.mkdir(exist_ok=True)

# Load trained model
model = joblib.load(MODEL_PATH)

# Extract preprocessing and classifier
preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Get transformed feature names
feature_names = preprocessor.get_feature_names_out()

# Extract feature importance
importance = classifier.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTOP 15 IMPORTANT FEATURES")
print(importance_df.head(15).to_string(index=False))

# Save results
importance_df.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False
)

# Create chart
top_features = importance_df.head(15).sort_values(
    by="Importance"
)

plt.figure(figsize=(11, 7))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Passenger Experience Features")
plt.title("Top 15 Factors Influencing Satisfaction Predictions")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "feature_importance.png",
    dpi=300
)

plt.close()

print("\nFeature importance saved successfully!")
