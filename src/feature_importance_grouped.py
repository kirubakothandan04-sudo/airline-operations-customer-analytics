
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "reports/feature_importance.csv"
OUTPUT_PATH = BASE_DIR / "reports/feature_importance_grouped.csv"

df = pd.read_csv(INPUT_PATH)

def get_original_feature(feature):
    feature = feature.replace("remainder__", "")
    feature = feature.replace("categorical__", "")

    categorical_columns = [
        "Class",
        "Type of Travel",
        "Customer Type",
        "Gender"
    ]

    for column in categorical_columns:
        if feature.startswith(column + "_"):
            return column

    return feature

df["Original Feature"] = df["Feature"].apply(
    get_original_feature
)

grouped = (
    df.groupby("Original Feature")["Importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

grouped.to_csv(OUTPUT_PATH, index=False)

print("\nFEATURE IMPORTANCE BY ORIGINAL VARIABLE")
print(grouped.head(15).to_string(index=False))

print("\nGrouped feature importance saved!")
