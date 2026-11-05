
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "train_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cleaned dataset
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("AIRLINE CUSTOMER EXPERIENCE - DATA ANALYSIS")
print("=" * 60)

# 1. Overall satisfaction
satisfaction = df["satisfaction"].value_counts(normalize=True) * 100

print("\n1. OVERALL SATISFACTION (%)")
print(satisfaction.round(2))

satisfaction.plot(kind="bar", title="Passenger Satisfaction Distribution")
plt.ylabel("Percentage")
plt.xlabel("Satisfaction")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "satisfaction_distribution.png")
plt.close()

# 2. Satisfaction by travel class
class_analysis = pd.crosstab(
    df["Class"],
    df["satisfaction"],
    normalize="index"
) * 100

print("\n2. SATISFACTION BY TRAVEL CLASS (%)")
print(class_analysis.round(2))

class_analysis.plot(
    kind="bar",
    stacked=True,
    title="Passenger Satisfaction by Travel Class"
)
plt.ylabel("Percentage")
plt.xlabel("Travel Class")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "satisfaction_by_class.png")
plt.close()

# 3. Satisfaction by travel purpose
travel_analysis = pd.crosstab(
    df["Type of Travel"],
    df["satisfaction"],
    normalize="index"
) * 100

print("\n3. SATISFACTION BY TRAVEL PURPOSE (%)")
print(travel_analysis.round(2))

travel_analysis.plot(
    kind="bar",
    stacked=True,
    title="Satisfaction by Type of Travel"
)
plt.ylabel("Percentage")
plt.xlabel("Travel Purpose")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "satisfaction_by_travel_type.png")
plt.close()

# 4. Average service ratings
service_columns = [
    "Inflight wifi service",
    "Ease of Online booking",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness"
]

service_ratings = df[service_columns].mean().sort_values()

print("\n4. AVERAGE SERVICE RATINGS")
print(service_ratings.round(2))

service_ratings.plot(
    kind="barh",
    title="Average Passenger Service Ratings"
)
plt.xlabel("Average Rating")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "service_ratings.png")
plt.close()

# 5. Flight delay analysis
delay_analysis = df.groupby("satisfaction")[
    ["Departure Delay in Minutes", "Arrival Delay in Minutes"]
].mean()

print("\n5. AVERAGE FLIGHT DELAYS")
print(delay_analysis.round(2))

delay_analysis.plot(
    kind="bar",
    title="Average Flight Delays by Satisfaction"
)
plt.ylabel("Delay in Minutes")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "flight_delays.png")
plt.close()

print("\nAll charts saved successfully!")
print("Location:", OUTPUT_DIR)
