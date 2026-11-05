
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "train_cleaned.csv"

st.set_page_config(
    page_title="Airline Business Analytics",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Airline Operations & Customer Experience Analytics")
st.caption("Business insights from the public Airline Passenger Satisfaction dataset")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.success(f"Dataset loaded successfully: {len(df):,} passengers")


# KPI calculations
total_passengers = len(df)

satisfied_count = (
    df["satisfaction"] == "satisfied"
).sum()

dissatisfied_count = total_passengers - satisfied_count

satisfaction_rate = (
    satisfied_count / total_passengers
) * 100

dissatisfaction_rate = (
    dissatisfied_count / total_passengers
) * 100

average_distance = df["Flight Distance"].mean()

# Dashboard KPI cards
st.subheader("Business Performance Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Passengers",
    f"{total_passengers:,}"
)

col2.metric(
    "Satisfaction Rate",
    f"{satisfaction_rate:.2f}%"
)

col3.metric(
    "Dissatisfaction Rate",
    f"{dissatisfaction_rate:.2f}%"
)

col4.metric(
    "Average Flight Distance",
    f"{average_distance:,.0f} miles"
)

st.divider()

st.header("Passenger Satisfaction Analysis")

col1, col2 = st.columns(2)

# Chart 1: Overall satisfaction distribution
with col1:
    st.subheader("Overall Satisfaction")

    satisfaction_counts = df["satisfaction"].value_counts()

    fig1, ax1 = plt.subplots(figsize=(6, 4))

    ax1.pie(
        satisfaction_counts.values,
        labels=satisfaction_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )

    ax1.set_title("Passenger Satisfaction Distribution")

    st.pyplot(fig1)
    plt.close(fig1)

# Chart 2: Satisfaction by travel class
with col2:
    st.subheader("Satisfaction by Travel Class")

    class_satisfaction = pd.crosstab(
        df["Class"],
        df["satisfaction"],
        normalize="index"
    ) * 100

    fig2, ax2 = plt.subplots(figsize=(6, 4))

    class_satisfaction.plot(
        kind="bar",
        stacked=True,
        ax=ax2
    )

    ax2.set_title("Satisfaction Percentage by Class")
    ax2.set_xlabel("Travel Class")
    ax2.set_ylabel("Percentage (%)")
    ax2.legend(title="Satisfaction")
    ax2.tick_params(axis="x", rotation=0)

    st.pyplot(fig2)
    plt.close(fig2)

st.divider()

st.header("Travel Purpose Analysis")

travel_satisfaction = pd.crosstab(
    df["Type of Travel"],
    df["satisfaction"],
    normalize="index"
) * 100

fig3, ax3 = plt.subplots(figsize=(9, 5))

travel_satisfaction.plot(
    kind="bar",
    stacked=True,
    ax=ax3
)

ax3.set_title("Passenger Satisfaction by Travel Purpose")
ax3.set_xlabel("Type of Travel")
ax3.set_ylabel("Percentage (%)")
ax3.legend(title="Satisfaction")
ax3.tick_params(axis="x", rotation=0)

st.pyplot(fig3)
plt.close(fig3)

st.info(
    "Business travel passengers have a 58.26% satisfaction rate, "
    "compared with 10.17% for personal travel passengers."
)

st.divider()

st.header("Airline Service Quality Analysis")

service_columns = [
    "Inflight wifi service",
    "Ease of Online booking",
    "Food and drink",
    "Online boarding",
    "Cleanliness",
    "Checkin service",
    "Leg room service",
    "Inflight entertainment",
    "On-board service",
    "Seat comfort",
    "Baggage handling",
    "Inflight service"
]

service_averages = (
    df[service_columns]
    .mean()
    .sort_values()
)

fig4, ax4 = plt.subplots(figsize=(10, 6))

service_averages.plot(
    kind="barh",
    ax=ax4
)

ax4.set_title("Average Passenger Service Ratings")
ax4.set_xlabel("Average Rating (0–5)")
ax4.set_ylabel("Service")

st.pyplot(fig4)
plt.close(fig4)

lowest_services = service_averages.head(3)

st.subheader("Services Requiring Attention")

for service, rating in lowest_services.items():
    st.write(f"**{service}:** {rating:.2f} / 5")

