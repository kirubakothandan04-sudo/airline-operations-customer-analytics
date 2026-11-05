
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "airline_random_forest.pkl"

# Page configuration
st.set_page_config(
    page_title="Airline Passenger Analytics",
    page_icon="✈️",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# Dashboard heading
st.title("✈️ Airline Passenger Satisfaction Predictor")
st.write(
    "Enter passenger and flight details to predict "
    "the passenger's satisfaction level."
)

st.divider()

# Passenger information
st.header("1. Passenger Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])

with col2:
    customer_type = st.selectbox(
        "Customer Type",
        ["Loyal Customer", "disloyal Customer"]
    )

with col3:
    age = st.number_input(
        "Age", min_value=7, max_value=100, value=30
    )

# Travel information
st.header("2. Travel Information")

col1, col2, col3 = st.columns(3)

with col1:
    travel_type = st.selectbox(
        "Type of Travel",
        ["Business travel", "Personal Travel"]
    )

with col2:
    travel_class = st.selectbox(
        "Travel Class",
        ["Business", "Eco", "Eco Plus"]
    )

with col3:
    flight_distance = st.number_input(
        "Flight Distance",
        min_value=0,
        max_value=6000,
        value=1000
    )

# Flight delays
st.header("3. Flight Details")

col1, col2 = st.columns(2)

with col1:
    departure_delay = st.number_input(
        "Departure Delay (minutes)",
        min_value=0,
        max_value=2000,
        value=0
    )

with col2:
    arrival_delay = st.number_input(
        "Arrival Delay (minutes)",
        min_value=0,
        max_value=2000,
        value=0
    )

# Service ratings
st.header("4. Airline Service Ratings")
st.caption("Rate each service from 0 (lowest) to 5 (highest).")

rating_columns = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
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

ratings = {}

rating_col1, rating_col2 = st.columns(2)

for index, feature in enumerate(rating_columns):
    with (rating_col1 if index % 2 == 0 else rating_col2):
        ratings[feature] = st.slider(
            feature,
            min_value=0,
            max_value=5,
            value=3
        )

st.divider()

# Prediction
if st.button("Predict Passenger Satisfaction", type="primary"):

    input_data = {
        "Gender": gender,
        "Customer Type": customer_type,
        "Age": age,
        "Type of Travel": travel_type,
        "Class": travel_class,
        "Flight Distance": flight_distance,
        "Departure Delay in Minutes": departure_delay,
        "Arrival Delay in Minutes": arrival_delay
    }

    input_data.update(ratings)

    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    probabilities = model.predict_proba(input_df)[0]
    classes = model.classes_

    satisfied_index = list(classes).index("satisfied")
    satisfaction_probability = probabilities[satisfied_index]

    st.header("Prediction Result")

    if prediction == "satisfied":
        st.success("Predicted Result: Satisfied 😊")
    else:
        st.warning("Predicted Result: Neutral or Dissatisfied")

    st.metric(
        "Model Estimated Probability of Satisfaction",
        f"{satisfaction_probability * 100:.2f}%"
    )

    st.progress(float(satisfaction_probability))

    st.caption(
        "This is a model estimate based on the supplied information, "
        "not a guarantee of actual passenger satisfaction."
    )

st.divider()

st.caption(
    "Machine Learning Model: Random Forest | "
    "Training Dataset: Public Airline Passenger Satisfaction Dataset"
)
