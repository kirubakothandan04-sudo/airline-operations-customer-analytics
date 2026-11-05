# ✈️ Airline Operations & Customer Experience Analytics

An end-to-end airline analytics project using Python, SQL, Machine Learning, Tableau and Streamlit.

## Overview

This project analyzes a public Airline Passenger Satisfaction dataset containing 103,904 training records.

The objective is to identify passenger satisfaction patterns, analyze service quality, understand flight delays, and build a machine learning model to predict passenger satisfaction.

Note: This project uses public airline survey data and does not use United Airlines internal data.

## Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- MySQL
- Tableau Public
- Streamlit

## Key Features

- Data cleaning and preprocessing
- Exploratory data analysis
- SQL customer segmentation
- Passenger satisfaction analysis
- Flight delay analysis
- Tableau dashboard
- Machine learning model comparison
- Random Forest prediction application
- Interactive Streamlit analytics dashboard

## Model Performance

| Model | Accuracy |
|---|---:|
| Dummy Classifier | 56.10% |
| Logistic Regression | 87.18% |
| Random Forest | 96.24% |

Random Forest achieved the highest accuracy on the provided holdout test dataset.

## Key Findings

- Overall satisfaction: 43.33%
- Business class satisfaction: 69.43%
- Economy class satisfaction: 18.61%
- Business travel satisfaction: 58.26%
- Personal travel satisfaction: 10.17%

Inflight Wi-Fi and online booking received relatively low average service ratings.

These findings show associations, not causal relationships.

## Applications

Run the prediction application:

    streamlit run dashboard/app.py

Run the business analytics dashboard:

    streamlit run dashboard/analytics.py

## Installation

Clone the repository:

    git clone YOUR_GITHUB_REPOSITORY_URL
    cd Airline-Analytics-Project

Create a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

## Dataset

Download the dataset from:

https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

Place the CSV files in data/archive-2/.

## Limitations

- Uses public survey data, not proprietary airline data.
- Model performance applies to the supplied dataset and holdout split.
- Feature importance indicates predictive contribution, not causation.
- Predictions are estimates, not guarantees.

## Author

Kirubanandhan K.

Integrated M.Tech – Software Engineering

Vellore Institute of Technology
