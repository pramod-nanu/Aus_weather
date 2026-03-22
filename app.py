import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# Load model files
# =========================
model = joblib.load("logistic_model.pkl")
model_columns = joblib.load("model_columns.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Rain Prediction App (Australia)")

st.write("Enter weather details to predict rain tomorrow")

# =========================
# Extract feature groups
# =========================
location_cols = [col for col in model_columns if col.startswith("Location_")]
windgust_cols = [col for col in model_columns if col.startswith("WindGustDir_")]
wind9am_cols = [col for col in model_columns if col.startswith("WindDir9am_")]
wind3pm_cols = [col for col in model_columns if col.startswith("WindDir3pm_")]

# Convert to readable values
locations = [col.replace("Location_", "") for col in location_cols]
windgust_dirs = [col.replace("WindGustDir_", "") for col in windgust_cols]
wind9am_dirs = [col.replace("WindDir9am_", "") for col in wind9am_cols]
wind3pm_dirs = [col.replace("WindDir3pm_", "") for col in wind3pm_cols]

# =========================
# User Inputs
# =========================

# Numeric Inputs
MinTemp = st.number_input("MinTemp", value=10.0)
Rainfall = st.number_input("Rainfall", value=0.0)
Evaporation = st.number_input("Evaporation", value=5.0)
Sunshine = st.number_input("Sunshine", value=8.0)
WindSpeed9am = st.number_input("WindSpeed9am", value=10.0)
WindSpeed3pm = st.number_input("WindSpeed3pm", value=15.0)
Humidity3pm = st.number_input("Humidity3pm", value=50.0)
Cloud9am = st.number_input("Cloud9am", value=5.0)
Cloud3pm = st.number_input("Cloud3pm", value=5.0)

# Categorical Inputs
selected_location = st.selectbox("Location", locations)
selected_windgust = st.selectbox("WindGustDir", windgust_dirs)
selected_wind9am = st.selectbox("WindDir9am", wind9am_dirs)
selected_wind3pm = st.selectbox("WindDir3pm", wind3pm_dirs)

RainToday = st.selectbox("RainToday", ["No", "Yes"])

# =========================
# Prediction
# =========================
if st.button("Predict"):

    # Create empty dataframe with all columns
    input_df = pd.DataFrame(columns=model_columns)
    input_df.loc[0] = 0

    # Fill numeric values
    input_df['MinTemp'] = MinTemp
    input_df['Rainfall'] = Rainfall
    input_df['Evaporation'] = Evaporation
    input_df['Sunshine'] = Sunshine
    input_df['WindSpeed9am'] = WindSpeed9am
    input_df['WindSpeed3pm'] = WindSpeed3pm
    input_df['Humidity3pm'] = Humidity3pm
    input_df['Cloud9am'] = Cloud9am
    input_df['Cloud3pm'] = Cloud3pm

    # One-hot encoding manually
    input_df[f"Location_{selected_location}"] = 1
    input_df[f"WindGustDir_{selected_windgust}"] = 1
    input_df[f"WindDir9am_{selected_wind9am}"] = 1
    input_df[f"WindDir3pm_{selected_wind3pm}"] = 1

    if RainToday == "Yes":
        input_df["RainToday_Yes"] = 1

    # Fill missing values
    input_df = input_df.fillna(0)

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Output
    st.subheader("Result")

    if prediction == 1:
        st.error(f" Rain Expected (Probability: {round(probability,2)})")
    else:
        st.success(f"  No Rain (Probability: {round(probability,2)})")