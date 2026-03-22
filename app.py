import streamlit as st
import pandas as pd
import joblib

encoders = joblib.load("encoders.pkl")
scaler = joblib.load("scaler.pkl")
model = joblib.load("logistic_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Rain Prediction App (Australia)")
st.write("Enter weather details to predict rain tomorrow")



MinTemp = st.number_input("MinTemp", value=10.0)

Rainfall = st.number_input("Rainfall", value=0.0)

Evaporation = st.number_input("Evaporation", value=5.0)

Sunshine = st.number_input("Sunshine", value=8.0)

WindGustSpeed = st.number_input("WindGustSpeed", value=30.0)

WindSpeed9am = st.number_input("WindSpeed9am", value=10.0)

WindSpeed3pm = st.number_input("WindSpeed3pm", value=15.0)

Humidity3pm = st.number_input("Humidity3pm", value=50.0)

Cloud9am = st.number_input("Cloud9am", value=5.0)

Cloud3pm = st.number_input("Cloud3pm", value=5.0)

location = st.selectbox("Location", encoders['Location'].classes_)
windgust = st.selectbox("WindGustDir", encoders['WindGustDir'].classes_)
wind9am = st.selectbox("WindDir9am", encoders['WindDir9am'].classes_)
wind3pm = st.selectbox("WindDir3pm", encoders['WindDir3pm'].classes_)

RainToday = st.selectbox("RainToday", ["No", "Yes"])

if st.button("Predict"):
    st.snow()

    input_df = pd.DataFrame([{
        'Location': location,
        'MinTemp': MinTemp,
        'Rainfall': Rainfall,
        'Evaporation': Evaporation,
        'Sunshine': Sunshine,
        'WindGustDir': windgust,
        'WindGustSpeed': WindGustSpeed,
        'WindDir9am': wind9am,
        'WindDir3pm': wind3pm,
        'WindSpeed9am': WindSpeed9am,
        'WindSpeed3pm': WindSpeed3pm,
        'Humidity3pm': Humidity3pm,
        'Cloud9am': Cloud9am,
        'Cloud3pm': Cloud3pm,
        'RainToday': RainToday
    }])

    for col in encoders.keys():
        if col in input_df.columns:
            try:
                input_df[col] = encoders[col].transform(input_df[col])
            except:
                input_df[col] = 0

   
    input_df['RainToday'] = input_df['RainToday'].map({'No': 0, 'Yes': 1})

  
    input_df = input_df[model_columns]

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Result")

    if prediction == 1:
        st.error(f" Rain Expected (Probability: {round(probability,2)})")
    else:
        st.success(f" No Rain (Probability: {round(probability,2)})")
