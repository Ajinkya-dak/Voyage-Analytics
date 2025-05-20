import streamlit as st
import joblib
import pandas as pd

model = joblib.load('flight_price_linreg_pipeline.joblib')
st.title("Flight Price Predictor")

from_ = st.text_input("From (City/Code)")
to_ = st.text_input("To (City/Code)")
flight_type = st.selectbox("Flight Type", ['firstClass','premium','economic'])
agency = st.selectbox("Agency", ['FlyingDrops','Rainbow','CloudFy'])
time = st.number_input("Time (hours)", min_value=0.0)
distance = st.number_input("Distance (km)", min_value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        'from': from_, 'to': to_, 'flightType': flight_type,
        'agency': agency, 'time': time, 'distance': distance
    }])
    price = model.predict(input_df)[0]
    st.success(f"Predicted Price: ₹{price:.2f}")
