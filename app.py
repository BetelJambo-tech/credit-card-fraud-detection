import streamlit as st
import pandas as pd
import joblib
import os

st.title("Credit Card Fraud Detection")
st.write("Enter transaction details and click Predict to check if it's fraud.")
st.caption("Predict whether a transaction is fraudulent using machine learning")

# Load model
model = joblib.load("src/fraud_model.pkl")

# Features
feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Input fields
user_input = {}
cols = st.columns(2)

for i, feature in enumerate(feature_names):
    with cols[i % 2]:
        user_input[feature] = st.number_input(feature, value=0.0)

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]

    if prediction == 1:
    st.error("⚠️ Fraudulent Transaction Detected")
else:
    st.success("✅ Legitimate Transaction")