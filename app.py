import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
st.sidebar.title("💳 Fraud Detection App")

st.sidebar.markdown("""
### About
This app uses a machine learning model to detect fraudulent credit card transactions.

### Features
- Real-time prediction
- Fraud probability score
- Interactive input

### Tech Stack
- Python
- Scikit-learn
- Streamlit
""")

# Load model
model = joblib.load("src/fraud_model.pkl")

# Page title
st.title("💳 Credit Card Fraud Detection")
st.caption("Predict whether a transaction is fraudulent using a machine learning model.")
st.info("Enter transaction feature values below, then click Predict.")
st.write("Use the sample values button for a quick demo, or enter custom values and click Predict Fraud.")

st.markdown("---")
st.sidebar.markdown("### Model Performance")
st.sidebar.write("Accuracy: 99%+")
st.sidebar.write("Fraud Recall: 76%")
st.sidebar.write("Fraud Precision: 96%")
st.subheader("📊 Enter Transaction Details")

# Features
feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Session state for sample values
if "user_input" not in st.session_state:
    st.session_state.user_input = {feature: 0.0 for feature in feature_names}

# Buttons
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Use Sample Values"):
        for feature in feature_names:
            st.session_state.user_input[feature] = 0.1
        st.session_state.user_input["Amount"] = 100.0

with col2:
    if st.button("Reset Values"):
        st.session_state.user_input = {feature: 0.0 for feature in feature_names}


# Two-column input layout
cols = st.columns(2)
user_input = {}

for i, feature in enumerate(feature_names):
    with cols[i % 2]:
        user_input[feature] = st.number_input(
            feature,
            value=float(st.session_state.user_input[feature]),
            format="%.2f",
            key=feature
        )

# Prediction
if st.button("Predict Fraud"):
    input_df = pd.DataFrame([user_input])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

    st.write(f"**Fraud Probability:** {probability:.4f}")

    st.progress(min(float(probability), 1.0))
    st.markdown("### About the Model")
st.write("This app uses a Random Forest classifier trained on anonymized credit card transaction data to predict whether a transaction is fraudulent.")

# Footer
st.markdown("---")
st.caption("Built with Python, Scikit-learn, Pandas, and Streamlit")
st.caption("Built by Betel | Machine Learning Project")