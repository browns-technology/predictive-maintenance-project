import streamlit as st
import pandas as pd
import joblib

st.title("AI4I Machine Failure Prediction App")

# Load the trained model
model = joblib.load("best_model_compressed.pkl")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.write(df.head())

    # Predict
    predictions = model.predict(df)

    st.subheader("Predictions")
    st.write(predictions)
