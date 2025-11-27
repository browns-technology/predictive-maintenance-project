import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("AI4I Machine Failure Prediction App")

# Load trained model
model = joblib.load("best_model_compressed.pkl")

# Exact feature order expected by the model
EXPECTED_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_L",
    "Type_M",
    "Temp_diff",
    "RPM_norm",
    "Torque_per_wear"
]

def preprocess(df):
    # Drop columns not used
    drop_cols = ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(c, axis=1)

    # Handle Type one-hot encoding
    df = pd.get_dummies(df, columns=["Type"], drop_first=False)

    # Ensure dummy columns exist
    if "Type_L" not in df.columns:
        df["Type_L"] = 0
    if "Type_M" not in df.columns:
        df["Type_M"] = 0

    # Create engineered features
    df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["RPM_norm"] = df["Rotational speed [rpm]"] / df["Rotational speed [rpm]"].max()
    df["Torque_per_wear"] = df["Torque [Nm]"] / (df["Tool wear [min]"] + 1)

    # Keep only expected columns and ensure correct order
    df = df[EXPECTED_COLUMNS]

    return df

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.write(df_raw.head())

    # Preprocess input CSV to match training data
    df_ready = preprocess(df_raw)

    # Predict
    predictions = model.predict(df_ready)

    st.subheader("Predictions")
    st.write(predictions)
