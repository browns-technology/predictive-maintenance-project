import streamlit as st
import pandas as pd
import joblib

st.title("AI4I Machine Failure Prediction App")

# Load model
model = joblib.load("best_model_compressed.pkl")

# Columns the model was trained on
EXPECTED_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_H",
    "Type_L",
    "Type_M"
]

def preprocess(df):
    # Drop irrelevant columns
    cols_to_drop = ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # One-hot encode Type
    df = pd.get_dummies(df, columns=["Type"], drop_first=False)

    # Ensure all expected dummy columns exist
    for col in ["Type_H", "Type_L", "Type_M"]:
        if col not in df.columns:
            df[col] = 0

    # Keep only expected columns
    df = df[EXPECTED_COLUMNS]

    return df

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.write(df_raw.head())

    # Process the raw data to match model training format
    df_ready = preprocess(df_raw)

    # Predict
    predictions = model.predict(df_ready)

    st.subheader("Predictions")
    st.write(predictions)
