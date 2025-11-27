import streamlit as st
import pandas as pd
import joblib

st.title("AI4I Machine Failure Prediction App")

# Load model
model = joblib.load("best_model_compressed.pkl")

# The exact feature columns your model expects
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
    # Drop irrelevant columns if they exist
    cols_to_drop = ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(col, axis=1)

    # One-hot encode "Type"
    df = pd.get_dummies(df, columns=["Type"], drop_first=False)

    # Add missing dummy columns if needed
    for col in ["Type_H", "Type_L", "Type_M"]:
        if col not in df.columns:
            df[col] = 0

    # Ensure correct column order
    df = df[EXPECTED_COLUMNS]

    return df

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.write(df_raw.head())

    # Preprocess
    df_ready = preprocess(df_raw)

    # Predict
    predictions = model.predict(df_ready)

    st.subheader("Predictions")
    st.write(predictions)
