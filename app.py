import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="AI4I Machine Failure Prediction", page_icon="⚙️", layout="wide")

# Global CSS theme (dark background + neon accents + animated sidebar icons)
st.markdown("""
    <style>
    body, .stApp {
        background-color: #0e0e0e;
        color: #e0e0e0;
    }
    h1 {
        text-align: center;
        font-family: 'Orbitron', sans-serif;
        color: #00ffcc;
        text-shadow: 0 0 15px #9b59b6, 0 0 25px #3498db;
        animation: glow 2s ease-in-out infinite alternate;
    }
    @keyframes glow {
        from { text-shadow: 0 0 15px #9b59b6; }
        to { text-shadow: 0 0 25px #3498db, 0 0 35px #00ffcc; }
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00ff99, #9b59b6, #3498db);
        animation: pulse 3s infinite;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 10px #00ff99; }
        33% { box-shadow: 0 0 20px #9b59b6; }
        66% { box-shadow: 0 0 20px #3498db; }
        100% { box-shadow: 0 0 10px #00ff99; }
    }
    .stTabs [role="tablist"] button {
        background-color: #1c1c1c;
        color: #00ffcc;
        border-radius: 5px;
    }
    .stTabs [role="tablist"] button[data-selected="true"] {
        background-color: #2c2c2c;
        color: #9b59b6;
        font-weight: bold;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #111111;
        border-right: 2px solid #3498db;
    }
    [data-testid="stSidebar"] h2 {
        color: #00ffcc;
        text-shadow: 0 0 10px #9b59b6;
    }
    [data-testid="stFileUploader"] {
        background-color: #1c1c1c;
        border: 2px dashed #9b59b6;
        border-radius: 10px;
        padding: 10px;
    }
    [data-testid="stFileUploader"] div {
        color: #00ffcc !important;
    }
    /* Animated gear icon */
    .gear {
        width: 40px;
        height: 40px;
        fill: #00ffcc;
        animation: spin 6s linear infinite;
        margin-right: 10px;
    }
    @keyframes spin {
        100% { transform: rotate(360deg); }
    }
    /* Pulsing spark icon */
    .spark {
        width: 20px;
        height: 20px;
        fill: #9b59b6;
        animation: sparkPulse 2s infinite;
        margin-left: 5px;
    }
    @keyframes sparkPulse {
        0% { opacity: 0.3; transform: scale(0.8); }
        50% { opacity: 1; transform: scale(1.2); }
        100% { opacity: 0.3; transform: scale(0.8); }
    }
    </style>
""", unsafe_allow_html=True)

st.title("AI4I Machine Failure Prediction Dashboard ⚡")

# Sidebar content with animated icons
st.sidebar.header("⚙️ Control Panel")
st.sidebar.markdown("""
<svg class="gear" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
<path d="M487.4 315.7l-42.6-24.6c2.6-13.5 4.1-27.4 4.1-41.6s-1.4-28.1-4.1-41.6l42.6-24.6c15.1-8.7 20.3-28.1 11.6-43.2l-43.2-74.8c-8.7-15.1-28.1-20.3-43.2-11.6l-42.6 24.6c-22.1-19.1-47.5-34.1-75.4-43.2V24.6C295.7 9.5 280.2 0 263.7 0h-87.4c-16.5 0-32 9.5-39.7 24.6v49.2c-27.9 9.1-53.3 24.1-75.4 43.2L18.6 92.5c-15.1-8.7-34.5-3.5-43.2 11.6L-68 179c-8.7 15.1-3.5 34.5 11.6 43.2l42.6 24.6c-2.6 13.5-4.1 27.4-4.1 41.6s1.4 28.1 4.1 41.6l-42.6 24.6c-15.1 8.7-20.3 28.1-11.6 43.2l43.2 74.8c8.7 15.1 28.1 20.3 43.2 11.6l42.6-24.6c22.1 19.1 47.5 34.1 75.4 43.2v49.2c7.7 15.1 23.2 24.6 39.7 24.6h87.4c16.5 0 32-9.5 39.7-24.6v-49.2c27.9-9.1 53.3-24.1 75.4-43.2l42.6 24.6c15.1 8.7 34.5 3.5 43.2-11.6l43.2-74.8c8.7-15.1 3.5-34.5-11.6-43.2z"/>
</svg>
<svg class="spark" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<circle cx="32" cy="32" r="10"/>
</svg>
""", unsafe_allow_html=True)

st.sidebar.markdown("Upload your dataset and explore machine analytics in neon style.")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📂 Upload a CSV file", type=["csv"])
st.sidebar.markdown("---")
st.sidebar.success("Theme: Neon Green + Purple + Blue")

# Load trained model
model = joblib.load("best_model_compressed.pkl")

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
    drop_cols = ["UDI", "Product ID", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(c, axis=1)

    df = pd.get_dummies(df, columns=["Type"], drop_first=False)
    if "Type_L" not in df.columns: df["Type_L"] = 0
    if "Type_M" not in df.columns: df["Type_M"] = 0

    df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["RPM_norm"] = df["Rotational speed [rpm]"] / df["Rotational speed [rpm]"].max()
    df["Torque_per_wear"] = df["Torque [Nm]"] / (df["Tool wear [min]"] + 1)

    return df[EXPECTED_COLUMNS]

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)

    # Tabs for better navigation
    tab1, tab2, tab3 = st.tabs(["📊 Data Preview", "⚙️ Predictions", "📈 Analytics"])

    with
