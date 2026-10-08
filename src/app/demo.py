"""Streamlit demo: samples a real transaction and sends it to the API.

Run: streamlit run src/app/demo.py
Requires the API running separately: uvicorn src.api.main:app --reload
"""

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "creditcard.csv"

st.set_page_config(page_title="Fraud Detection Demo", page_icon="💳")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def check_health() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


st.title("Credit Card Fraud Detection — Demo")
st.caption(
    "Samples a real transaction from the Kaggle creditcardfraud dataset "
    "and sends it to the FastAPI prediction service."
)

health = check_health()
if health is None:
    st.error(
        f"API not reachable at {API_URL}. Start it with "
        "`uvicorn src.api.main:app --reload`."
    )
    st.stop()
elif not health.get("model_loaded"):
    st.warning("API is up but the model failed to load — check its logs.")

if not DATA_PATH.exists():
    st.error(f"{DATA_PATH} not found. Run ./scripts/download_data.sh first.")
    st.stop()

df = load_data()

if "current_row" not in st.session_state:
    st.session_state.current_row = df[df["Class"] == 0].sample(1).iloc[0]

col1, col2 = st.columns(2)
if col1.button("Sample a legit transaction"):
    st.session_state.current_row = df[df["Class"] == 0].sample(1).iloc[0]
if col2.button("Sample a fraudulent transaction"):
    st.session_state.current_row = df[df["Class"] == 1].sample(1).iloc[0]

row = st.session_state.current_row
transaction = row.drop("Class").to_dict()

st.subheader("Transaction")
st.dataframe(pd.DataFrame([transaction]), width="stretch")
st.caption(
    f"True label (from the dataset, not shown to the model): "
    f"{'fraud' if row['Class'] == 1 else 'legit'}"
)

if st.button("Predict", type="primary"):
    try:
        response = requests.post(f"{API_URL}/predict", json=transaction, timeout=5)
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as e:
        st.error(f"Request to API failed: {e}")
    else:
        st.metric("Fraud probability", f"{result['fraud_probability']:.4%}")
        st.write(f"Decision threshold: {result['threshold']:.4f}")
        if result["is_fraud"]:
            st.error("Model flags this as **FRAUD**")
        else:
            st.success("Model flags this as **legit**")
