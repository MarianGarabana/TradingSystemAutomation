"""
go_live.py — Ticker selector, latest predictions, and trading signals.
"""

import os

import joblib
import streamlit as st

st.set_page_config(page_title="Go Live", page_icon="🚀", layout="wide")
st.title("🚀 Go Live — Real-Time Predictions")

TRAINED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "model", "trained")


def available_tickers() -> list[str]:
    """Return tickers that have a trained model on disk."""
    if not os.path.exists(TRAINED_DIR):
        return []
    files = [f for f in os.listdir(TRAINED_DIR) if f.startswith("model_") and f.endswith(".pkl")]
    return [f.replace("model_", "").replace(".pkl", "") for f in files]


tickers = available_tickers()

if not tickers:
    st.warning("No trained models found. Run `python model/train.py --ticker <TICKER>` first.")
    st.stop()

ticker = st.selectbox("Select a ticker", tickers)

if st.button("Predict"):
    model_path = os.path.join(TRAINED_DIR, f"model_{ticker}.pkl")
    model = joblib.load(model_path)

    st.info("TODO: load latest processed data and call model.predict()")
    # TODO: load the latest row from data/processed/<ticker>.csv
    # prediction = model.predict(X_latest)[0]
    # signal = prediction_to_signal(prediction)
    # st.metric("Signal", signal)
