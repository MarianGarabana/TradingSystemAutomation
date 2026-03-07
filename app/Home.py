"""
Home.py — Streamlit entry point.

Run with:
    streamlit run app/Home.py
"""

import os

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Trading System Automation",
    page_icon="📈",
    layout="wide",
)

# ── Robinhood-inspired theme: lime green + black ───────────────────────────────



@st.cache_data
def load_ticker_table() -> pd.DataFrame:
    """Load processed tickers and enrich with company name from us-companies.csv.

    We only show tickers that have a processed CSV — those are the ones
    the app can actually make predictions for.
    """
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    companies_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "us-companies.csv")

    if not os.path.exists(processed_dir):
        return pd.DataFrame(columns=["Ticker", "Company Name"])

    # Get all tickers that have a processed CSV file
    processed_tickers = sorted(
        f.replace(".csv", "")
        for f in os.listdir(processed_dir)
        if f.endswith(".csv")
    )

    if not processed_tickers:
        return pd.DataFrame(columns=["Ticker", "Company Name"])

    # Enrich with company name from the raw companies file
    if os.path.exists(companies_path):
        companies = pd.read_csv(companies_path, sep=";", usecols=["Ticker", "Company Name"])
        companies = companies.dropna(subset=["Ticker"])
        # Keep only the tickers we have processed data for
        df = companies[companies["Ticker"].isin(processed_tickers)].reset_index(drop=True)
    else:
        # Fallback: tickers only, no company name
        df = pd.DataFrame({"Ticker": processed_tickers, "Company Name": ["—"] * len(processed_tickers)})

    return df[["Ticker", "Company Name"]]


# Load ticker data once (cached)
ticker_df = load_ticker_table()

# ── Hero banner ────────────────────────────────────────────────────────────────
st.title("📈 Trading System Automation")
st.markdown("##### *Know what you should bet on!*")

# ── Metrics row ────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Supported Tickers", len(ticker_df))
col2.metric("Data Range", "2020 – 2024")
col3.metric("Model", "Regression")
col4.metric("Data Source", "SimFin")

st.divider()

# ── Feature cards: Go Live + Backtesting ──────────────────────────────────────
left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.markdown("### 🚀 Go Live")
        st.markdown(
            "Select a stock ticker and get the **latest ML prediction** — "
            "is it a Buy, Sell, or Hold?"
        )
        st.markdown("- Choose from all supported tickers")
        st.markdown("- See the predicted signal with confidence")
        st.markdown("- View key indicators that drove the prediction")
        st.page_link("pages/go_live.py", label="Open Go Live →", icon="🚀")

with right:
    with st.container(border=True):
        st.markdown("### 🔍 Backtesting")
        st.markdown(
            "Evaluate the model's **historical performance** — "
            "how well did it predict past price movements?"
        )
        st.markdown("- Accuracy and win rate over time")
        st.markdown("- Cumulative return vs. buy-and-hold")
        st.markdown("- Per-ticker breakdown")
        st.page_link("pages/backtesting.py", label="Open Backtesting →", icon="🔍")

st.divider()

# ── Searchable ticker table ────────────────────────────────────────────────────
st.markdown("#### Supported Tickers")

if not ticker_df.empty:
    # Text input to filter the table by ticker or company name
    search = st.text_input(
        label="Search",
        placeholder="Search by ticker or company name (e.g. AAPL or Apple)...",
        label_visibility="collapsed",
    )

    if search:
        # Filter rows where ticker OR company name contains the search string
        mask = (
            ticker_df["Ticker"].str.contains(search, case=False, na=False)
            | ticker_df["Company Name"].str.contains(search, case=False, na=False)
        )
        display_df = ticker_df[mask]
    else:
        display_df = ticker_df

    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("No processed tickers yet. Run `python etl/etl.py --ticker AAPL` to get started.")
