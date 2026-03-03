"""
Home.py — Streamlit entry point.

Run with:
    streamlit run app/Home.py
"""

import streamlit as st

st.set_page_config(
    page_title="Trading System Automation",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Trading System Automation")
st.subheader("Machine-learning predictions for stock trading signals")

st.markdown("""
Welcome to the Trading System Automation app.

Use the sidebar to navigate:
- **Go Live** — select a ticker and see the latest ML prediction
- **Backtesting** — evaluate the model's historical performance

---
### Team
| Name | Role |
|------|------|
| *Add your name* | ETL / Data Engineering |
| *Add your name* | ML Model |
| *Add your name* | API Wrapper |
| *Add your name* | Streamlit App |

---
### About
This app uses financial data from [SimFin](https://simfin.com/) and a scikit-learn
model to generate Buy / Sell / Hold signals for individual stock tickers.
""")
