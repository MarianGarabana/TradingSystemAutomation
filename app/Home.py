"""
Home.py — Streamlit entry point.

Run with:
    streamlit run app/Home.py
"""

import os

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Trading System Automation",
    page_icon="📈",
    layout="wide",
)

# ── Company metadata ───────────────────────────────────────────────────────────
# Hardcoded so the cloud deployment doesn't need the raw us-companies.csv (gitignored).
COMPANIES = {
    "AAPL": ("Apple Inc.",             "Technology"),
    "ABBV": ("AbbVie Inc.",            "Healthcare"),
    "ADBE": ("Adobe Inc.",             "Technology"),
    "AMD":  ("Advanced Micro Devices", "Technology"),
    "AMZN": ("Amazon.com Inc.",        "Technology"),
    "AVGO": ("Broadcom Inc.",          "Technology"),
    "BAC":  ("Bank of America Corp.",  "Financials"),
    "COST": ("Costco Wholesale Corp.", "Consumer"),
    "CRM":  ("Salesforce Inc.",        "Technology"),
    "DIS":  ("The Walt Disney Co.",    "Consumer"),
    "GOOG": ("Alphabet Inc.",          "Technology"),
    "GS":   ("Goldman Sachs Group",    "Financials"),
    "INTC": ("Intel Corp.",            "Technology"),
    "JNJ":  ("Johnson & Johnson",      "Healthcare"),
    "JPM":  ("JPMorgan Chase & Co.",   "Financials"),
    "KO":   ("The Coca-Cola Co.",      "Consumer"),
    "MA":   ("Mastercard Inc.",        "Financials"),
    "MCD":  ("McDonald's Corp.",       "Consumer"),
    "META": ("Meta Platforms Inc.",    "Technology"),
    "MSFT": ("Microsoft Corp.",        "Technology"),
    "NFLX": ("Netflix Inc.",           "Technology"),
    "NVDA": ("NVIDIA Corp.",           "Technology"),
    "ORCL": ("Oracle Corp.",           "Technology"),
    "PEP": ("PepsiCo Inc.",           "Consumer"),
    "PFE":  ("Pfizer Inc.",            "Healthcare"),
    "PLTR": ("Palantir Technologies",  "Technology"),
    "QCOM": ("Qualcomm Inc.",          "Technology"),
    "TSLA": ("Tesla Inc.",             "Technology"),
    "UNH":  ("UnitedHealth Group",     "Healthcare"),
    "V":    ("Visa Inc.",              "Financials"),
    "WMT":  ("Walmart Inc.",           "Consumer"),
}


@st.cache_data
def load_ticker_table() -> pd.DataFrame:
    """Build the ticker table from processed CSVs + hardcoded company metadata."""
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

    if not os.path.exists(processed_dir):
        return pd.DataFrame(columns=["Ticker", "Company", "Sector"])

    processed_tickers = sorted(
        f.replace(".csv", "")
        for f in os.listdir(processed_dir)
        if f.endswith(".csv") and not f.startswith(".")
    )

    rows = []
    for ticker in processed_tickers:
        company, sector = COMPANIES.get(ticker, ("—", "—"))
        rows.append({"Ticker": ticker, "Company": company, "Sector": sector})

    return pd.DataFrame(rows)


ticker_df = load_ticker_table()

# ── Hero banner ────────────────────────────────────────────────────────────────
st.title("📈 Trading System Automation")
st.markdown(
    "##### ML-powered daily trading signals for US stocks — know what to bet on."
)

# ── Key metrics ────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Supported tickers", len(ticker_df))
col2.metric("Data range", "2020 – 2024")
col3.metric("Model type", "LogReg + GBC")
col4.metric("Data source", "SimFin")

st.divider()

# ── Feature cards ──────────────────────────────────────────────────────────────
# Force equal height on the three bordered containers.
left, mid, right = st.columns(3, gap="medium")

with left:
    with st.container(border=True):
        st.markdown("### 🚀 Go Live")
        st.markdown(
            "Select a ticker and get **today's ML-generated signal** "
            "— Buy, Sell, or Hold — powered by live SimFin API data."
        )
        st.markdown("- Real-time price data via SimFin API")
        st.markdown("- BUY / SELL / HOLD with confidence score")
        st.markdown("- RSI, MACD, Bollinger Bands charts")
        st.page_link("pages/1_go_live.py", label="Open Go Live →", icon="🚀")

with mid:
    with st.container(border=True):
        st.markdown("### 🎯 Prediction Bet")
        st.markdown(
            "Choose a stock, set your **bet direction and leverage**, "
            "then compare the model's call with what actually happened."
        )
        st.markdown("- UP / DOWN direction with 1× – 10× leverage")
        st.markdown("- Your bet vs. AI signal side-by-side")
        st.markdown("- Entry / exit chart + top feature drivers")
        st.page_link("pages/2_prediction_bet.py", label="Open Prediction Bet →", icon="🎯")

with right:
    with st.container(border=True):
        st.markdown("### 🔍 Backtesting")
        st.markdown(
            "Explore the model's **full historical track record** "
            "— returns, rolling accuracy, and every past signal in one view."
        )
        st.markdown("- Strategy return vs. buy-and-hold baseline")
        st.markdown("- Rolling 30-day prediction accuracy chart")
        st.markdown("- Complete signal history with dates & labels")
        st.page_link("pages/3_backtesting.py", label="Open Backtesting →", icon="🔍")

st.divider()

# ── How it works ───────────────────────────────────────────────────────────────
st.subheader("How it works")

tab1, tab2, tab3 = st.tabs(["📥 Data & ETL", "🧠 ML model", "🌐 Live predictions"])

with tab1:
    st.markdown(
        """
        We download **5 years of daily price data** (2020–2024) from SimFin's bulk download
        for 29 US companies across four sectors. The ETL pipeline:

        1. **Cleans** price errors — outlier returns are detected and forward-filled
        2. **Engineers technical indicators** — moving averages, RSI, MACD, Bollinger Bands,
           volatility-normalised returns, and quarterly fundamental ratios (margins, leverage, cash flow)
        3. **Exports** a clean processed CSV per ticker, ready for model training
        """
    )

with tab2:
    st.markdown(
        """
        We train **two pooled classification models**, one per ticker group:

        - **Standard model** — 26 tickers (Technology, Healthcare, Consumer)
        - **Fallback model** — 5 Financial tickers (Banks & Payment Networks)

        Four candidate algorithms are evaluated — Logistic Regression, Random Forest,
        Gradient Boosting, and LightGBM — using **time-series cross-validation**
        (no look-ahead bias). The best-performing model is exported as a `.pkl` file.

        **Target:** next-day direction — price goes up (1) or down (0).
        """
    )

with tab3:
    st.markdown(
        """
        When you open the **Go Live** page:

        1. The app fetches the last 200 days of prices from the **SimFin API** (live)
        2. The same ETL transformations are applied — guaranteeing feature consistency
           with what the model was trained on (no train/serve skew)
        3. The exported model predicts tomorrow's price direction
        4. A **BUY / SELL / HOLD** signal is generated based on prediction confidence

        If the API is unavailable, the app falls back to the latest processed CSV and
        shows a ⚠️ badge so you always know the data freshness.
        """
    )

st.divider()

# ── The team ───────────────────────────────────────────────────────────────────
st.subheader("The team")

st.markdown(
    """
    <div style="display:flex;justify-content:center;align-items:center;
                flex-wrap:wrap;gap:0;padding:12px 0;">
        <span style="font-size:15px;color:#f0f0f0;">Omar Ajlouni</span>
        <span style="color:#444;padding:0 16px;font-size:18px;">·</span>
        <span style="font-size:15px;color:#f0f0f0;">Marian Garabana</span>
        <span style="color:#444;padding:0 16px;font-size:18px;">·</span>
        <span style="font-size:15px;color:#f0f0f0;">Marco Ortiz</span>
        <span style="color:#444;padding:0 16px;font-size:18px;">·</span>
        <span style="font-size:15px;color:#f0f0f0;">Lorenz Rösgen</span>
        <span style="color:#444;padding:0 16px;font-size:18px;">·</span>
        <span style="font-size:15px;color:#f0f0f0;">Jorge Vildoso</span>
        <span style="color:#444;padding:0 16px;font-size:18px;">·</span>
        <span style="font-size:15px;color:#f0f0f0;">Yaxin Wu</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# t1, t2 = st.columns(2, gap="large")
#
# with t1:
#     with st.container(border=True):
#         st.markdown("**Marian Garabana**")
#         st.caption(
#             "ETL pipeline · SimFin API wrapper · Streamlit web app · Cloud deployment"
#         )
#
# with t2:
#     with st.container(border=True):
#         st.markdown("**Jorge Vildoso**")
#         st.caption(
#             "Feature engineering · ML model training · Model evaluation · Trading strategy"
#         )

st.divider()

# ── Supported companies ────────────────────────────────────────────────────────
st.subheader("Supported companies")

if not ticker_df.empty:

    # Filters
    filter_col, search_col = st.columns([1, 3])

    with filter_col:
        sectors = ["All sectors"] + sorted(ticker_df["Sector"].unique().tolist())
        selected_sector = st.selectbox('Sector',sectors)

    with search_col:
        search = st.text_input(
            label="Search",
            placeholder="Search by ticker or company name (e.g. AAPL or Apple)...",
            label_visibility="hidden",
        )

    # Apply filters
    display_df = ticker_df.copy()
    if selected_sector != "All sectors":
        display_df = display_df[display_df["Sector"] == selected_sector]
    if search:
        mask = (
            display_df["Ticker"].str.contains(search, case=False, na=False)
            | display_df["Company"].str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]

    # Table + donut chart side by side
    table_col, chart_col = st.columns([3, 2])

    with table_col:
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Ticker": st.column_config.TextColumn("Ticker", pinned=True),
                "Company": st.column_config.TextColumn("Company"),
                "Sector": st.column_config.TextColumn("Sector"),
            },
        )
        st.caption(f"Showing {len(display_df)} of {len(ticker_df)} tickers")

    with chart_col:
        sector_counts = ticker_df.groupby("Sector").size().reset_index(name="Count")
        fig = px.pie(
            sector_counts,
            names="Sector",
            values="Count",
            title="Tickers by sector",
            hole=0.45,
            color_discrete_sequence=["#89F336", "#D1E231", "#f6c86a", "#ADEBB3"],
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(
            showlegend=False,
            height=350,
            margin=dict(t=50, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f0f0f0",
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info(
        "No processed tickers yet. Run `python etl/etl.py --ticker AAPL` to get started."
    )
