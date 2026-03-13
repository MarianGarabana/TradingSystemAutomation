"""
go_live.py — Ticker selector, latest predictions, and trading signals.

Data loading strategy (API-first with CSV fallback):
  1. PRIMARY — fetch the last 200 days of price data from SimFin via PySimFin,
     apply the same ETL feature engineering (engineer_features from etl.py),
     and serve the result as "live" data.  Cache result for 1 hour to avoid
     hammering the SimFin free-tier rate limit on every Streamlit rerun.
  2. FALLBACK — if the API call fails for any reason (network error, missing
     API key, rate-limit burst), silently fall back to the processed static
     CSV on disk and show a small ⚠️ badge so the user knows data is not live.

  200 days is chosen as the lookback window because the slowest indicator is
  the 26-day EMA used by MACD; adding buffer for weekends, holidays, and the
  rolling-window NaN warmup rows that are dropped by engineer_features().
"""

import datetime
import os
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Add the project root to sys.path so we can import from model/strategy.py
# and etl/etl.py.  os.path.dirname(__file__) is app/pages/, so going up two
# levels reaches the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model.strategy import Signal, prediction_to_signal, is_fallback_ticker, get_feature_cols
from etl.etl import engineer_features  # reuse the same feature engineering as training

st.set_page_config(page_title="Go Live", page_icon="🚀", layout="wide")

# ── Constants ──────────────────────────────────────────────────────────────────

# Paths relative to this file's location (app/pages/).
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
TRAINED_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "model", "trained")

# Feature columns are defined in model/strategy.py (STANDARD_FEATURE_COLS /
# FALLBACK_FEATURE_COLS) and selected at runtime via get_feature_cols(ticker).
# This ensures the inference schema always matches what was used at training time.

# Visual style for each trading signal so the UI is consistent everywhere.
SIGNAL_STYLE = {
    Signal.BUY:  {"emoji": "🟢", "color": "#00c805", "label": "BUY"},
    Signal.SELL: {"emoji": "🔴", "color": "#ff4b4b", "label": "SELL"},
    Signal.HOLD: {"emoji": "🟡", "color": "#ffd600", "label": "HOLD"},
}


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data
def available_tickers() -> list[str]:
    """Return sorted list of tickers that have a processed CSV on disk.

    @st.cache_data means this runs once and is reused across reruns,
    avoiding redundant directory listings every time the page refreshes.
    """
    if not os.path.exists(PROCESSED_DIR):
        return []
    return sorted(
        f.replace(".csv", "")
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith(".csv")
    )


@st.cache_data
def _load_processed_csv(ticker: str) -> pd.DataFrame:
    """Load the static ETL-processed CSV for a ticker (used as fallback).

    Kept private (_) because callers should use load_ticker_data() which
    tries the live API first and only calls this on failure.
    """
    path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=3600)  # cache live data for 1 hour — balances freshness vs API rate limits
def load_ticker_data(ticker: str) -> tuple[pd.DataFrame, str]:
    """Fetch and engineer features for *ticker*, returning (df, source).

    Tries the SimFin API first so the app shows real-time data.
    Falls back to the static processed CSV if anything goes wrong
    (missing API key, network error, rate-limit burst, etc.).

    The 'source' return value is "live" or "csv" — used by the UI to
    display a small badge so users know whether data is fresh.

    The lookback window is 200 calendar days:
      - MACD needs 26-day EMA warmup (slowest indicator)
      - Add 20 days for MA20/Bollinger, 14 for RSI, lags, etc.
      - Roughly double the minimum to account for weekends/holidays
      - engineer_features() drops ~26 warmup rows; we still get ~130+
        usable rows, which is more than enough for the 90-day chart.
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    start = (datetime.date.today() - datetime.timedelta(days=200)).strftime("%Y-%m-%d")

    try:
        from api_wrapper.pysimfin import PySimFin  # imported here to avoid crashing if
                                                   # the module or dotenv key is missing
        client = PySimFin()
        df_prices = client.get_share_prices(ticker, start=start, end=today)

        # The API's get_share_prices() returns Date, OHLCV, Dividend, Shares Outstanding.
        # Before calling engineer_features() we need to match the column names and
        # dtype expectations of the ETL:
        #   - Dividend: API may return None for days with no payout; ETL expects 0.
        #   - engineer_features() computes Market_Cap = Adj.Close × Shares Outstanding,
        #     so both columns must be present (they are — see get_share_prices()).
        df_prices["Dividend"] = df_prices["Dividend"].fillna(0)

        # engineer_features() is imported from etl/etl.py — same function used during
        # training, so the feature set is guaranteed to be identical to what the model
        # was trained on (no train/serve skew).
        df = engineer_features(df_prices)
        return df, "live"

    except Exception:
        # Any failure — bad key, network down, SimFin outage — falls back to CSV.
        # We intentionally swallow the exception here because go_live.py should
        # always show something useful to the user, never a crash screen.
        df = _load_processed_csv(ticker)
        return df, "csv"


def load_model(ticker: str):
    """Load the correct pooled model for the given ticker.

    - Standard tickers → model_pooled.pkl          (16 features)
    - Fallback tickers → model_pooled_fallback.pkl  (11 features, no fundamentals)

    Returns None if the .pkl file doesn't exist yet.
    Run:  python model/train.py --all
    """
    filename = "model_pooled_fallback.pkl" if is_fallback_ticker(ticker) else "model_pooled.pkl"
    path = os.path.join(TRAINED_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None


def fmt_pct(value: float) -> str:
    """Format a decimal fraction as a signed percentage string, e.g. +1.23%."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value * 100:.2f}%"


def fmt_large(value: float) -> str:
    """Format a large dollar number with T/B/M suffix for readability."""
    if value >= 1e12:
        return f"${value / 1e12:.2f}T"
    if value >= 1e9:
        return f"${value / 1e9:.2f}B"
    if value >= 1e6:
        return f"${value / 1e6:.2f}M"
    return f"${value:,.0f}"


# ── Chart functions ────────────────────────────────────────────────────────────
# Each chart function receives a filtered DataFrame (last 90 days) and returns
# a matplotlib chart.  We use st.pyplot(fig) to embed them in the page.
# All backgrounds are set to match the app's dark theme (#0d0d0d / #1a1a1a).

def plot_price(df: pd.DataFrame, ticker: str) -> plt.Figure:
    """Price chart: Adj. Close + MA5 + MA20 + Bollinger Band shading."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0d0d0d")   # figure background
    ax.set_facecolor("#0d0d0d")           # axes background

    # Bollinger Band shading: fill between upper and lower bands.
    # Low alpha (0.07) keeps it subtle so the price line is still readable.
    ax.fill_between(df["Date"], df["BB_Lower"], df["BB_Upper"],
                    color="#00c805", alpha=0.07, label="Bollinger Bands")

    # Main price line in the project's primary green.
    ax.plot(df["Date"], df["Adj. Close"], color="#00c805", linewidth=1.8, label="Adj. Close")

    # Dashed overlays for moving averages — use contrasting colours so they
    # are easy to distinguish from the price line at a glance.
    ax.plot(df["Date"], df["MA5"],  color="#80eaff", linewidth=1.0, linestyle="--", label="MA5")
    ax.plot(df["Date"], df["MA20"], color="#ff9f43", linewidth=1.0, linestyle="--", label="MA20")

    ax.set_title(f"{ticker} — Price (last 90 days)", color="#f0f0f0", fontsize=13)
    ax.tick_params(colors="#f0f0f0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(facecolor="#1a1a1a", labelcolor="#f0f0f0", fontsize=9)
    fig.tight_layout()
    return fig


def plot_rsi(df: pd.DataFrame) -> plt.Figure:
    """RSI chart with overbought (>70) and oversold (<30) horizontal zones.

    The shaded areas and dashed lines are standard RSI visualisation conventions
    that help users interpret the signal without reading a number.
    """
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    ax.plot(df["Date"], df["RSI"], color="#00c805", linewidth=1.4)

    # Horizontal reference lines at the standard overbought / oversold levels.
    ax.axhline(70, color="#ff4b4b", linewidth=0.8, linestyle="--")
    ax.axhline(30, color="#80eaff", linewidth=0.8, linestyle="--")

    # Subtle shading to highlight the overbought and oversold regions.
    ax.fill_between(df["Date"], 70, 100, color="#ff4b4b", alpha=0.06)
    ax.fill_between(df["Date"], 0, 30,  color="#80eaff", alpha=0.06)

    ax.set_ylim(0, 100)   # RSI is always in [0, 100]
    ax.set_title("RSI (14)", color="#f0f0f0", fontsize=11)
    ax.tick_params(colors="#f0f0f0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    fig.tight_layout()
    return fig


def plot_macd(df: pd.DataFrame) -> plt.Figure:
    """MACD chart: MACD line, signal line, and colour-coded histogram.

    The histogram (MACD - Signal) is green when MACD is above the signal
    (bullish momentum) and red when below (bearish momentum).
    """
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    histogram = df["MACD"] - df["MACD_Signal"]
    # Colour each bar individually based on sign of the histogram value.
    bar_colors = ["#00c805" if v >= 0 else "#ff4b4b" for v in histogram]
    ax.bar(df["Date"], histogram, color=bar_colors, width=1.0, alpha=0.7, label="Histogram")

    ax.plot(df["Date"], df["MACD"],        color="#80eaff", linewidth=1.2, label="MACD")
    ax.plot(df["Date"], df["MACD_Signal"], color="#ff9f43", linewidth=1.2, label="Signal")
    ax.axhline(0, color="#555555", linewidth=0.6)  # zero line reference

    ax.set_title("MACD", color="#f0f0f0", fontsize=11)
    ax.tick_params(colors="#f0f0f0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(facecolor="#1a1a1a", labelcolor="#f0f0f0", fontsize=8)
    fig.tight_layout()
    return fig


# ── Page layout ────────────────────────────────────────────────────────────────

st.title("🚀 Go Live — Real-Time Predictions")

tickers = available_tickers()
if not tickers:
    st.error("No processed data found. Run `python etl/etl.py --ticker AAPL` first.")
    st.stop()

# Ticker selector — driven by which processed CSVs exist on disk.
ticker = st.selectbox("Select ticker", tickers)

# Fetch live data from the API (or fall back to static CSV).
# data_source is "live" or "csv" — shown as a badge next to the ticker.
df, data_source = load_ticker_data(ticker)
last = df.iloc[-1]   # the most recent complete trading day

# Show a small data-freshness badge so users know the data origin.
if data_source == "live":
    st.success(f"🟢 Live data via SimFin API — last updated: {last['Date'].strftime('%Y-%m-%d')}")
else:
    st.warning("⚠️ Showing cached data (API unavailable). Data may not reflect today's prices.")

# Load the correct pooled model for this ticker (standard or fallback schema).
model = load_model(ticker)

# ── Section 1: Signal ──────────────────────────────────────────────────────────
st.subheader("Latest Signal")

if model is not None:
    # Build a single-row DataFrame of the latest feature values.
    # Using a DataFrame (not a plain array) preserves column names, which
    # sklearn pipelines require to match the training schema exactly.
    X_latest = pd.DataFrame([last[get_feature_cols(ticker)]])
    pred_class = int(model.predict(X_latest)[0])

    # Extract prediction confidence from the classifier's predict_proba().
    # confidence < 0.52 → HOLD (low confidence); otherwise BUY or SELL.
    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_latest)[0]
            confidence = float(max(proba))   # probability of the predicted class
        except Exception:
            confidence = None

    signal = prediction_to_signal(pred_class, confidence)
    style  = SIGNAL_STYLE[signal]

    # Render signal label in large coloured text.
    sig_col, conf_col, *_ = st.columns([1, 2, 3])
    with sig_col:
        st.markdown(
            f"<h1 style='color:{style['color']};margin:0'>"
            f"{style['emoji']} {style['label']}</h1>",
            unsafe_allow_html=True,
        )
    # Show confidence progress bar only when we have a probability score.
    if confidence is not None:
        with conf_col:
            st.markdown(f"**Confidence:** {confidence * 100:.1f}%")
            st.progress(confidence)
else:
    # Model file not found — app still works, just without a prediction.
    st.warning(
        "No trained model found. "
        "Run `python model/train.py --all` to generate one. "
        "Technical indicators are shown below in the meantime."
    )

# ── Key metrics row ────────────────────────────────────────────────────────────
# Four st.metric widgets in a row give a quick at-a-glance snapshot.
# The 'delta' parameter adds the green/red directional arrow automatically.
m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest Price",  f"${last['Adj. Close']:.2f}")
m2.metric("Daily Return",  fmt_pct(last["Return"]), delta=fmt_pct(last["Return"]))
m3.metric(
    "RSI (14)", f"{last['RSI']:.1f}",
    delta="Overbought" if last["RSI"] > 70 else ("Oversold" if last["RSI"] < 30 else "Neutral"),
)
m4.metric("Market Cap", fmt_large(last["Market_Cap"]))

st.divider()

# ── Section 2: Price chart ─────────────────────────────────────────────────────
st.subheader("Price History")

# Slice to last 90 calendar days for a focused view — 90 days is enough
# to see short-term trends while keeping the chart readable.
cutoff = df["Date"].max() - pd.Timedelta(days=90)
df_90  = df[df["Date"] >= cutoff]

st.pyplot(plot_price(df_90, ticker))

st.divider()

# ── Section 3: Technical indicator charts ─────────────────────────────────────
st.subheader("Technical Indicators")

# Side-by-side layout: RSI on the left, MACD on the right.
left, right = st.columns(2)
with left:
    st.pyplot(plot_rsi(df_90))
with right:
    st.pyplot(plot_macd(df_90))

# Collapsible table with the exact numeric values of all indicators.
# Useful for deeper inspection without cluttering the main view.
with st.expander("Latest indicator values"):
    indicator_cols = [
        "RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower",
        "Volume_Change", "Return_Lag1", "Return_Lag2",
    ]
    st.dataframe(
        last[indicator_cols].to_frame("Value").round(4),
        use_container_width=True,
    )
