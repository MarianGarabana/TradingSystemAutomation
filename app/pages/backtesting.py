"""
backtesting.py — Historical performance evaluation of the trading strategy.

CHANGES FROM STUB:
  - Full implementation replacing the single st.info("Backtesting not yet implemented") placeholder.
  - Lets the user pick any processed ticker and a date range to evaluate.
  - If a trained .pkl model exists: runs predictions on the selected period,
    computes accuracy and win rate against the actual Target column.
  - If no model: falls back to a "Baseline — Always Buy" strategy so the
    page is functional for the demo even before ML models are trained.
  - Simulates a $10,000 portfolio: invested on BUY signals, cash on SELL/HOLD.
  - Compares strategy cumulative return vs. buy-and-hold on a chart.
  - Shows rolling 30-day accuracy chart (reveals if the model degrades over time).
  - Shows per-row signal history table for the last 30 days.
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path so we can import model/strategy.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model.strategy import prediction_to_signal

st.set_page_config(page_title="Backtesting", page_icon="🔍", layout="wide")

# ── Constants ──────────────────────────────────────────────────────────────────

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
TRAINED_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "model", "trained")

# Same feature columns as go_live.py — must match what was used at training time.
FEATURE_COLS = [
    "Return", "MA5", "MA20", "Volume_Change", "RSI",
    "MACD", "MACD_Signal", "BB_Upper", "BB_Lower",
    "Return_Lag1", "Return_Lag2",
]

INITIAL_CAPITAL = 10_000.0   # starting portfolio value in USD for simulation


# ── Helpers ────────────────────────────────────────────────────────────────────

@st.cache_data
def available_tickers() -> list[str]:
    """Return sorted list of tickers that have a processed CSV on disk."""
    if not os.path.exists(PROCESSED_DIR):
        return []
    return sorted(
        f.replace(".csv", "")
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith(".csv")
    )


@st.cache_data
def load_processed(ticker: str) -> pd.DataFrame:
    """Load processed CSV and sort by date (cached to avoid repeated disk reads)."""
    path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def load_model(ticker: str):
    """Load trained model if available, else return None."""
    path = os.path.join(TRAINED_DIR, f"model_{ticker}.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


def run_backtest(df: pd.DataFrame, model) -> pd.DataFrame:
    """Run the strategy simulation on the filtered DataFrame.

    Parameters
    ----------
    df    : processed data slice (already filtered to the chosen date range)
    model : loaded sklearn pipeline, or None for baseline mode

    Returns
    -------
    result_df : df augmented with columns:
        Actual_Signal   — ground truth: 1 if next-day return > 0 else 0
        Predicted       — model's prediction (or 1 for baseline)
        Correct         — whether the prediction matched the actual signal
        Strategy_Value  — simulated $10k portfolio under the ML strategy
        BuyHold_Value   — simulated $10k portfolio under buy-and-hold
    """
    df = df.copy()

    # Ground truth: binarise the continuous next-day return.
    # Target > 0 means the price actually rose → labelled 1 (would have been a good BUY).
    # Target <= 0 means the price fell or was flat → labelled 0.
    df["Actual_Signal"] = (df["Target"] > 0).astype(int)

    if model is not None:
        # Run the full feature matrix through the trained pipeline.
        # We use only the rows that have all feature columns present.
        X = df[FEATURE_COLS]
        raw_preds = model.predict(X)

        # Handle both regression models (float output) and classifiers (int output).
        # For regression: positive predicted return → BUY (1), negative → SELL (0).
        if raw_preds.dtype.kind == "f":
            df["Predicted"] = (raw_preds > 0).astype(int)
        else:
            # Map class labels to {0, 1}: treat anything > 0 as BUY.
            df["Predicted"] = (raw_preds.astype(int) > 0).astype(int)
        strategy_label = "ML Strategy"
    else:
        # Baseline: always predict BUY (1).
        # This represents the naive "always be invested" strategy and is a
        # meaningful benchmark — any real model should beat it.
        df["Predicted"] = 1
        strategy_label = "Baseline (Always Buy)"

    # Correct prediction: 1 if model and reality agreed, 0 otherwise.
    df["Correct"] = (df["Predicted"] == df["Actual_Signal"]).astype(int)

    # ── Portfolio simulation ────────────────────────────────────────────────────
    # We simulate holding $10k.
    # On days where the model predicts BUY (1): the portfolio is invested and
    # its value follows the actual next-day return.
    # On days where the model predicts SELL/HOLD (0): the portfolio stays in
    # cash and does not change.
    #
    # daily_factor = 1 + actual_return when invested, else 1.0 (no change).
    # We use cumprod() to compound these daily multipliers into a running value.
    actual_return = df["Target"]   # continuous next-day return column from ETL

    strategy_factor = np.where(df["Predicted"] == 1, 1 + actual_return, 1.0)
    buyhold_factor  = 1 + actual_return   # always invested

    df["Strategy_Value"] = INITIAL_CAPITAL * pd.Series(strategy_factor).cumprod().values
    df["BuyHold_Value"]  = INITIAL_CAPITAL * pd.Series(buyhold_factor).cumprod().values

    return df, strategy_label


# ── Chart helpers ──────────────────────────────────────────────────────────────

def plot_cumulative_return(df: pd.DataFrame, strategy_label: str) -> plt.Figure:
    """Line chart comparing strategy portfolio value vs. buy-and-hold over time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    ax.plot(df["Date"], df["Strategy_Value"], color="#00c805", linewidth=1.8,
            label=strategy_label)
    ax.plot(df["Date"], df["BuyHold_Value"],  color="#ff9f43", linewidth=1.4,
            linestyle="--", label="Buy & Hold")
    ax.axhline(INITIAL_CAPITAL, color="#555555", linewidth=0.7, linestyle=":")

    ax.set_title("Portfolio Value Over Time ($10,000 start)", color="#f0f0f0", fontsize=13)
    ax.set_ylabel("Portfolio Value ($)", color="#f0f0f0")
    ax.tick_params(colors="#f0f0f0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(facecolor="#1a1a1a", labelcolor="#f0f0f0")
    fig.tight_layout()
    return fig


def plot_rolling_accuracy(df: pd.DataFrame, window: int = 30) -> plt.Figure:
    """Rolling accuracy chart: shows whether prediction quality is stable over time.

    A flat line suggests consistent performance; a declining trend may indicate
    the model is drifting as market regimes change.
    """
    rolling_acc = df["Correct"].rolling(window).mean() * 100

    fig, ax = plt.subplots(figsize=(10, 3))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    ax.plot(df["Date"], rolling_acc, color="#80eaff", linewidth=1.4)

    # 50% reference line: below this means the model is worse than random guessing.
    ax.axhline(50, color="#555555", linewidth=0.8, linestyle="--")

    ax.set_title(f"Rolling {window}-Day Accuracy (%)", color="#f0f0f0", fontsize=12)
    ax.set_ylabel("Accuracy (%)", color="#f0f0f0")
    ax.set_ylim(0, 100)
    ax.tick_params(colors="#f0f0f0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    fig.tight_layout()
    return fig


# ── Page layout ────────────────────────────────────────────────────────────────

st.title("🔍 Backtesting — Historical Performance")

tickers = available_tickers()
if not tickers:
    st.error("No processed data found. Run `python etl/etl.py --ticker AAPL` first.")
    st.stop()

# ── Controls ───────────────────────────────────────────────────────────────────
col_tick, col_start, col_end = st.columns([1, 1, 1])

with col_tick:
    ticker = st.selectbox("Select ticker", tickers)

# Load full history to derive min/max date for the date pickers.
df_full = load_processed(ticker)
min_date = df_full["Date"].min().date()
max_date = df_full["Date"].max().date()

with col_start:
    start_date = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)
with col_end:
    end_date   = st.date_input("End date",   value=max_date, min_value=min_date, max_value=max_date)

run_btn = st.button("▶ Run Backtest", type="primary")

if not run_btn:
    st.info("Select a ticker and date range, then press **Run Backtest**.")
    st.stop()

# ── Run ────────────────────────────────────────────────────────────────────────

# Filter data to selected date range.
mask = (df_full["Date"].dt.date >= start_date) & (df_full["Date"].dt.date <= end_date)
df_slice = df_full[mask].copy()

if df_slice.empty:
    st.error("No data in the selected date range.")
    st.stop()

# Drop rows where the target is NaN (last row after shift in ETL has no next-day return).
df_slice = df_slice.dropna(subset=["Target"] + FEATURE_COLS).reset_index(drop=True)

if len(df_slice) < 10:
    st.error("Not enough rows to run a backtest. Widen the date range.")
    st.stop()

model = load_model(ticker)

if model is None:
    st.warning(
        f"No trained model found for **{ticker}** — showing **Baseline (Always Buy)** strategy. "
        f"Run `python model/train.py --ticker {ticker}` to compare against the ML model."
    )

# Run the simulation.
result_df, strategy_label = run_backtest(df_slice, model)

# ── Summary metrics ────────────────────────────────────────────────────────────
st.subheader("Summary")

final_strategy = result_df["Strategy_Value"].iloc[-1]
final_buyhold  = result_df["BuyHold_Value"].iloc[-1]
total_return   = (final_strategy - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
bh_return      = (final_buyhold  - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
accuracy       = result_df["Correct"].mean() * 100

# Win rate: of the trades where we predicted BUY and the price actually rose.
# This measures precision on the BUY signal, not overall accuracy.
buy_mask = result_df["Predicted"] == 1
win_rate = result_df.loc[buy_mask, "Correct"].mean() * 100 if buy_mask.any() else 0.0
total_trades = int(buy_mask.sum())  # number of days we "bought in"

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(
    f"{strategy_label} Return",
    f"{total_return:+.1f}%",
    delta=f"{total_return - bh_return:+.1f}% vs B&H",
)
m2.metric("Buy & Hold Return", f"{bh_return:+.1f}%")
m3.metric("Prediction Accuracy", f"{accuracy:.1f}%")
m4.metric("BUY Win Rate", f"{win_rate:.1f}%")
m5.metric("Total Trade Days", total_trades)

st.divider()

# ── Cumulative return chart ────────────────────────────────────────────────────
st.subheader("Portfolio Value Over Time")
st.pyplot(plot_cumulative_return(result_df, strategy_label))

st.divider()

# ── Rolling accuracy chart ─────────────────────────────────────────────────────
st.subheader("Rolling 30-Day Accuracy")
st.caption(
    "A value above 50% means the model is beating random guessing. "
    "A declining trend may indicate the model is drifting with changing market conditions."
)
st.pyplot(plot_rolling_accuracy(result_df))

st.divider()

# ── Signal history table ───────────────────────────────────────────────────────
st.subheader("Signal History (last 30 rows)")

# Build a clean display table from the result DataFrame.
history = result_df[["Date", "Adj. Close", "Actual_Signal", "Predicted", "Correct"]].copy()
history["Date"] = history["Date"].dt.strftime("%Y-%m-%d")
history["Adj. Close"] = history["Adj. Close"].round(2)

# Map numeric labels to readable strings so the table is self-explanatory.
history["Actual Movement"] = history["Actual_Signal"].map({1: "⬆ Rise", 0: "⬇ Fall"})
history["Prediction"]      = history["Predicted"].map({1: "BUY", 0: "SELL"})
history["Correct?"]        = history["Correct"].map({1: "✅ Yes", 0: "❌ No"})

display_cols = ["Date", "Adj. Close", "Actual Movement", "Prediction", "Correct?"]
st.dataframe(
    history[display_cols].tail(30).reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
)
