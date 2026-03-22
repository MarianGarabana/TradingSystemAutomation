"""
prediction_bet.py — Interactive prediction betting simulator.

Users select a ticker, choose UP or DOWN, set their investment amount,
pick a leverage multiplier (1x / 2x / 5x / 10x), and a time horizon
(1 day / 1 month / 1 year).

The page simulates the outcome using the most recent available historical
data from the processed CSVs and the trained ML model's signal. It shows:
  - Whether the bet won or lost
  - The model's signal and confidence gauge
  - A price chart with entry/exit markers and MA overlays
  - The top 3 signal-driving features with educational interpretations
  - A scenario comparison table across all leverage options
  - Historical model win rate for that ticker (out-of-sample period)
"""

import datetime
import os
import sys

import joblib
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Optional Streamlit extensions — degrade gracefully if not installed.
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False


# Add project root to sys.path so we can import from model/ and etl/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from model.strategy import (
    Signal,
    prediction_to_signal,
    is_fallback_ticker,
    get_feature_cols,
)
from etl.etl import engineer_features  # same function used at training time — no train/serve skew

st.set_page_config(page_title="Prediction Bet", page_icon="🎯", layout="wide")

# ── Constants ──────────────────────────────────────────────────────────────────

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
TRAINED_DIR   = os.path.join(os.path.dirname(__file__), "..", "..", "model", "trained")

# The 5 fundamental ratio columns that only standard (non-fallback) tickers have.
# The SimFin price API doesn't return these — we carry the last known values
# forward from the processed CSV, exactly as go_live.py does.
_FUND_COLS = [
    "Gross_Margin", "Operating_Margin", "Net_Margin",
    "Debt_to_Equity", "Operating_CF_Ratio",
]

# Leverage options shown to the user.
LEVERAGE_OPTIONS = [1, 2, 5, 10]

# Time horizon → number of trading days.
HORIZON_OPTIONS = {
    "📅 1 Day  (Short Term)":   1,
    "🗓️ 1 Month  (Medium Term)": 21,
    "📆 1 Year  (Long Term)":   252,
}

# Company metadata (mirrors Home.py so we don't need the raw CSV on disk).
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
    "PEP":  ("PepsiCo Inc.",           "Consumer"),
    "PFE":  ("Pfizer Inc.",            "Healthcare"),
    "PLTR": ("Palantir Technologies",  "Technology"),
    "QCOM": ("Qualcomm Inc.",          "Technology"),
    "TSLA": ("Tesla Inc.",             "Technology"),
    "UNH":  ("UnitedHealth Group",     "Healthcare"),
    "V":    ("Visa Inc.",              "Financials"),
    "WMT":  ("Walmart Inc.",           "Consumer"),
}

# Signal colours reused from go_live.py so the UI is consistent.
SIGNAL_STYLE = {
    Signal.BUY:  {"emoji": "🟢", "color": "#00c805", "label": "BUY"},
    Signal.SELL: {"emoji": "🔴", "color": "#ff4b4b", "label": "SELL"},
    Signal.HOLD: {"emoji": "🟡", "color": "#ffd600", "label": "HOLD"},
}

# Human-readable explanations for the top signal-driving features.
# Used in the educational "Why the model said…" section.
FEATURE_INTERPRETATIONS = {
    "RSI": {
        "high":    "RSI above 70 signals overbought conditions — buying momentum may be exhausted.",
        "low":     "RSI below 30 signals oversold conditions — a price bounce may be near.",
        "neutral": "RSI is in neutral territory (30–70), showing balanced buying and selling pressure.",
    },
    "MACD": {
        "high":    "MACD is positive — short-term momentum is bullish (fast EMA above slow EMA).",
        "low":     "MACD is negative — short-term momentum is bearish (fast EMA below slow EMA).",
        "neutral": "MACD is near zero — no strong directional momentum at the moment.",
    },
    "Return_norm": {
        "high":    "Volatility-normalised return is well above average — unusually strong recent gains.",
        "low":     "Volatility-normalised return is well below average — recent selling pressure.",
        "neutral": "Returns are tracking close to their historical average for this stock.",
    },
    "Return_norm_Lag1": {
        "high":    "Yesterday's normalised return was strongly positive — recent upward momentum.",
        "low":     "Yesterday's normalised return was strongly negative — recent downward momentum.",
        "neutral": "Yesterday's return was close to average — no strong prior-day signal.",
    },
    "Return_norm_Lag2": {
        "high":    "Returns two days ago were unusually high — multi-day upward momentum building.",
        "low":     "Returns two days ago were unusually low — multi-day downward momentum.",
        "neutral": "Returns two days ago were near average — no lagged momentum signal.",
    },
    "Volatility_20": {
        "high":    "20-day volatility is elevated — the stock is moving more than usual, increasing risk.",
        "low":     "20-day volatility is low — the stock is relatively calm and predictable.",
        "neutral": "Volatility is near its historical average for this ticker.",
    },
    "Log_Return": {
        "high":    "Log return today is unusually high — strong single-day price gain.",
        "low":     "Log return today is unusually low — strong single-day price drop.",
        "neutral": "Log return is close to normal levels for this stock.",
    },
    "MA5": {
        "high":    "Price is tracking well above the 5-day moving average — short-term uptrend.",
        "low":     "Price is below the 5-day moving average — short-term downward pressure.",
        "neutral": "Price is near its 5-day moving average — no clear short-term trend.",
    },
    "MA20": {
        "high":    "Price is above the 20-day moving average — medium-term bullish momentum.",
        "low":     "Price is below the 20-day moving average — medium-term bearish pressure.",
        "neutral": "Price is near its 20-day moving average — no clear medium-term trend.",
    },
    "Volume_Change": {
        "high":    "Trading volume is surging above normal — unusual market activity, possible breakout.",
        "low":     "Volume has dropped significantly — lower interest or conviction in current move.",
        "neutral": "Volume is within its normal range — no unusual activity.",
    },
    "Market_Cap": {
        "high":    "Market cap is above its historical norm — the stock has grown recently.",
        "low":     "Market cap is below its historical norm — relative undervaluation vs. history.",
        "neutral": "Market cap is within its historical range.",
    },
}


# ── Data & Model Helpers ───────────────────────────────────────────────────────

@st.cache_data
def available_tickers() -> list[str]:
    """Return sorted list of tickers that have a processed CSV on disk."""
    if not os.path.exists(PROCESSED_DIR):
        return []
    return sorted(
        f.replace(".csv", "")
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith(".csv") and not f.startswith(".")
    )


@st.cache_data
def _load_processed_csv(ticker: str) -> pd.DataFrame | None:
    """Load the static processed CSV from disk. Used as fallback when the API
    is unavailable, and also for the historical win rate calculation which needs
    the full multi-year history (not just a recent API window).
    """
    path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=3600)  # cache for 1 hour so we don't hit the API on every click
def load_ticker_data(ticker: str, n_days: int) -> tuple[pd.DataFrame, str]:
    """Fetch fresh price data from the SimFin API and engineer features.

    Returns a tuple of (dataframe, source) where source is:
        "live" — data came from the SimFin API (fresh, updates daily)
        "csv"  — API was unavailable, fell back to the local processed CSV

    Why API-first?
        The processed CSVs are static — they only update when someone
        manually re-runs the ETL. The API always returns today's data,
        so the simulation stays current without any manual work.

    Lookback window:
        We fetch enough calendar days so that after engineer_features()
        drops its ~26-row MACD warmup, we still have all n_days we need.
        Short/medium horizons (≤1 month): 200 days — same as go_live.py.
        Long horizon (1 year): 500 days to cover 252 trading days + warmup.
    """
    # How many calendar days to fetch from the API.
    lookback_days = 500 if n_days > 21 else 200

    today = datetime.date.today().strftime("%Y-%m-%d")
    start = (datetime.date.today() - datetime.timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    try:
        # Import here so the page still loads even if the API key is missing.
        # Importing pysimfin also runs load_dotenv() inside that module,
        # which loads SIMFIN_API_KEY from the .env file into os.environ.
        from api_wrapper.pysimfin import PySimFin

        # Read the API key — os.getenv() works because load_dotenv() just ran above.
        # We also check st.secrets as a fallback for Streamlit Cloud deployments.
        # IMPORTANT: st.secrets throws if no secrets.toml exists (local dev),
        # so it must be wrapped in its own try/except.
        api_key = os.getenv("SIMFIN_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets.get("SIMFIN_API_KEY")
            except Exception:
                pass  # no secrets.toml — that's fine, we already tried os.getenv()

        client = PySimFin(api_key=api_key)

        # Step 1 — fetch raw daily prices from SimFin.
        df_prices = client.get_share_prices(ticker, start=start, end=today)
        df_prices["Dividend"] = df_prices["Dividend"].fillna(0)

        # Step 2 — run the same feature engineering as the ETL pipeline.
        # This guarantees the features exactly match what the model was trained on.
        df = engineer_features(df_prices)

        # Step 3 — standard tickers need 5 fundamental ratio columns that the
        # price API doesn't provide. We attach the last known values from the
        # processed CSV (fundamentals change quarterly so this is still valid).
        if not is_fallback_ticker(ticker):
            try:
                csv_df   = _load_processed_csv(ticker)
                last_fund = csv_df[_FUND_COLS].dropna().iloc[-1]
                for col in _FUND_COLS:
                    df[col] = last_fund[col]
            except Exception:
                # If the CSV is missing, fill with NaN — the model handles this.
                for col in _FUND_COLS:
                    df[col] = float("nan")

        return df, "live"

    except Exception:
        # Any failure (no API key, network error, rate limit) → fall back to CSV.
        # The page always shows something useful rather than crashing.
        df = _load_processed_csv(ticker)
        if df is None:
            df = pd.DataFrame()  # empty df — caller will show an error message
        return df, "csv"


@st.cache_resource
def load_model(ticker: str):
    """Load the pooled model (standard or fallback) for the given ticker."""
    fname = "model_pooled_fallback.pkl" if is_fallback_ticker(ticker) else "model_pooled.pkl"
    path = os.path.join(TRAINED_DIR, fname)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


@st.cache_data(ttl=3600)
def compute_historical_win_rate(ticker: str) -> tuple[float | None, int | None]:
    """Return (accuracy, n_rows) on the OOS portion (last 20%) of the ticker's history.

    Uses the full processed CSV (not the API window) because we need several
    years of data to compute a meaningful out-of-sample accuracy score.
    """
    df = _load_processed_csv(ticker)  # needs full history, not just recent API data
    if df is None or len(df) < 50:
        return None, None
    model = load_model(ticker)
    if model is None:
        return None, None

    feature_cols = get_feature_cols(ticker)
    available_cols = [c for c in feature_cols if c in df.columns]
    if not available_cols:
        return None, None

    split_idx = int(len(df) * 0.8)
    df_oos = df.iloc[split_idx:].dropna(subset=available_cols + ["Target"])
    df_oos = df_oos[df_oos["Target"] != 0]
    if len(df_oos) < 10:
        return None, None

    X = df_oos[available_cols].values
    y_true = (df_oos["Target"] > 0).astype(int).values
    try:
        y_pred = model.predict(X)
        return float((y_true == y_pred).mean()), len(df_oos)
    except Exception:
        return None, None


def load_lottie_url(url: str) -> dict | None:
    """Fetch a Lottie animation JSON from a URL. Returns None on failure."""
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ── Chart Helpers ──────────────────────────────────────────────────────────────

def render_confidence_gauge(confidence: float, signal: Signal) -> go.Figure:
    """Plotly indicator gauge showing the model's confidence (0–100%)."""
    color = SIGNAL_STYLE[signal]["color"]
    label = SIGNAL_STYLE[signal]["label"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        title={
            "text": (
                f"Prediction Score<br>"
                f"<span style='font-size:0.85em;color:{color}'>"
                f"{SIGNAL_STYLE[signal]['emoji']} {label}</span>"
            ),
            "font": {"size": 14},
        },
        number={"suffix": "%", "font": {"size": 26, "color": "white"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "rgba(255,255,255,0.4)",
                "tickfont": {"color": "white"},
            },
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1a1a2e",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  52], "color": "#252538"},
                {"range": [52, 70], "color": "#2e2e4a"},
                {"range": [70, 100], "color": "#38385e"},
            ],
            "threshold": {
                "line": {"color": "#ffd600", "width": 3},
                "thickness": 0.75,
                "value": 51,  # the minimum confidence to act (matches strategy.py)
            },
        },
    ))
    fig.update_layout(
        height=230,
        margin=dict(t=60, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


def render_price_chart(
    df_period: pd.DataFrame,
    ticker: str,
    horizon_label: str,
    entry_date,
    exit_date,
    price_col: str,
) -> go.Figure:
    """Line chart of Adj. Close with MA overlays and entry/exit markers."""
    fig = go.Figure()

    # ── Main price line ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df_period["Date"],
        y=df_period[price_col],
        mode="lines",
        name="Adj. Close",
        line=dict(color="#4da6ff", width=2),
    ))

    # ── MA5 overlay ────────────────────────────────────────────────────────────
    if "MA5" in df_period.columns:
        fig.add_trace(go.Scatter(
            x=df_period["Date"],
            y=df_period["MA5"],
            mode="lines",
            name="MA5",
            line=dict(color="#00e5ff", width=1, dash="dash"),
            opacity=0.7,
        ))

    # ── MA20 overlay ───────────────────────────────────────────────────────────
    if "MA20" in df_period.columns:
        fig.add_trace(go.Scatter(
            x=df_period["Date"],
            y=df_period["MA20"],
            mode="lines",
            name="MA20",
            line=dict(color="#ff9800", width=1, dash="dash"),
            opacity=0.7,
        ))

    # ── Entry marker ───────────────────────────────────────────────────────────
    entry_rows = df_period[df_period["Date"] == entry_date]
    if not entry_rows.empty:
        entry_price_val = entry_rows[price_col].iloc[0]
        fig.add_trace(go.Scatter(
            x=[entry_date],
            y=[entry_price_val],
            mode="markers+text",
            name="Entry",
            marker=dict(color="#ffd600", size=14, symbol="triangle-up"),
            text=["Entry"],
            textposition="top center",
            textfont=dict(color="#ffd600", size=11),
            showlegend=True,
        ))

    # ── Exit marker ────────────────────────────────────────────────────────────
    exit_rows = df_period[df_period["Date"] == exit_date]
    if not exit_rows.empty:
        exit_price_val = exit_rows[price_col].iloc[0]
        # Green exit if price rose, red if fell
        if not entry_rows.empty:
            exit_color = "#00c805" if exit_price_val >= entry_price_val else "#ff4b4b"
        else:
            exit_color = "#ffffff"
        fig.add_trace(go.Scatter(
            x=[exit_date],
            y=[exit_price_val],
            mode="markers+text",
            name="Exit",
            marker=dict(color=exit_color, size=14, symbol="triangle-down"),
            text=["Exit"],
            textposition="bottom center",
            textfont=dict(color=exit_color, size=11),
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(
            text=f"{ticker} — {horizon_label} Simulation Window",
            font=dict(color="white", size=15),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            showgrid=True,
            color="white",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            showgrid=True,
            title="Price (USD)",
            color="white",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            font=dict(color="white"),
        ),
        height=380,
        margin=dict(t=50, b=20, l=10, r=10),
        hovermode="x unified",
    )
    return fig


def render_feature_chart(top_features: list[tuple[str, float]]) -> go.Figure:
    """Horizontal bar chart showing z-scores of the top signal-driving features."""
    names   = [f[0] for f in top_features]
    zscores = [f[1] for f in top_features]
    colors  = ["#00c805" if z > 0 else "#ff4b4b" for z in zscores]

    fig = go.Figure(go.Bar(
        y=names,
        x=zscores,
        orientation="h",
        marker_color=colors,
        text=[f"{z:+.2f}σ" for z in zscores],
        textposition="outside",
        textfont=dict(color="white", size=12),
    ))

    # Zero baseline
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.3)", line_width=1)

    fig.update_layout(
        title=dict(
            text="Top Signal Drivers vs. Historical Mean",
            font=dict(color="white", size=13),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            title="Std Deviations from Mean (σ)",
            color="white",
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            color="white",
        ),
        height=220,
        margin=dict(t=40, b=20, l=10, r=70),
    )
    return fig


def render_return_distribution(
    df_full: pd.DataFrame,
    actual_return: float,
    n_days: int,
    horizon_clean: str,
    ticker: str,
) -> go.Figure:
    """Histogram of historical returns for the chosen horizon with the simulation's
    return marked as a vertical line. Gives the user instant context: 'was this
    a typical outcome or an extreme one?'

    For 1-day bets   → uses daily returns from the 'Return' column.
    For 1-month bets → computes rolling 21-day cumulative returns.
    For 1-year bets  → computes rolling 252-day cumulative returns.
    """
    price_col = "Adj. Close" if "Adj. Close" in df_full.columns else "Close"

    # Build the correct return series depending on the time horizon.
    if n_days == 1:
        # Daily percentage returns — already computed in ETL.
        returns = df_full["Return"].dropna()
        x_label = "Daily Return (%)"
    else:
        # Rolling cumulative return over n_days trading days.
        prices  = df_full[price_col].dropna()
        returns = prices.pct_change(periods=n_days).dropna()
        x_label = f"{horizon_clean} Cumulative Return (%)"

    if len(returns) < 20:
        # Not enough data — return an empty placeholder chart.
        fig = go.Figure()
        fig.update_layout(
            title="Not enough data for return distribution.",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=280,
        )
        return fig

    returns_pct = returns * 100  # convert to % for display

    # Work out which percentile the simulation return falls in.
    percentile = float((returns_pct < actual_return * 100).mean() * 100)

    # Color of the marker line: green if positive return, red if negative.
    marker_color = "#00c805" if actual_return >= 0 else "#ff4b4b"
    marker_label = f"Your result: {actual_return * 100:+.2f}% (top {100 - percentile:.0f}%)" \
                   if actual_return >= 0 else \
                   f"Your result: {actual_return * 100:+.2f}% (bottom {percentile:.0f}%)"

    fig = go.Figure()

    # Main histogram — split into negative (red) and positive (green) bars.
    fig.add_trace(go.Histogram(
        x=returns_pct[returns_pct < 0],
        nbinsx=40,
        name="Negative returns",
        marker_color="rgba(255,75,75,0.55)",
        marker_line=dict(color="rgba(255,75,75,0.8)", width=0.5),
    ))
    fig.add_trace(go.Histogram(
        x=returns_pct[returns_pct >= 0],
        nbinsx=40,
        name="Positive returns",
        marker_color="rgba(0,200,5,0.55)",
        marker_line=dict(color="rgba(0,200,5,0.8)", width=0.5),
    ))

    # Vertical line marking the simulation's actual return.
    fig.add_vline(
        x=actual_return * 100,
        line_color=marker_color,
        line_width=2.5,
        line_dash="dash",
        annotation_text=marker_label,
        annotation_position="top right" if actual_return >= 0 else "top left",
        annotation_font=dict(color=marker_color, size=12),
    )

    # Vertical line at zero for reference.
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.25)", line_width=1)

    fig.update_layout(
        title=dict(
            text=f"{ticker} — Historical {horizon_clean} Return Distribution",
            font=dict(color="white", size=14),
        ),
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        xaxis=dict(
            title=x_label,
            gridcolor="rgba(255,255,255,0.08)",
            color="white",
            ticksuffix="%",
        ),
        yaxis=dict(
            title="Number of periods",
            gridcolor="rgba(255,255,255,0.08)",
            color="white",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.3)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
            font=dict(color="white"),
        ),
        height=300,
        margin=dict(t=50, b=40, l=10, r=10),
        hovermode="x unified",
    )
    return fig


# ── UI Helpers ─────────────────────────────────────────────────────────────────

def _metric_card(
    label: str,
    value: str,
    value_color: str = "#ffffff",
    delta_text: str | None = None,
    delta_positive: bool | None = None,
    sub_text: str | None = None,
) -> str:
    """Return an HTML metric card that matches the app's dark theme."""
    if delta_text is not None and delta_positive is not None:
        dc    = "#00c805" if delta_positive else "#ff4b4b"
        arrow = "▲" if delta_positive else "▼"
        delta_html = (
            f"<p style='margin:5px 0 0;font-size:13px;color:{dc};'>"
            f"{arrow} {delta_text}</p>"
        )
    else:
        delta_html = "<p style='margin:5px 0 0;font-size:13px;'>&nbsp;</p>"
    sub_html = (
        f"<p style='margin:4px 0 0;font-size:11px;color:#666;'>{sub_text}</p>"
        if sub_text else
        "<p style='margin:4px 0 0;font-size:11px;'>&nbsp;</p>"
    )
    return (
        f"<div style='background:#1a1a2e;border:1px solid rgba(255,255,255,0.12);"
        f"border-radius:10px;padding:14px 16px;'>"
        f"<p style='margin:0 0 6px;font-size:12px;color:#888;"
        f"text-transform:uppercase;letter-spacing:0.6px;'>{label}</p>"
        f"<p style='margin:0;font-size:22px;font-weight:700;color:{value_color};'>{value}</p>"
        f"{delta_html}{sub_html}</div>"
    )


# ── Analysis Helpers ───────────────────────────────────────────────────────────

def get_top_feature_drivers(
    entry_row: pd.Series,
    df: pd.DataFrame,
    feature_cols: list[str],
    n: int = 3,
) -> list[tuple[str, float]]:
    """Return top-n features ranked by |z-score| at the entry date.

    A high absolute z-score means that feature was unusually far from its
    historical average on the day the model made its prediction, making it
    a likely driver of the signal.
    """
    available = [c for c in feature_cols if c in df.columns and c in entry_row.index]
    z_scores: dict[str, float] = {}
    for col in available:
        std = df[col].std()
        if std > 0:
            z_scores[col] = float((entry_row[col] - df[col].mean()) / std)
    return sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:n]


def interpret_feature(feature_name: str, z_score: float) -> str:
    """Return a human-readable explanation for a feature value."""
    interp = FEATURE_INTERPRETATIONS.get(feature_name)
    if interp is None:
        direction = "above" if z_score > 0 else "below"
        return (
            f"{feature_name} is {direction} its historical average "
            f"by {abs(z_score):.1f} standard deviations."
        )
    if z_score > 1.0:
        return interp["high"]
    if z_score < -1.0:
        return interp["low"]
    return interp["neutral"]


# ── Page ───────────────────────────────────────────────────────────────────────

st.title("🎯 Prediction Bet")
st.markdown(
    "Pick a stock, choose UP or DOWN, set your stake and leverage, then let the AI signal "
    "guide you — and see what actually happened with the most recent market data."
)
st.info(
    "ℹ️ **SIMULATION MODE** — This is a paper trading simulator that uses our historical "
    "processed CSV data to show what *would have happened* over the most recent available "
    "period. All results are hypothetical. **This is not financial advice.**"
)

st.divider()

# ── Controls ───────────────────────────────────────────────────────────────────

tickers = available_tickers()
if not tickers:
    st.error("No processed ticker data found. Run `python etl/etl.py --all` first.")
    st.stop()

ctrl_col1, ctrl_col2 = st.columns([1, 1])

with ctrl_col1:
    default_idx = tickers.index("AAPL") if "AAPL" in tickers else 0
    ticker = st.selectbox(
        "📌 Stock Ticker",
        tickers,
        index=default_idx,
        help="Select the stock you want to simulate trading.",
    )
    company_name, sector = COMPANIES.get(ticker, ("Unknown", "—"))
    st.caption(f"**{company_name}** · {sector}")

with ctrl_col2:
    amount = st.number_input(
        "💵 Investment Amount ($)",
        min_value=10,
        max_value=1_000_000,
        value=500,
        step=50,
        help="The virtual amount of money you are placing on this bet.",
    )

ctrl_col3, ctrl_col4, ctrl_col5 = st.columns([1, 1, 1])

with ctrl_col3:
    direction_raw = st.radio(
        "📊 Your Direction",
        ["📈 UP (Long)", "📉 DOWN (Short)"],
        horizontal=True,
        help="UP = you profit if the price rises. DOWN = you profit if the price falls.",
    )
    direction: str = "UP" if "UP" in direction_raw else "DOWN"

with ctrl_col4:
    leverage: int = st.select_slider(
        "⚡ Leverage",
        options=LEVERAGE_OPTIONS,
        value=1,
        format_func=lambda x: f"{x}x",
        help=(
            "Multiplies both gains AND losses. "
            "1x = no leverage (safest). 10x = high risk, high reward."
        ),
    )

with ctrl_col5:
    horizon_label: str = st.radio(
        "⏱️ Time Horizon",
        list(HORIZON_OPTIONS.keys()),
        help="How long to hold the position before settling the bet.",
    )
    n_days: int = HORIZON_OPTIONS[horizon_label]
    # Clean label without the emoji prefix for display, e.g. "1 Day (Short Term)"
    horizon_clean: str = " ".join(horizon_label.split()[1:])

st.divider()

simulate_btn = st.button(
    "🎲 Simulate Bet",
    use_container_width=True,
    type="primary",
)

# ── Simulation ─────────────────────────────────────────────────────────────────

if simulate_btn:
    with st.spinner("Running simulation..."):
        # Try the SimFin API first so the simulation uses today's real data.
        # Falls back to the local CSV automatically if the API is unavailable.
        df, data_source = load_ticker_data(ticker, n_days)

    if df.empty:
        st.error(f"No data found for {ticker}. Run the ETL pipeline first.")
        st.stop()

    feature_cols    = get_feature_cols(ticker)
    available_fcols = [c for c in feature_cols if c in df.columns]

    if not available_fcols:
        st.error("Feature columns not found in processed data.")
        st.stop()

    # Drop rows with NaN in any required feature to get a clean dataset.
    df_clean = df.dropna(subset=available_fcols).reset_index(drop=True)

    # We need at least n_days + 1 clean rows: the entry row plus the full period.
    if len(df_clean) < n_days + 2:
        st.error(
            f"Not enough clean data for **{ticker}** to simulate a {horizon_clean} bet "
            f"(need ≥ {n_days + 2} rows, have {len(df_clean)})."
        )
        st.stop()

    # Entry = n_days before the last row; Exit = last row.
    entry_row = df_clean.iloc[-(n_days + 1)]
    exit_row  = df_clean.iloc[-1]

    # For the chart we show a slightly wider window for visual context.
    chart_lookback = n_days + max(30, n_days // 4)
    df_period = df_clean.iloc[-chart_lookback:].copy()

    # ── Prices and actual return ───────────────────────────────────────────────
    price_col   = "Adj. Close" if "Adj. Close" in df_clean.columns else "Close"
    entry_price = float(entry_row[price_col])
    exit_price  = float(exit_row[price_col])
    actual_return = (exit_price - entry_price) / entry_price

    entry_date = entry_row["Date"]
    exit_date  = exit_row["Date"]

    # ── Model prediction at the entry date ────────────────────────────────────
    model      = load_model(ticker)
    signal     = Signal.HOLD
    confidence = 0.5

    if model is not None:
        try:
            X = entry_row[available_fcols].values.reshape(1, -1)
            pred_class = int(model.predict(X)[0])
            pred_proba = model.predict_proba(X)[0]
            confidence = float(max(pred_proba))
            signal     = prediction_to_signal(pred_class, confidence)
        except Exception as exc:
            st.warning(f"Model prediction unavailable: {exc}")
    else:
        st.warning("Trained model not found — showing simulation without AI signal.")

    # ── Win / Loss calculation ─────────────────────────────────────────────────
    actual_direction = "UP" if actual_return > 0 else "DOWN"
    won = (direction == actual_direction)

    # Signed return from the user's perspective: positive = won, negative = lost.
    signed_return     = actual_return if direction == "UP" else -actual_return
    leveraged_return  = max(signed_return * leverage, -1.0)   # floor at -100%
    pnl               = amount * leveraged_return
    final_amount      = amount + pnl

    # Does the model's signal agree with the user's bet direction?
    model_direction = (
        "UP"   if signal == Signal.BUY  else
        "DOWN" if signal == Signal.SELL else
        "HOLD"
    )
    model_agrees = (model_direction != "HOLD") and (model_direction == direction)

    # ── Historical OOS win rate ────────────────────────────────────────────────
    win_rate, n_oos = compute_historical_win_rate(ticker)

    # ── Top signal-driving features ───────────────────────────────────────────
    top_features = get_top_feature_drivers(entry_row, df_clean, available_fcols, n=3)

    # Load the full CSV history for the return distribution chart.
    # The API window (df_clean) only has ~200 days — not enough for a meaningful histogram.
    # The full CSV has 5 years of data, which gives a much better distribution picture.
    df_full = _load_processed_csv(ticker)
    if df_full is None:
        df_full = df_clean  # fall back to the API window if CSV is unavailable

    # ── Lottie animation ──────────────────────────────────────────────────────
    lottie_data = None
    if LOTTIE_AVAILABLE:
        url = (
            "https://assets5.lottiefiles.com/packages/lf20_jR229r.json"   # confetti (win)
            if won else
            "https://assets3.lottiefiles.com/packages/lf20_qp1q7mct.json"  # sad (loss)
        )
        lottie_data = load_lottie_url(url)

    # ──────────────────────────────────────────────────────────────────────────
    # RESULTS SECTION
    # ──────────────────────────────────────────────────────────────────────────

    # Show where the data came from so users know if it is live or cached.
    if data_source == "live":
        st.success("🟢 **Live data** — fetched fresh from the SimFin API today.")
    else:
        st.warning("⚠️ **Cached data** — API unavailable, using the local processed CSV.")

    outcome_color = "#00c805" if won else "#ff4b4b"
    outcome_label = "✅ YOU WON!" if won else "❌ YOU LOST"
    pnl_sign      = "+" if pnl >= 0 else ""

    st.divider()
    st.markdown("## Results")

    # ── Outcome header row ─────────────────────────────────────────────────────
    header_col1, header_col2, header_col3 = st.columns([1, 2, 1])

    with header_col1:
        if LOTTIE_AVAILABLE and lottie_data is not None:
            st_lottie(lottie_data, height=150, key="outcome_anim")
        else:
            emoji = "🎉" if won else "😔"
            st.markdown(
                f"<h1 style='text-align:center;font-size:64px;line-height:1.2'>{emoji}</h1>",
                unsafe_allow_html=True,
            )
            if won:
                st.balloons()

    with header_col2:
        st.markdown(
            f"<h2 style='text-align:center;color:{outcome_color};margin-bottom:6px'>"
            f"{outcome_label}</h2>",
            unsafe_allow_html=True,
        )
        dir_color = "#00c805" if direction == "UP" else "#ff4b4b"
        dir_label = "📈 UP" if direction == "UP" else "📉 DOWN"
        st.markdown(
            f"<p style='text-align:center;font-size:17px;'>"
            f"You bet <b>${amount:,.0f}</b> "
            f"<b style='color:{dir_color}'>{dir_label}</b> "
            f"on <b>{ticker}</b> over <b>{horizon_clean}</b> "
            f"with <b>{leverage}x leverage</b>.</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='text-align:center;font-size:24px;font-weight:bold;'>"
            f"<span style='color:{outcome_color};'>"
            f"{pnl_sign}${pnl:,.2f}</span>"
            f"<span style='font-size:14px;color:#aaa;font-weight:normal;'> "
            f"({leveraged_return*100:+.1f}% leveraged)</span></p>",
            unsafe_allow_html=True,
        )

        # Show whether user aligned with the model signal.
        if model_direction != "HOLD":
            agree_icon  = "✅" if model_agrees else "⚠️"
            agree_text  = "Your bet aligned with the AI signal" if model_agrees else "Your bet was against the AI signal"
            agree_color = "#00c805" if model_agrees else "#ffd600"
            st.markdown(
                f"<p style='text-align:center;font-size:13px;color:{agree_color};'>"
                f"{agree_icon} {agree_text} "
                f"(Model: {SIGNAL_STYLE[signal]['emoji']} {signal.value}, "
                f"score {confidence*100:.0f}%)</p>",
                unsafe_allow_html=True,
            )

    with header_col3:
        st.plotly_chart(
            render_confidence_gauge(confidence, signal),
            use_container_width=True,
        )

    # ── Key metrics (custom HTML cards) ────────────────────────────────────────
    st.divider()

    m1, m2, m3, m4, m5 = st.columns(5)

    with m1:
        st.markdown(
            _metric_card(
                "Entry Price", f"${entry_price:,.2f}",
                sub_text=pd.Timestamp(entry_date).strftime("%b %d, %Y"),
            ),
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            _metric_card(
                "Exit Price", f"${exit_price:,.2f}",
                delta_text=f"{actual_return * 100:+.2f}%",
                delta_positive=(actual_return >= 0),
                sub_text=pd.Timestamp(exit_date).strftime("%b %d, %Y"),
            ),
            unsafe_allow_html=True,
        )
    with m3:
        ret_color = "#00c805" if actual_return >= 0 else "#ff4b4b"
        st.markdown(
            _metric_card(
                "Actual Return", f"{actual_return * 100:+.2f}%",
                value_color=ret_color,
                sub_text="Not leveraged",
            ),
            unsafe_allow_html=True,
        )
    with m4:
        pnl_color = "#00c805" if pnl >= 0 else "#ff4b4b"
        st.markdown(
            _metric_card(
                f"Your P&L  ({leverage}x)", f"{pnl_sign}${abs(pnl):,.2f}",
                value_color=pnl_color,
                delta_text=f"{leveraged_return * 100:+.1f}%",
                delta_positive=(pnl >= 0),
                sub_text=f"Leverage: {leverage}x",
            ),
            unsafe_allow_html=True,
        )
    with m5:
        final_color = "#00c805" if final_amount >= amount else "#ff4b4b"
        st.markdown(
            _metric_card(
                "Final Amount", f"${final_amount:,.2f}",
                value_color=final_color,
                sub_text=f"Started: ${amount:,.0f}",
            ),
            unsafe_allow_html=True,
        )

    # ── Tabs: deeper analysis below the always-visible result cards ───────────
    st.divider()
    tab_chart, tab_analysis, tab_history = st.tabs(
        ["📈 Price Chart", "🧠 Analysis", "📚 Historical Context"]
    )

    # ── Tab 1: Price chart ─────────────────────────────────────────────────────
    with tab_chart:
        fig_chart = render_price_chart(
            df_period, ticker, horizon_clean, entry_date, exit_date, price_col
        )
        st.plotly_chart(fig_chart, use_container_width=True)

    # ── Tab 2: Analysis ────────────────────────────────────────────────────────
    with tab_analysis:
        sig_emoji = SIGNAL_STYLE[signal]["emoji"]
        sig_label = SIGNAL_STYLE[signal]["label"]

        if model is None:
            st.info("Train the model (`python model/train.py --all`) to see the AI signal.")

        if top_features:
            drivers_col, explain_col = st.columns([3, 2])

            with drivers_col:
                st.markdown(f"#### 📡 Signal Drivers — {sig_emoji} {sig_label}")
                st.caption(
                    f"Entry date: {pd.Timestamp(entry_date).strftime('%b %d, %Y')}. "
                    "Bars show how many standard deviations (σ) each feature was "
                    "away from its historical average at the moment the model predicted."
                )
                st.plotly_chart(render_feature_chart(top_features), use_container_width=True)

            with explain_col:
                st.markdown("#### How to Read This")
                st.markdown(
                    "Each bar represents one of the features the model used to make "
                    "its prediction. The number on each bar (e.g. **+1.8σ**) shows "
                    "how unusual that feature value was:\n\n"
                    "- **Positive bar (green)** → feature was above its historical average\n"
                    "- **Negative bar (red)** → feature was below its historical average\n"
                    "- **Longer bar** → stronger deviation, bigger influence on the signal\n\n"
                    "A σ (sigma) of ±1 means the value was in the top/bottom ~16% of "
                    "all historical observations for this stock."
                )

        st.markdown(f"### Why the Model Said {sig_emoji} {sig_label}")
        st.caption(
            "The three features that deviated most from their historical average on the "
            "entry date — these most likely drove the model's signal."
        )

        if top_features:
            feat_cols = st.columns(len(top_features))
            for col, (feat_name, z_score) in zip(feat_cols, top_features):
                with col:
                    icon       = "🔺" if z_score > 0 else "🔻"
                    card_color = "#00c805" if z_score > 0 else "#ff4b4b"
                    text       = interpret_feature(feat_name, z_score)
                    st.markdown(
                        f"<div style='background:#1a1a2e;border:1px solid rgba(255,255,255,0.1);"
                        f"border-left:4px solid {card_color};border-radius:8px;"
                        f"padding:14px 16px;height:100%;'>"
                        f"<p style='margin:0 0 8px;font-size:15px;font-weight:700;color:#fff;'>"
                        f"{icon} {feat_name}</p>"
                        f"<p style='margin:0 0 8px;'>"
                        f"<code style='font-size:13px;color:{card_color};"
                        f"background:rgba(255,255,255,0.08);padding:3px 8px;"
                        f"border-radius:4px;'>{z_score:+.2f}σ</code></p>"
                        f"<p style='margin:0;font-size:13px;color:#cccccc;line-height:1.55;'>"
                        f"{text}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    # ── Tab 3: Historical context ──────────────────────────────────────────────
    with tab_history:
        st.markdown("### 📈 Where Does Your Return Rank Historically?")
        st.caption(
            f"Distribution of all historical {horizon_clean.lower()} returns for {ticker} "
            f"(based on {len(df_full):,} days of data). The dashed line marks your simulation's result."
        )
        fig_dist = render_return_distribution(df_full, actual_return, n_days, horizon_clean, ticker)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.divider()
        st.markdown("### 📊 What If You'd Used Different Leverage?")
        st.caption(
            f"Same bet direction ({direction}) and amount (${amount:,.0f}) — only leverage varies."
        )

        scenario_rows = []
        for lev in [1, 2, 5, 10]:
            lev_ret   = max(signed_return * lev, -1.0)
            lev_pnl   = amount * lev_ret
            lev_final = amount + lev_pnl
            scenario_rows.append({
                "Leverage":         f"{lev}x  {'◀ your pick' if lev == leverage else ''}",
                "Invested":         f"${amount:,.0f}",
                "Leveraged Return": f"{lev_ret*100:+.1f}%",
                "P&L":              f"{'+'if lev_pnl>=0 else ''} ${lev_pnl:,.2f}",
                "Final Amount":     f"${lev_final:,.2f}",
                "Outcome":          "✅ Win" if won else "❌ Loss",
            })

        st.dataframe(
            pd.DataFrame(scenario_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()
        if win_rate is not None:
            acc_pct   = win_rate * 100
            acc_color = "#00c805" if acc_pct > 55 else ("#ffd600" if acc_pct > 50 else "#ff4b4b")
            st.markdown(
                f"<p style='text-align:center;font-size:15px;'>"
                f"📈 On <b>{ticker}</b>, the model correctly predicted the next-day direction "
                f"<b style='color:{acc_color};'>{acc_pct:.1f}%</b> of the time "
                f"across its <b>{n_oos:,}</b> out-of-sample trading days.  "
                f"<span style='color:#888;'>(Accuracy above 50% means the model "
                f"beats a random coin flip.)</span></p>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Historical accuracy not available — train the model first.")

    # ── Disclaimer footer ──────────────────────────────────────────────────────
    st.divider()
    st.caption(
        "⚠️ All results shown are based on historical data and are hypothetical. "
        "Past performance does not guarantee future results. "
        "This simulation uses the most recent period available in our processed dataset "
        "and is intended purely for educational purposes."
    )
