"""
strategy.py — Signal logic and canonical feature schemas for the trading system.

This module is the single source of truth for:
  - Feature column lists (STANDARD_FEATURE_COLS, FALLBACK_FEATURE_COLS)
  - Ticker classification (FALLBACK_TICKERS, is_fallback_ticker, get_feature_cols)
  - Signal enum and prediction-to-signal conversion

Both model/train.py and app/pages/ import from here to guarantee that the
feature schema used at training time always matches the schema used at inference.

NOTE — Fundamental features (Gross_Margin, Operating_Margin, Net_Margin,
Debt_to_Equity, Operating_CF_Ratio) are commented out of STANDARD_FEATURE_COLS
because the quarterly SimFin CSVs are not available locally.  Both schemas use
the same 11 price + volatility features for now.  When the API wrapper is
integrated and supplies live fundamental data, restore the commented block and
retrain with model/train.py --all.
"""

from enum import Enum


# ── Feature schemas ────────────────────────────────────────────────────────────

# Standard schema: price-based + volatility-normalised features.
# Fundamental ratios are commented out until quarterly data is available via API.
STANDARD_FEATURE_COLS = [
    # Price-based (6)
    "MA5", "MA20", "Volume_Change", "Market_Cap", "RSI", "MACD",
    # Volatility-normalised (5)
    "Log_Return", "Volatility_20", "Return_norm", "Return_norm_Lag1", "Return_norm_Lag2",
    # Fundamental (5) — restore when quarterly data is available via the API:
    # "Gross_Margin", "Operating_Margin", "Net_Margin",
    # "Debt_to_Equity", "Operating_CF_Ratio",
]

# Fallback schema: price + volatility features only — no fundamental ratios.
# Used for 5 tickers where quarterly data is structurally incompatible:
#   - Banks (BAC, GS, JPM): non-standard income statement format
#   - Payment networks (MA, V): no Gross Profit line → Gross_Margin undefined
FALLBACK_FEATURE_COLS = [
    # Price-based (6)
    "MA5", "MA20", "Volume_Change", "Market_Cap", "RSI", "MACD",
    # Volatility-normalised (5)
    "Log_Return", "Volatility_20", "Return_norm", "Return_norm_Lag1", "Return_norm_Lag2",
]

# The 5 tickers that use the fallback schema.
FALLBACK_TICKERS = {"BAC", "GS", "JPM", "MA", "V"}


def is_fallback_ticker(ticker: str) -> bool:
    """Return True if the ticker uses the fallback (price + vol only) schema."""
    return ticker.upper() in FALLBACK_TICKERS


def get_feature_cols(ticker: str) -> list[str]:
    """Return the correct feature column list for a given ticker.

    Standard tickers → STANDARD_FEATURE_COLS (16 features).
    Fallback tickers  → FALLBACK_FEATURE_COLS (11 features).
    """
    return FALLBACK_FEATURE_COLS if is_fallback_ticker(ticker) else STANDARD_FEATURE_COLS


# ── Signal logic ───────────────────────────────────────────────────────────────

class Signal(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


def prediction_to_signal(prediction: int) -> Signal:
    """Convert a raw model prediction integer to a trading signal.

    Convention:
        1  → BUY  (price expected to rise)
        -1 → SELL (price expected to fall)
        0  → HOLD
    """
    if prediction == 1:
        return Signal.BUY
    elif prediction == -1:
        return Signal.SELL
    else:
        return Signal.HOLD


def backtest(predictions, actuals) -> dict:
    """Simple backtest: compare predicted signals to actual price movements."""
    # TODO: implement backtesting logic
    raise NotImplementedError("backtest not yet implemented")
