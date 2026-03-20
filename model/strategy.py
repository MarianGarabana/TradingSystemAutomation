"""
strategy.py — Signal logic and canonical feature schemas for the trading system.

This module is the single source of truth for:
  - Feature column lists (STANDARD_FEATURE_COLS, FALLBACK_FEATURE_COLS)
  - Ticker classification (FALLBACK_TICKERS, is_fallback_ticker, get_feature_cols)
  - Signal enum and prediction-to-signal conversion

Both model/train.py and app/pages/ import from here to guarantee that the
feature schema used at training time always matches the schema used at inference.

Two schemas exist:
  STANDARD_FEATURE_COLS (16 features) — price + volatility + 5 fundamental ratios.
    Used for all tickers except the 5 fallback ones.  Fundamental columns are
    sourced from the quarterly SimFin CSVs during ETL (merge_asof, no look-ahead bias).
    For live inference in go_live.py, last-known fundamental values are attached
    from the processed CSV (valid because fundamentals change only quarterly).

  FALLBACK_FEATURE_COLS (11 features) — price + volatility only, no fundamentals.
    Used for BAC, GS, JPM, MA, V where quarterly data is structurally incompatible
    (bank/insurance income statement format; no Gross Profit line for MA/V).
"""

from enum import Enum


# ── Feature schemas ────────────────────────────────────────────────────────────

# Standard schema: price-based + volatility-normalised + fundamental ratio features.
STANDARD_FEATURE_COLS = [
    # Price-based (6)
    "MA5", "MA20", "Volume_Change", "Market_Cap", "RSI", "MACD",
    # Volatility-normalised (5)
    "Log_Return", "Volatility_20", "Return_norm", "Return_norm_Lag1", "Return_norm_Lag2",
    # Fundamental (5) — sourced from quarterly SimFin CSVs via point-in-time merge
    "Gross_Margin", "Operating_Margin", "Net_Margin",
    "Debt_to_Equity", "Operating_CF_Ratio",
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

    Standard tickers → STANDARD_FEATURE_COLS (16 features: 6 price + 5 vol + 5 fundamental).
    Fallback tickers  → FALLBACK_FEATURE_COLS (11 features: 6 price + 5 vol, no fundamentals).
    """
    return FALLBACK_FEATURE_COLS if is_fallback_ticker(ticker) else STANDARD_FEATURE_COLS


# ── Signal logic ───────────────────────────────────────────────────────────────

class Signal(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


def prediction_to_signal(prediction, confidence: float | None = None) -> Signal:
    """Convert a classifier prediction (0 or 1) and optional confidence to a Signal.

    Parameters
    ----------
    prediction : int
        1 → predicted up   (generates BUY if confident enough)
        0 → predicted down (generates SELL if confident enough)
    confidence : float | None
        max(predict_proba) for the predicted class.
        If provided and < 0.52, returns HOLD (low-confidence prediction).
        If None, confidence threshold is not applied.

    Returns
    -------
    Signal.BUY  — prediction == 1 and confidence >= 0.52 (or confidence not given)
    Signal.SELL — prediction == 0 and confidence >= 0.52 (or confidence not given)
    Signal.HOLD — confidence < 0.52 (low-confidence prediction)
    """
    if confidence is not None and confidence < 0.52:
        return Signal.HOLD
    if prediction == 1:
        return Signal.BUY
    return Signal.SELL


def backtest(_predictions, _actuals) -> dict:
    """Compare predicted signals to actual price movements.

    Note: full backtesting — portfolio simulation, rolling accuracy, and signal
    history — is implemented at the app layer in app/pages/backtesting.py
    (see run_backtest()). That function operates on the complete DataFrame with
    access to actual returns and the loaded model, which is the appropriate
    context for a simulation that needs to compound daily returns.

    This function is retained as an interface placeholder for any future
    extraction of pure metrics (accuracy, win rate) into this strategy module.
    """
    return {}
