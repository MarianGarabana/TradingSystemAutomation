"""
strategy.py — Buy / Sell / Hold signal logic based on model predictions.
"""

from enum import Enum


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


def prediction_to_signal(prediction: int) -> Signal:
    """
    Convert a raw model prediction to a trading signal.

    Convention (adjust to match your target encoding):
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
    """
    Simple backtest: compare predicted signals to actual price movements.

    Returns a dict with accuracy, total trades, and win rate.
    """
    # TODO: implement backtesting logic
    raise NotImplementedError("backtest not yet implemented")
