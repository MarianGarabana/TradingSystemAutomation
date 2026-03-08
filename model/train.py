"""
train.py — Production ML training script for the trading system.

Two pooled models are trained — one per feature schema:

  model_pooled.pkl           Standard model (25 tickers, 16 features)
                             Covers all tickers except the 5 fallback ones.

  model_pooled_fallback.pkl  Fallback model (5 tickers, 11 features)
                             Covers BAC, GS, JPM, MA, V — tickers where
                             fundamental data is unavailable or structurally
                             incompatible (banks / payment networks).

Usage:
    # Train BOTH models — covers all 30 tickers (recommended)
    python model/train.py --all

    # Train a single-ticker model for quick local testing only
    python model/train.py --ticker AAPL
"""

import argparse
import glob
import logging
import os
import sys

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path so we can import from model/strategy.py.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.strategy import (
    STANDARD_FEATURE_COLS,
    FALLBACK_FEATURE_COLS,
    FALLBACK_TICKERS,
    is_fallback_ticker,
    get_feature_cols,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
TRAINED_DIR   = os.path.join(os.path.dirname(__file__), "trained")
TARGET_COL    = "Target"

# Model filenames — one per feature schema.
STANDARD_MODEL_FILE = "model_pooled.pkl"
FALLBACK_MODEL_FILE = "model_pooled_fallback.pkl"


# ── Core helpers ──────────────────────────────────────────────────────────────

def load_processed(ticker: str) -> pd.DataFrame:
    """Load the ETL-processed CSV for a ticker."""
    path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No processed data found for '{ticker}'. "
            f"Run:  python etl/etl.py --ticker {ticker}"
        )
    return pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)


def build_pipeline() -> Pipeline:
    """Return the sklearn Pipeline: StandardScaler → GradientBoostingRegressor.

    StandardScaler normalises all features to zero mean / unit variance so that
    Market_Cap (~10^12) and MACD (~0.01–10) are on the same scale for the model.
    GradientBoostingRegressor is typically the best performer on tabular financial data.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg", GradientBoostingRegressor(n_estimators=200, random_state=42)),
    ])


def discover_tickers() -> list[str]:
    """Return sorted list of tickers that have a processed CSV in data/processed/."""
    csv_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.csv")))
    return [os.path.basename(f).replace(".csv", "") for f in csv_files]


# ── Internal training helper ──────────────────────────────────────────────────

def _train_pooled(
    tickers: list[str],
    feature_cols: list[str],
    model_filename: str,
    label: str,
) -> str:
    """Pool data from `tickers`, train on `feature_cols`, save to `model_filename`.

    Parameters
    ----------
    tickers        : ticker symbols to include in the training pool
    feature_cols   : feature column list (STANDARD_FEATURE_COLS or FALLBACK_FEATURE_COLS)
    model_filename : output filename inside model/trained/
    label          : label for log messages, e.g. "standard" or "fallback"

    Returns the path to the saved .pkl file.
    """
    logger.info(f"[{label}] Loading data for {len(tickers)} tickers: {tickers}")

    frames, skipped = [], []
    for ticker in tickers:
        try:
            df = load_processed(ticker)
            df["Ticker"] = ticker   # tag for logging only — not fed to the model
            frames.append(df)
        except Exception as e:
            logger.warning(f"[{ticker}] Skipped — could not load: {e}")
            skipped.append(ticker)

    if not frames:
        raise RuntimeError(f"[{label}] No ticker data could be loaded. Aborting.")

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        f"[{label}] Combined: {len(combined):,} rows across {len(frames)} tickers "
        f"(skipped {len(skipped)})"
    )

    # Drop rows where any required feature or the target is NaN.
    # For the standard schema this removes rolling-window warm-up rows and the
    # last row of each ticker (Target = NaN because there is no next trading day).
    # For the fallback schema it removes only price-feature NaN rows since
    # fundamentals are not required.
    before = len(combined)
    combined = combined.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)
    logger.info(
        f"[{label}] After NaN drop: {len(combined):,} clean rows "
        f"(dropped {before - len(combined):,})"
    )

    if len(combined) < 50:
        raise ValueError(
            f"[{label}] Only {len(combined)} clean rows — not enough to train reliably."
        )

    X = combined[feature_cols]
    y = combined[TARGET_COL]

    logger.info(
        f"[{label}] Fitting GradientBoostingRegressor "
        f"({len(feature_cols)} features, {len(combined):,} rows)…"
    )
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    os.makedirs(TRAINED_DIR, exist_ok=True)
    model_path = os.path.join(TRAINED_DIR, model_filename)
    joblib.dump(pipeline, model_path)

    logger.info(f"[{label}] Saved → {model_path}")
    counts = combined.groupby("Ticker").size().sort_values(ascending=False)
    logger.info(f"[{label}] Rows per ticker:\n{counts.to_string()}")

    return model_path


# ── Public training functions ─────────────────────────────────────────────────

def train_pooled_standard() -> str:
    """Train the standard pooled model on all non-fallback tickers.

    Uses STANDARD_FEATURE_COLS (16 features: price + vol + fundamentals).
    Saves to model/trained/model_pooled.pkl.
    """
    all_tickers = discover_tickers()
    standard_tickers = [t for t in all_tickers if not is_fallback_ticker(t)]
    if not standard_tickers:
        raise FileNotFoundError(
            f"No standard-ticker CSVs found in {PROCESSED_DIR}. Run etl/etl.py first."
        )
    return _train_pooled(
        tickers=standard_tickers,
        feature_cols=STANDARD_FEATURE_COLS,
        model_filename=STANDARD_MODEL_FILE,
        label="standard",
    )


def train_pooled_fallback() -> str:
    """Train the fallback pooled model on BAC, GS, JPM, MA, V.

    Uses FALLBACK_FEATURE_COLS (11 features: price + vol, no fundamentals).
    Saves to model/trained/model_pooled_fallback.pkl.
    """
    all_tickers = discover_tickers()
    fallback_tickers = [t for t in all_tickers if is_fallback_ticker(t)]
    if not fallback_tickers:
        raise FileNotFoundError(
            f"No fallback-ticker CSVs found in {PROCESSED_DIR}. "
            f"Expected: {sorted(FALLBACK_TICKERS)}. Run etl/etl.py first."
        )
    return _train_pooled(
        tickers=fallback_tickers,
        feature_cols=FALLBACK_FEATURE_COLS,
        model_filename=FALLBACK_MODEL_FILE,
        label="fallback",
    )


def train_ticker(ticker: str) -> str:
    """Train a single-ticker model for quick local testing.

    Automatically selects the correct schema (standard or fallback) for the ticker.
    Note: the web app always loads pooled models — this function is for development only.
    """
    feature_cols = get_feature_cols(ticker)
    schema = "fallback" if is_fallback_ticker(ticker) else "standard"
    logger.info(f"Training single-ticker model for {ticker} ({schema} schema)…")

    df = load_processed(ticker)
    df = df.dropna(subset=feature_cols + [TARGET_COL])

    if len(df) < 50:
        raise ValueError(
            f"Only {len(df)} clean rows for '{ticker}' — not enough to train reliably."
        )

    pipeline = build_pipeline()
    pipeline.fit(df[feature_cols], df[TARGET_COL])

    os.makedirs(TRAINED_DIR, exist_ok=True)
    model_path = os.path.join(TRAINED_DIR, f"model_{ticker}.pkl")
    joblib.dump(pipeline, model_path)

    logger.info(f"[{ticker}] Saved → {model_path}  ({len(df)} rows, {schema} schema)")
    return model_path


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the pooled trading models covering all 30 tickers."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help=(
            f"Train BOTH pooled models (recommended): "
            f"{STANDARD_MODEL_FILE} (standard, 16 features) and "
            f"{FALLBACK_MODEL_FILE} (fallback for {sorted(FALLBACK_TICKERS)}, 11 features)."
        ),
    )
    group.add_argument(
        "--ticker",
        help="Train a single-ticker model for quick local testing (not used by the web app).",
    )
    args = parser.parse_args()

    try:
        if args.all:
            train_pooled_standard()
            train_pooled_fallback()
        else:
            train_ticker(args.ticker)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
