"""
train.py — Production ML training script for the trading system.

Two pooled models are trained — one per feature schema:

  model_pooled.pkl           Standard model (25 tickers, 16 features)
                             Covers all tickers except the 5 fallback ones.

  model_pooled_fallback.pkl  Fallback model (5 tickers, 11 features)
                             Covers BAC, GS, JPM, MA, V — tickers where
                             fundamental data is unavailable or structurally
                             incompatible (banks / payment networks).

Training strategy:
  - 80/20 temporal split applied PER TICKER before pooling.
    Each ticker contributes its own final 20% period to the test set,
    preventing the test set from being dominated by whichever tickers
    appear last alphabetically in the concatenated pool.
  - Four models are evaluated: Ridge (with StandardScaler + alpha CV),
    RandomForestRegressor, GradientBoostingRegressor, and LightGBM.
  - StandardScaler is applied ONLY to Ridge. Tree-based models
    (RF, GBR, LightGBM) are scale-invariant and do not require normalisation.
  - Ridge alpha is selected via TimeSeriesSplit(n_splits=5) CV on the
    training set, optimising for mean direction accuracy across folds.
  - The model with the highest test R² is saved as the .pkl file.
  - After model selection, direction accuracy and signal-bin breakdown
    are logged for both the winning model and all candidates.

Usage:
    # Train BOTH models — covers all 30 tickers (recommended)
    python model/train.py --all

    # Train a single-ticker model for quick local testing only
    python model/train.py --ticker AAPL
"""

import argparse
import glob
import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
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
THRESHOLDS_FILE     = "thresholds.json"

# Signal bin thresholds (fraction, matching strategy.py 5-tier scheme).
_HIGH_RISE =  0.02
_LOW_RISE  =  0.005
_LOW_FALL  = -0.005
_HIGH_FALL = -0.02

# Ridge alpha candidates for cross-validation.
_RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


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


def _build_ridge(alpha: float = 1.0) -> Pipeline:
    """Ridge regression wrapped in StandardScaler.

    StandardScaler is required because Market_Cap (~10^12) and Log_Return
    (~0.01) are on vastly different scales, which distorts coefficient
    estimation without normalisation. Ridge adds L2 regularisation (α‖w‖²)
    which reduces variance compared to plain OLS (LinearRegression).
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def _select_ridge_alpha(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    label: str,
) -> float:
    """Select the best Ridge alpha via TimeSeriesSplit CV on the training set.

    Evaluates each alpha in _RIDGE_ALPHAS using 5-fold time-series CV.
    Selection criterion: mean direction accuracy (sign(y_pred)==sign(y_true))
    across folds — the most operationally relevant metric for a trading system.

    Parameters
    ----------
    X_train : feature matrix (training split only — never touches test set)
    y_train : target vector
    label   : pool label for log messages ("standard" or "fallback")

    Returns the selected alpha value.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    best_alpha, best_score = _RIDGE_ALPHAS[0], -float("inf")
    scores_by_alpha = {}

    for alpha in _RIDGE_ALPHAS:
        fold_scores = []
        for fold_train_idx, fold_val_idx in tscv.split(X_train):
            X_f_train = X_train.iloc[fold_train_idx]
            y_f_train = y_train.iloc[fold_train_idx]
            X_f_val   = X_train.iloc[fold_val_idx]
            y_f_val   = y_train.iloc[fold_val_idx]

            pipe = _build_ridge(alpha)
            pipe.fit(X_f_train, y_f_train)
            preds = pipe.predict(X_f_val)
            dir_acc = np.mean(np.sign(preds) == np.sign(y_f_val.values))
            fold_scores.append(dir_acc)

        mean_score = float(np.mean(fold_scores))
        scores_by_alpha[alpha] = mean_score
        if mean_score > best_score:
            best_score, best_alpha = mean_score, alpha

    scores_str = "  ".join(
        f"α={a}: {s:.4f}" for a, s in scores_by_alpha.items()
    )
    logger.info(
        f"[{label}] Ridge alpha CV (TimeSeriesSplit n=5, metric=dir_acc):  {scores_str}"
    )
    logger.info(
        f"[{label}] Selected Ridge alpha={best_alpha}  "
        f"(CV mean dir_acc={best_score:.4f})"
    )
    return best_alpha


def _build_rf() -> RandomForestRegressor:
    """RandomForestRegressor — tree-based, scale-invariant, no scaler needed."""
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)


def _build_gbr() -> GradientBoostingRegressor:
    """GradientBoostingRegressor — tree-based, scale-invariant, no scaler needed."""
    return GradientBoostingRegressor(n_estimators=200, random_state=42)


def _build_lgbm():
    """LightGBM regressor — tree-based, scale-invariant, no scaler needed.

    Returns None if lightgbm is not installed (ImportError handled in caller).
    Hyperparameters chosen to balance capacity and regularisation:
      - n_estimators=500 with low learning_rate=0.03 (slow learning, less overfit)
      - num_leaves=31 (default — keeps trees shallow)
      - min_child_samples=30 (minimum leaf size — key regulariser for small pools)
      - colsample_bytree=0.8, subsample=0.7 (stochastic, adds regularisation)
    """
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            min_child_samples=30,
            colsample_bytree=0.8,
            subsample=0.7,
            random_state=42,
            n_jobs=-1,
            verbose=-1,  # suppress LightGBM's own stdout logging
        )
    except ImportError:
        return None


def discover_tickers() -> list[str]:
    """Return sorted list of tickers that have a processed CSV in data/processed/."""
    csv_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.csv")))
    return [os.path.basename(f).replace(".csv", "") for f in csv_files]


def _split_ticker(
    df: pd.DataFrame, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply 80/20 temporal split to a single ticker's clean rows.

    NaN rows are dropped before splitting so the split index is computed
    on usable rows only — avoiding rolling-window warm-up rows inflating
    the apparent training size. Rows are already date-sorted by load_processed.
    """
    df = df.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    return df.iloc[:split_idx], df.iloc[split_idx:]


def _log_metrics(label: str, model_name: str, y_true, y_pred) -> float:
    """Compute and log MAE, RMSE, R² for one model on the test set. Returns R²."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    logger.info(
        f"[{label}] {model_name:40s}  MAE={mae:.6f}  RMSE={rmse:.6f}  R²={r2:.4f}"
    )
    return r2


def _log_direction_and_signals(
    label: str, model_name: str, y_true: np.ndarray, y_pred: np.ndarray
) -> None:
    """Log direction accuracy and signal-bin breakdown for the winning model.

    Direction accuracy: fraction of test rows where sign(y_pred)==sign(y_true).
    Signal bins mirror the 5-tier trading signal scheme used in strategy.py:
      HIGH RISE  (>+2%)   → BUY
      LOW RISE   (+0.5–2%)→ BUY
      STAY       (±0.5%)  → HOLD
      LOW FALL   (-2–-0.5%)→ SELL
      HIGH FALL  (<-2%)   → SELL
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    dir_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
    logger.info(
        f"[{label}] {model_name} — Direction accuracy on test set: "
        f"{dir_acc:.4f} ({dir_acc*100:.2f}%)  [n={n}]"
    )

    # Signal bin counts (based on predicted return).
    bins = {
        "HIGH RISE  (pred > +2%)      ": y_pred >  _HIGH_RISE,
        "LOW RISE   (+0.5% to +2%)    ": (y_pred >= _LOW_RISE) & (y_pred <= _HIGH_RISE),
        "STAY       (-0.5% to +0.5%)  ": (y_pred > _LOW_FALL) & (y_pred < _LOW_RISE),
        "LOW FALL   (-2% to -0.5%)    ": (y_pred >= _HIGH_FALL) & (y_pred <= _LOW_FALL),
        "HIGH FALL  (pred < -2%)      ": y_pred <  _HIGH_FALL,
    }

    logger.info(f"[{label}] Signal-bin breakdown (predicted distribution on test set):")
    for bin_name, mask in bins.items():
        count = int(mask.sum())
        pct   = count / n * 100
        # Within-bin direction accuracy
        if count > 0:
            bin_dir_acc = np.mean(np.sign(y_pred[mask]) == np.sign(y_true[mask]))
            logger.info(
                f"[{label}]   {bin_name}  n={count:5d} ({pct:5.1f}%)  "
                f"within-bin dir_acc={bin_dir_acc:.3f}"
            )
        else:
            logger.info(f"[{label}]   {bin_name}  n=    0 (  0.0%)")


# ── Internal training helper ──────────────────────────────────────────────────

def _train_pooled(
    tickers: list[str],
    feature_cols: list[str],
    model_filename: str,
    label: str,
) -> str:
    """Pool data from `tickers`, train on `feature_cols`, save to `model_filename`.

    Per-ticker temporal split is applied before pooling: each ticker's own
    final 20% of rows is reserved for the test set. This prevents the test
    set from being dominated by whichever tickers appear last alphabetically
    in the concatenated pool.

    Four models are trained and evaluated on the held-out test set:
      - Ridge (with StandardScaler + alpha selected via TimeSeriesSplit CV)
      - RandomForestRegressor  (scale-invariant tree model — no scaler)
      - GradientBoostingRegressor  (scale-invariant tree model — no scaler)
      - LightGBMRegressor  (scale-invariant — skipped if not installed)

    The model with the highest test R² is saved as the .pkl file.
    After selection, direction accuracy and signal-bin breakdown are logged.

    Parameters
    ----------
    tickers        : ticker symbols to include in the training pool
    feature_cols   : feature column list (STANDARD_FEATURE_COLS or FALLBACK_FEATURE_COLS)
    model_filename : output filename inside model/trained/
    label          : label for log messages, e.g. "standard" or "fallback"

    Returns the path to the saved .pkl file.
    """
    logger.info(f"[{label}] Loading data for {len(tickers)} tickers: {tickers}")

    train_frames, test_frames, skipped = [], [], []
    for ticker in tickers:
        try:
            df = load_processed(ticker)
            df["Ticker"] = ticker   # tag for logging only — not fed to the model
            train_ticker, test_ticker = _split_ticker(df, feature_cols)
            train_frames.append(train_ticker)
            test_frames.append(test_ticker)
        except Exception as e:
            logger.warning(f"[{ticker}] Skipped — could not load: {e}")
            skipped.append(ticker)

    if not train_frames:
        raise RuntimeError(f"[{label}] No ticker data could be loaded. Aborting.")

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df  = pd.concat(test_frames,  ignore_index=True)

    logger.info(
        f"[{label}] Temporal split applied per ticker: 80% train / 20% test  "
        f"(train={len(train_df):,} rows, test={len(test_df):,} rows, "
        f"tickers={len(train_frames)}, skipped={len(skipped)})"
    )

    if len(train_df) < 50:
        raise ValueError(
            f"[{label}] Only {len(train_df)} train rows — not enough to train reliably."
        )

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COL]
    X_test  = test_df[feature_cols]
    y_test  = test_df[TARGET_COL]

    # ── Select Ridge alpha via TimeSeriesSplit CV (training set only) ─────────
    best_alpha = _select_ridge_alpha(X_train, y_train, label)

    # ── Build candidate list ──────────────────────────────────────────────────
    candidates = [
        (f"Ridge (α={best_alpha}, + StandardScaler)", _build_ridge(best_alpha)),
        ("RandomForestRegressor",                      _build_rf()),
        ("GradientBoostingRegressor",                  _build_gbr()),
    ]

    lgbm_model = _build_lgbm()
    if lgbm_model is not None:
        candidates.append(("LightGBMRegressor", lgbm_model))
    else:
        logger.warning(
            f"[{label}] LightGBM not installed — skipping. "
            "Install with: pip install lightgbm"
        )

    logger.info(
        f"[{label}] Training {len(candidates)} models on {len(X_train):,} rows "
        f"({len(feature_cols)} features)…"
    )

    # ── Train all candidates, evaluate on held-out test set ───────────────────
    best_name, best_model, best_r2 = None, None, -float("inf")
    for name, model in candidates:
        model.fit(X_train, y_train)
        r2 = _log_metrics(label, name, y_test, model.predict(X_test))
        if r2 > best_r2:
            best_r2, best_name, best_model = r2, name, model

    logger.info(
        f"[{label}] Best model: {best_name}  (test R²={best_r2:.4f}) → saving"
    )

    # ── Direction accuracy + signal-bin breakdown for the winner ──────────────
    _log_direction_and_signals(
        label, best_name, y_test.values, best_model.predict(X_test)
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(TRAINED_DIR, exist_ok=True)
    model_path = os.path.join(TRAINED_DIR, model_filename)
    joblib.dump(best_model, model_path)
    logger.info(f"[{label}] Saved → {model_path}")

    counts = train_df.groupby("Ticker").size().sort_values(ascending=False)
    logger.info(f"[{label}] Training rows per ticker:\n{counts.to_string()}")

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

    pipeline = _build_gbr()
    pipeline.fit(df[feature_cols], df[TARGET_COL])

    os.makedirs(TRAINED_DIR, exist_ok=True)
    model_path = os.path.join(TRAINED_DIR, f"model_{ticker}.pkl")
    joblib.dump(pipeline, model_path)

    logger.info(f"[{ticker}] Saved → {model_path}  ({len(df)} rows, {schema} schema)")
    return model_path


# ── Threshold computation ─────────────────────────────────────────────────────

def compute_and_save_thresholds() -> dict:
    """Compute percentile-based signal thresholds from each pooled model's
    prediction distribution and save them to model/trained/thresholds.json.

    For each pool (standard, fallback):
      - Loads all processed CSVs for the relevant tickers.
      - Runs model.predict() on every usable row (after NaN removal).
      - Sets buy threshold = 75th percentile of predictions,
              sell threshold = 25th percentile of predictions.

    Because Ridge with high alpha shrinks all predictions toward zero, the
    hardcoded ±0.5% thresholds in the original strategy.py always map to HOLD.
    Percentile-based thresholds adapt to whatever range the model actually
    produces — the top 25% of predictions generate BUY signals and the bottom
    25% generate SELL signals regardless of the absolute magnitudes.

    Returns the thresholds dict:
        {"standard": {"buy": float, "sell": float},
         "fallback":  {"buy": float, "sell": float}}

    The file is loaded at startup by go_live.py and backtesting.py.
    If the file is missing those pages fall back to legacy integer signals.
    """
    all_tickers      = discover_tickers()
    standard_tickers = [t for t in all_tickers if not is_fallback_ticker(t)]
    fallback_tickers  = [t for t in all_tickers if is_fallback_ticker(t)]

    configs = [
        ("standard", standard_tickers, STANDARD_FEATURE_COLS, STANDARD_MODEL_FILE),
        ("fallback",  fallback_tickers,  FALLBACK_FEATURE_COLS,  FALLBACK_MODEL_FILE),
    ]

    thresholds = {}
    for label, tickers, feature_cols, model_file in configs:
        model_path = os.path.join(TRAINED_DIR, model_file)
        if not os.path.exists(model_path):
            logger.warning(
                f"[{label}] Model not found at {model_path} — "
                "skipping threshold computation for this pool."
            )
            continue

        model = joblib.load(model_path)

        frames = []
        for ticker in tickers:
            try:
                df = load_processed(ticker)
                df = df.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)
                frames.append(df[feature_cols])
            except Exception as e:
                logger.warning(f"[{ticker}] Skipped in threshold computation: {e}")

        if not frames:
            logger.warning(f"[{label}] No usable data — cannot compute thresholds.")
            continue

        X_all = pd.concat(frames, ignore_index=True)
        preds = model.predict(X_all)

        buy_threshold  = float(np.percentile(preds, 75))
        sell_threshold = float(np.percentile(preds, 25))

        thresholds[label] = {"buy": buy_threshold, "sell": sell_threshold}
        logger.info(
            f"[{label}] Thresholds computed from {len(preds):,} predictions — "
            f"buy (p75): {buy_threshold:+.6f}   sell (p25): {sell_threshold:+.6f}"
        )

    out_path = os.path.join(TRAINED_DIR, THRESHOLDS_FILE)
    os.makedirs(TRAINED_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    logger.info(f"Thresholds saved → {out_path}")

    return thresholds


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
            compute_and_save_thresholds()
        else:
            train_ticker(args.ticker)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
