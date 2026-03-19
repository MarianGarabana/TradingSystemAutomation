"""
train.py — Production ML training script for the trading system.

Two pooled models are trained — one per feature schema:

  model_pooled.pkl           Standard model (26 tickers, 16 features)
                             Covers all tickers except the 5 fallback ones.
                             Features: 6 price + 5 vol-normalised + 5 fundamental.

  model_pooled_fallback.pkl  Fallback model (5 tickers, 11 features)
                             Covers BAC, GS, JPM, MA, V — tickers where
                             fundamental data is unavailable or structurally
                             incompatible (banks / payment networks).
                             Features: 6 price + 5 vol-normalised (no fundamentals).

Training strategy:
  - Binary classification: target = 1 if next-day return > 0 else 0.
    Rows with exact zero next-day return are dropped (ambiguous direction).
  - 80/20 temporal split applied PER TICKER before pooling.
    Each ticker contributes its own final 20% period to the test set,
    preventing the test set from being dominated by whichever tickers
    appear last alphabetically in the concatenated pool.
  - Four models are evaluated: LogisticRegression (with StandardScaler + C CV),
    RandomForestClassifier, GradientBoostingClassifier, and LightGBM.
  - ALL classifiers use balanced class weights (v6):
      LogisticRegression      → class_weight='balanced' in constructor
      RandomForestClassifier  → class_weight='balanced' in constructor
      GradientBoostingClassifier → sample_weight=compute_sample_weight('balanced') in .fit()
      LGBMClassifier          → is_unbalance=True in constructor
    This compensates for the UP bias in daily returns (more UP days than DOWN)
    and ensures DOWN recall is not collapsed to near-zero.
  - StandardScaler is applied ONLY to LogisticRegression. Tree-based models
    (RF, GBR, LightGBM) are scale-invariant and do not require normalisation.
  - LogisticRegression C is selected via TimeSeriesSplit(n_splits=5) CV on the
    training set, optimising for mean accuracy across folds.
  - The model with the highest test accuracy is saved as the .pkl file.
  - After model selection, the full classification_report is logged.

Usage:
    # Train BOTH models — covers all 31 tickers (recommended)
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
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

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

# LogisticRegression C candidates for cross-validation.
_LR_C_VALUES = [0.01, 0.1, 1.0, 10.0]


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


def _build_logistic(C: float = 0.1) -> Pipeline:
    """LogisticRegression wrapped in StandardScaler.

    StandardScaler is required because Market_Cap (~10^12) and Log_Return
    (~0.01) are on vastly different scales, which distorts coefficient
    estimation without normalisation. class_weight='balanced' compensates
    for any residual class imbalance in the binary up/down target.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        )),
    ])


def _select_logistic_C(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    label: str,
) -> float:
    """Select the best LogisticRegression C via TimeSeriesSplit CV on the training set.

    Evaluates each C in _LR_C_VALUES using 5-fold time-series CV.
    Selection criterion: mean accuracy across folds.

    Parameters
    ----------
    X_train : feature matrix (training split only — never touches test set)
    y_train : binary target vector (0=down, 1=up)
    label   : pool label for log messages ("standard" or "fallback")

    Returns the selected C value.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    best_C, best_score = _LR_C_VALUES[0], -float("inf")
    scores_by_C = {}

    for C in _LR_C_VALUES:
        fold_scores = []
        for fold_train_idx, fold_val_idx in tscv.split(X_train):
            X_f_train = X_train.iloc[fold_train_idx]
            y_f_train = y_train.iloc[fold_train_idx]
            X_f_val   = X_train.iloc[fold_val_idx]
            y_f_val   = y_train.iloc[fold_val_idx]

            pipe = _build_logistic(C)
            pipe.fit(X_f_train, y_f_train)
            acc = accuracy_score(y_f_val, pipe.predict(X_f_val))
            fold_scores.append(acc)

        mean_score = float(np.mean(fold_scores))
        scores_by_C[C] = mean_score
        if mean_score > best_score:
            best_score, best_C = mean_score, C

    scores_str = "  ".join(
        f"C={c}: {s:.4f}" for c, s in scores_by_C.items()
    )
    logger.info(
        f"[{label}] LogisticRegression C CV (TimeSeriesSplit n=5, metric=accuracy):  {scores_str}"
    )
    logger.info(
        f"[{label}] Selected C={best_C}  "
        f"(CV mean accuracy={best_score:.4f})"
    )
    return best_C


def _build_rf() -> RandomForestClassifier:
    """RandomForestClassifier — tree-based, scale-invariant, no scaler needed."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=15,
        max_features=0.7,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def _build_gbr() -> GradientBoostingClassifier:
    """GradientBoostingClassifier — tree-based, scale-invariant, no scaler needed."""
    return GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=2,
        min_samples_leaf=20,
        min_samples_split=40,
        subsample=0.7,
        max_features=0.8,
        random_state=42,
    )


def _build_lgbm():
    """LightGBM classifier — tree-based, scale-invariant, no scaler needed.

    Returns None if lightgbm is not installed (ImportError handled in caller).
    Hyperparameters chosen to balance capacity and regularisation:
      - n_estimators=500 with low learning_rate=0.03 (slow learning, less overfit)
      - num_leaves=15 (shallow trees — reduces overfitting on ~25k rows)
      - min_child_samples=50 (minimum leaf size — key regulariser for small pools)
      - colsample_bytree=0.8, subsample=0.7 (stochastic, adds regularisation)
    """
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=50,
            colsample_bytree=0.8,
            subsample=0.7,
            is_unbalance=True,
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
    on usable rows only. Rows with exact zero Target are also dropped —
    they are ambiguous for binary up/down classification.
    Rows are already date-sorted by load_processed.
    """
    df = df.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)
    df = df[df[TARGET_COL] != 0].reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    return df.iloc[:split_idx], df.iloc[split_idx:]


# ── Internal training helper ──────────────────────────────────────────────────

def _train_pooled(
    tickers: list[str],
    feature_cols: list[str],
    model_filename: str,
    label: str,
) -> str:
    """Pool data from `tickers`, train on `feature_cols`, save to `model_filename`.

    Per-ticker temporal split is applied before pooling: each ticker's own
    final 20% of rows is reserved for the test set. Rows with zero Target are
    dropped before splitting (ambiguous for binary classification).

    Binary target: y = 1 if next-day return > 0 else 0.

    Four models are trained and evaluated on the held-out test set:
      - LogisticRegression (with StandardScaler + C selected via TimeSeriesSplit CV)
      - RandomForestClassifier   (scale-invariant tree model — no scaler)
      - GradientBoostingClassifier  (scale-invariant tree model — no scaler)
      - LGBMClassifier  (scale-invariant — skipped if not installed)

    The model with the highest test accuracy is saved as the .pkl file.
    After selection, the full classification_report is logged.

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
    X_test  = test_df[feature_cols]

    # Derive binary target: 1 = next-day price up, 0 = next-day price down.
    # Zero-return rows were already dropped in _split_ticker.
    y_train = (train_df[TARGET_COL] > 0).astype(int)
    y_test  = (test_df[TARGET_COL] > 0).astype(int)

    logger.info(
        f"[{label}] Binary target distribution — train: "
        f"{int(y_train.sum())} up / {int((y_train == 0).sum())} down  |  "
        f"test: {int(y_test.sum())} up / {int((y_test == 0).sum())} down"
    )

    # ── Select LogisticRegression C via TimeSeriesSplit CV (training set only) ─
    best_C = _select_logistic_C(X_train, y_train, label)

    # ── Build candidate list ──────────────────────────────────────────────────
    candidates = [
        (f"LogisticRegression(C={best_C}, balanced)", _build_logistic(best_C)),
        ("RandomForestClassifier",                      _build_rf()),
        ("GradientBoostingClassifier",                  _build_gbr()),
    ]

    lgbm_model = _build_lgbm()
    if lgbm_model is not None:
        candidates.append(("LGBMClassifier", lgbm_model))
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
    # GradientBoostingClassifier does not accept class_weight in its constructor;
    # balanced sample weights are passed directly to .fit() instead.
    gbr_sample_weight = compute_sample_weight("balanced", y_train)

    best_name, best_model, best_acc = None, None, -float("inf")
    results = []
    for name, model in candidates:
        if isinstance(model, GradientBoostingClassifier):
            model.fit(X_train, y_train, sample_weight=gbr_sample_weight)
        else:
            model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append((name, model, acc, y_pred))
        if acc > best_acc:
            best_acc, best_name, best_model = acc, name, model

    # Log all candidates, marking the winner.
    for name, _, acc, y_pred in results:
        f1    = f1_score(y_test, y_pred, average="binary", zero_division=0)
        check = "✓" if name == best_name else " "
        logger.info(
            f"[{label}] Model: {name} | "
            f"Accuracy: {acc * 100:.1f}% | F1: {f1:.3f} | Selected: {check}"
        )

    logger.info(
        f"[{label}] Best model: {best_name}  (test accuracy={best_acc * 100:.2f}%) → saving"
    )

    # ── Full classification report for the winner ─────────────────────────────
    best_preds = best_model.predict(X_test)
    logger.info(
        f"[{label}] Classification report ({best_name}):\n"
        + classification_report(y_test, best_preds, target_names=["DOWN (0)", "UP (1)"])
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
    df = df[df[TARGET_COL] != 0].reset_index(drop=True)

    if len(df) < 50:
        raise ValueError(
            f"Only {len(df)} clean rows for '{ticker}' — not enough to train reliably."
        )

    y = (df[TARGET_COL] > 0).astype(int)
    pipeline = _build_gbr()
    pipeline.fit(df[feature_cols], y)

    os.makedirs(TRAINED_DIR, exist_ok=True)
    model_path = os.path.join(TRAINED_DIR, f"model_{ticker}.pkl")
    joblib.dump(pipeline, model_path)

    logger.info(f"[{ticker}] Saved → {model_path}  ({len(df)} rows, {schema} schema)")
    return model_path


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the pooled trading models covering all 31 tickers."
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
