"""
calibration.py — Calibration plot generator for the pooled trading models.

Loads model_pooled.pkl and model_pooled_fallback.pkl, reconstructs the
per-ticker 80/20 temporal test sets (same logic as train.py), and generates:

  model/trained/calibration_standard.png   — decile calibration chart (25 tickers)
  model/trained/calibration_fallback.png   — decile calibration chart (5 tickers)

A well-calibrated regression model shows a monotonically increasing bar chart:
lower-decile predictions should correspond to lower actual returns and
upper-decile predictions to higher actual returns. Flatness or inversion
indicates that the model's predicted magnitudes carry no information about
actual magnitudes.

Also prints a summary table per decile:
  Decile | Mean Pred Return | Mean Actual Return | Direction Accuracy

Usage:
    python model/calibration.py
"""

import glob
import logging
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for script execution
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.strategy import (
    STANDARD_FEATURE_COLS,
    FALLBACK_FEATURE_COLS,
    FALLBACK_TICKERS,
    is_fallback_ticker,
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
N_DECILES     = 10


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_model(filename: str):
    path = os.path.join(TRAINED_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Run:  python model/train.py --all"
        )
    return joblib.load(path)


def _discover_tickers() -> list[str]:
    csv_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*.csv")))
    return [os.path.basename(f).replace(".csv", "") for f in csv_files]


def _build_test_set(tickers: list[str], feature_cols: list[str]) -> pd.DataFrame:
    """Reconstruct the pooled test set using the same 80/20 temporal split as train.py.

    For each ticker: drop NaN rows on the feature+target subset, then take
    the final 20% of rows as the test set. Concatenates across all tickers.
    """
    test_frames = []
    for ticker in tickers:
        path = os.path.join(PROCESSED_DIR, f"{ticker}.csv")
        if not os.path.exists(path):
            logger.warning(f"[{ticker}] Processed CSV not found — skipping.")
            continue
        df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date").reset_index(drop=True)
        df = df.dropna(subset=feature_cols + [TARGET_COL]).reset_index(drop=True)
        split_idx = int(len(df) * 0.8)
        test_frames.append(df.iloc[split_idx:])

    if not test_frames:
        raise RuntimeError("No test data could be reconstructed. Check data/processed/.")

    return pd.concat(test_frames, ignore_index=True)


def _calibration_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_deciles: int = N_DECILES,
) -> pd.DataFrame:
    """Bin predictions into deciles; compute mean pred, mean actual, dir accuracy per bin."""
    df = pd.DataFrame({"pred": y_pred, "actual": y_true})
    df["decile"] = pd.qcut(df["pred"], q=n_deciles, labels=False, duplicates="drop")

    rows = []
    for d in sorted(df["decile"].dropna().unique()):
        subset = df[df["decile"] == d]
        dir_acc = np.mean(np.sign(subset["pred"].values) == np.sign(subset["actual"].values))
        rows.append({
            "Decile": int(d) + 1,
            "N": len(subset),
            "Mean Pred Return": subset["pred"].mean(),
            "Mean Actual Return": subset["actual"].mean(),
            "Direction Accuracy": dir_acc,
        })

    return pd.DataFrame(rows)


def _plot_calibration(
    table: pd.DataFrame,
    title: str,
    save_path: str,
) -> None:
    """Bar chart: mean actual return per decile, overlaid with mean predicted return line."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    deciles = table["Decile"].values
    actual  = table["Mean Actual Return"].values
    pred    = table["Mean Pred Return"].values

    # Bar: actual return per decile.
    colors = ["#00c805" if v >= 0 else "#ff4444" for v in actual]
    ax.bar(deciles, actual, color=colors, alpha=0.85, label="Mean Actual Return")

    # Line: predicted return per decile.
    ax.plot(deciles, pred, color="#80eaff", linewidth=1.8, marker="o",
            markersize=4, label="Mean Predicted Return")

    ax.axhline(0, color="#555555", linewidth=0.8, linestyle="--")

    ax.set_title(title, color="#f0f0f0", fontsize=13, pad=12)
    ax.set_xlabel("Prediction Decile (1 = most negative prediction → 10 = most positive)",
                  color="#f0f0f0", fontsize=10)
    ax.set_ylabel("Mean Return", color="#f0f0f0")
    ax.set_xticks(deciles)
    ax.tick_params(colors="#f0f0f0")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    ax.legend(facecolor="#1a1a1a", labelcolor="#f0f0f0")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Calibration plot saved → {save_path}")


def _print_table(label: str, table: pd.DataFrame) -> None:
    """Print the decile summary table to the log."""
    logger.info(f"\n[{label}] Calibration summary table:")
    header = (
        f"  {'Decile':>6}  {'N':>5}  {'Mean Pred':>12}  "
        f"{'Mean Actual':>12}  {'Dir Acc':>8}"
    )
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))
    for _, row in table.iterrows():
        logger.info(
            f"  {int(row['Decile']):>6}  {int(row['N']):>5}  "
            f"{row['Mean Pred Return']:>+12.6f}  "
            f"{row['Mean Actual Return']:>+12.6f}  "
            f"{row['Direction Accuracy']:>8.4f}"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def run_calibration() -> None:
    all_tickers = _discover_tickers()
    standard_tickers = [t for t in all_tickers if not is_fallback_ticker(t)]
    fallback_tickers  = [t for t in all_tickers if is_fallback_ticker(t)]

    os.makedirs(TRAINED_DIR, exist_ok=True)

    for label, tickers, feature_cols, model_file, plot_file in [
        (
            "standard",
            standard_tickers,
            STANDARD_FEATURE_COLS,
            "model_pooled.pkl",
            "calibration_standard.png",
        ),
        (
            "fallback",
            fallback_tickers,
            FALLBACK_FEATURE_COLS,
            "model_pooled_fallback.pkl",
            "calibration_fallback.png",
        ),
    ]:
        logger.info(f"[{label}] Running calibration for {len(tickers)} tickers…")

        model = _load_model(model_file)
        test_df = _build_test_set(tickers, feature_cols)

        X_test = test_df[feature_cols]
        y_test = test_df[TARGET_COL].values
        y_pred = model.predict(X_test)

        logger.info(
            f"[{label}] Test set: {len(test_df):,} rows  |  "
            f"Overall direction accuracy: "
            f"{np.mean(np.sign(y_pred) == np.sign(y_test)):.4f}"
        )

        table = _calibration_table(y_test, y_pred)
        _print_table(label, table)

        save_path = os.path.join(TRAINED_DIR, plot_file)
        _plot_calibration(
            table,
            title=(
                f"Calibration — {label.capitalize()} Pool  "
                f"({len(tickers)} tickers, {len(feature_cols)} features)\n"
                "Mean actual return per prediction decile"
            ),
            save_path=save_path,
        )


if __name__ == "__main__":
    try:
        run_calibration()
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
