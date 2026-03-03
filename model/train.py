"""
train.py — Train a predictive ML model for a given stock ticker.

Usage:
    python model/train.py --ticker AAPL
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

# TODO: import and configure your chosen model


TRAINED_DIR = os.path.join(os.path.dirname(__file__), "trained")


def load_processed(ticker: str) -> pd.DataFrame:
    """Load the processed CSV produced by etl.py."""
    path = os.path.join(
        os.path.dirname(__file__), "..", "data", "processed", f"{ticker}.csv"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No processed data found for {ticker}. Run etl/etl.py first."
        )
    return pd.read_csv(path, parse_dates=["date"])


def build_features_and_target(df: pd.DataFrame):
    """Split DataFrame into feature matrix X and target vector y."""
    # TODO: define which columns are features and what the target is
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def build_pipeline() -> Pipeline:
    """Define the sklearn Pipeline (preprocessing + model)."""
    # TODO: replace with your actual pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])


def train(ticker: str) -> str:
    """Train the model and save it to model/trained/model_<ticker>.pkl."""
    df = load_processed(ticker)
    X, y = build_features_and_target(df)

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    os.makedirs(TRAINED_DIR, exist_ok=True)
    model_path = os.path.join(TRAINED_DIR, f"model_{ticker}.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model for a stock ticker.")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol, e.g. AAPL")
    args = parser.parse_args()
    train(args.ticker)
