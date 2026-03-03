"""
etl.py — Extract, Transform, Load pipeline for SimFin financial data.

Usage:
    python etl/etl.py --ticker AAPL
"""

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def fetch_data(ticker: str) -> pd.DataFrame:
    """Download raw financial data for a given ticker from SimFin."""
    # TODO: implement using PySimFin wrapper or simfin library
    raise NotImplementedError("fetch_data not yet implemented")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data: handle missing values, fix dtypes, sort by date."""
    # TODO: implement cleaning logic
    raise NotImplementedError("clean_data not yet implemented")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features used by the ML model."""
    # TODO: implement feature engineering
    raise NotImplementedError("engineer_features not yet implemented")


def save_processed(df: pd.DataFrame, ticker: str) -> str:
    """Save the processed DataFrame to data/processed/<ticker>.csv."""
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved processed data to {out_path}")
    return out_path


def run(ticker: str) -> pd.DataFrame:
    print(f"Running ETL for {ticker}...")
    raw = fetch_data(ticker)
    cleaned = clean_data(raw)
    featured = engineer_features(cleaned)
    save_processed(featured, ticker)
    return featured


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL pipeline for a stock ticker.")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol, e.g. AAPL")
    args = parser.parse_args()
    run(args.ticker)
