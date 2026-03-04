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
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "us-shareprices-daily.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")
    df = pd.read_csv(raw_path, sep=';')
    result = df[df['Ticker'] == ticker]
    if result.empty:
        raise ValueError(f"Ticker '{ticker}' not found in the dataset.")
    return result



def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Dividend'] = df['Dividend'].fillna(0)
    df = df.drop(columns=['SimFinId'])
    return df



def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 20:
        raise ValueError(f"Not enough rows ({len(df)}) to compute a 20-day moving average.")
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna().reset_index(drop=True)
    return df


def save_processed(df: pd.DataFrame, ticker: str) -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved processed data to {out_path}")
    return out_path


def run(ticker: str) -> pd.DataFrame:
    print(f"Running ETL for {ticker}...")
    try:
        raw = fetch_data(ticker)
        cleaned = clean_data(raw)
        featured = engineer_features(cleaned)
        save_processed(featured, ticker)
        return featured
    except (FileNotFoundError, ValueError) as e:
        print(f"ETL failed for {ticker}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL pipeline for a stock ticker.")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol, e.g. AAPL")
    args = parser.parse_args()
    run(args.ticker)
