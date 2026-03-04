"""
etl.py — Extract, Transform, Load pipeline for SimFin financial data.

Usage:
    python etl/etl.py --ticker AAPL

Improvements over baseline:
- Uses Adj. Close instead of Close for all price-based calculations,
  which accounts for stock splits and dividends for historical accuracy.
- Filters out extreme daily returns (>50%) which are likely data errors.
- Uses Python's logging module instead of print() for structured output.
- Prints a data quality summary after each run.
"""

import argparse
import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Configure logging: timestamps + log level + message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def fetch_data(ticker: str) -> pd.DataFrame:
    raw_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "us-shareprices-daily.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")
    df = pd.read_csv(raw_path, sep=';')
    result = df[df['Ticker'] == ticker]
    if result.empty:
        raise ValueError(f"Ticker '{ticker}' not found in the dataset.")
    logger.info(f"Fetched {len(result)} rows for {ticker}.")
    return result


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    df['Dividend'] = df['Dividend'].fillna(0)
    df = df.drop(columns=['SimFinId'])

    # Remove outliers: daily returns above 50% are almost certainly data errors
    # or unadjusted stock splits — keeping them would distort all rolling features.
    raw_len = len(df)
    daily_return = df['Adj. Close'].pct_change()
    df = df[daily_return.abs() < 0.5].reset_index(drop=True)
    removed = raw_len - len(df)
    if removed > 0:
        logger.warning(f"Removed {removed} rows with extreme daily returns (>50%).")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 20:
        raise ValueError(f"Not enough rows ({len(df)}) to compute a 20-day moving average.")
    df = df.copy()

    # Use Adj. Close instead of Close for all price-based features.
    # Adj. Close corrects for historical stock splits and dividend payouts,
    # making returns and moving averages accurate across the full time range.
    price = df['Adj. Close']

    # Daily percentage return: how much the adjusted price moved vs yesterday
    df['Return'] = price.pct_change()

    # Moving averages: smooth out noise to reveal the underlying trend
    df['MA5'] = price.rolling(5).mean()   # short-term trend (1 week)
    df['MA20'] = price.rolling(20).mean() # medium-term trend (1 month)

    # Volume change: detects unusual trading activity
    df['Volume_Change'] = df['Volume'].pct_change()

    # Target: 1 if tomorrow's adjusted price is higher than today's, 0 otherwise
    # This is what the ML model will predict (binary classification)
    df['Target'] = (price.shift(-1) > price).astype(int)

    # RSI: momentum indicator — >70 means overbought (likely to fall), <30 means oversold (likely to rise)
    delta = price.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    # MACD: trend indicator — when MACD crosses above its signal line, bullish signal; below, bearish
    ema12 = price.ewm(span=12, adjust=False).mean()
    ema26 = price.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands: volatility indicator — price near upper band = overbought, near lower band = oversold
    std20 = price.rolling(20).std()
    df['BB_Upper'] = df['MA20'] + 2 * std20
    df['BB_Lower'] = df['MA20'] - 2 * std20

    # Lag features: give the model memory of recent price movements
    df['Return_Lag1'] = df['Return'].shift(1)  # yesterday's return
    df['Return_Lag2'] = df['Return'].shift(2)  # 2 days ago

    # Drop rows with NaN introduced by rolling windows and shifts
    df = df.dropna().reset_index(drop=True)
    return df


def save_processed(df: pd.DataFrame, ticker: str) -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved processed data to {out_path}")
    return out_path


def run(ticker: str) -> pd.DataFrame:
    logger.info(f"Running ETL for {ticker}...")
    try:
        raw = fetch_data(ticker)
        cleaned = clean_data(raw)
        featured = engineer_features(cleaned)
        save_processed(featured, ticker)

        # Data quality summary: quick sanity check after processing
        target_counts = featured['Target'].value_counts()
        logger.info(
            f"[{ticker}] Done — "
            f"{len(featured)} rows | "
            f"{featured['Date'].min().date()} → {featured['Date'].max().date()} | "
            f"Target: {target_counts.get(1, 0)} up / {target_counts.get(0, 0)} down"
        )
        return featured
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"ETL failed for {ticker}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL pipeline for a stock ticker.")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol, e.g. AAPL")
    args = parser.parse_args()
    run(args.ticker)
