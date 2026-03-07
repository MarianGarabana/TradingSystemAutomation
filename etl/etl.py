"""
etl.py — Extract, Transform, Load pipeline for SimFin financial data.

Usage:
    python etl/etl.py --ticker AAPL

Improvements over baseline:
- Uses Adj. Close instead of Close for all price-based calculations,
  which accounts for stock splits and dividends for historical accuracy.
- Filters out extreme daily returns (>50%) which are likely data errors.
- Uses Python's logging module instead of print() for structured output.
- Enriches price data with quarterly fundamental ratios (point-in-time merge,
  no look-ahead bias) from income, balance sheet, and cash flow statements.
- Prints a data quality summary after each run.
"""

import argparse
import logging
import os
import sys

import numpy as np
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

    # Market Cap: total market value = price × shares outstanding.
    # Changes daily with price — a dynamic proxy for company size.
    df['Market_Cap'] = price * df['Shares Outstanding']

    # Target: next-day return (continuous regression target).
    # pct_change() gives today's return; .shift(-1) moves tomorrow's return to today's row.
    # Result: for each row, Target = (price_tomorrow - price_today) / price_today
    # Positive = price rises next day, negative = price falls.
    df['Target'] = price.pct_change().shift(-1)

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


def fetch_fundamentals(ticker: str) -> pd.DataFrame | None:
    """Load and combine quarterly fundamental data for a single ticker.

    Returns None if the quarterly CSV files are not present in data/raw/,
    so the ETL can fall back to price-only mode.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

    income_path   = os.path.join(data_dir, "us-income-quarterly.csv")
    balance_path  = os.path.join(data_dir, "us-balance-quarterly.csv")
    cashflow_path = os.path.join(data_dir, "us-cashflow-quarterly.csv")

    # Check if all three quarterly files exist before trying to read them.
    # If any are missing (e.g. not yet downloaded from SimFin), we skip
    # the fundamentals step and continue with price data only.
    if not all(os.path.exists(p) for p in [income_path, balance_path, cashflow_path]):
        logger.warning("Quarterly fundamental files not found — running in price-only mode.")
        return None

    income = pd.read_csv(
        income_path, sep=';',
        usecols=['Ticker', 'Publish Date', 'Revenue', 'Gross Profit',
                 'Operating Income (Loss)', 'Net Income'],
    )
    balance = pd.read_csv(
        balance_path, sep=';',
        usecols=['Ticker', 'Publish Date', 'Total Assets', 'Total Liabilities', 'Total Equity'],
    )
    cashflow = pd.read_csv(
        cashflow_path, sep=';',
        usecols=['Ticker', 'Publish Date', 'Net Cash from Operating Activities'],
    )

    # Filter to the requested ticker only
    income   = income[income['Ticker']   == ticker].copy()
    balance  = balance[balance['Ticker'] == ticker].copy()
    cashflow = cashflow[cashflow['Ticker'] == ticker].copy()

    # If this ticker has no fundamental records at all, skip gracefully
    # instead of crashing — some tickers in the price file may not have
    # corresponding quarterly statements.
    if income.empty:
        logger.warning(f"No fundamental data found for '{ticker}' — running in price-only mode.")
        return None

    return _compute_fundamental_features(income, balance, cashflow)


def _compute_fundamental_features(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    cashflow: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive normalised financial ratios from raw quarterly statements.

    Dimensionless ratios are comparable across companies of different sizes
    and do not need rescaling before being passed to tree-based models.
    """
    # --- Income ratios ---
    inc = income.copy()
    inc['Gross_Margin']     = inc['Gross Profit'] / inc['Revenue']
    inc['Operating_Margin'] = inc['Operating Income (Loss)'] / inc['Revenue']
    inc['Net_Margin']       = inc['Net Income'] / inc['Revenue']
    inc = inc[['Ticker', 'Publish Date', 'Gross_Margin', 'Operating_Margin', 'Net_Margin']]

    # --- Balance ratio ---
    bal = balance.copy()
    # Debt-to-Equity: how much of the company is financed by debt vs equity.
    bal['Debt_to_Equity'] = bal['Total Liabilities'] / bal['Total Equity']
    bal_ratios = bal[['Ticker', 'Publish Date', 'Debt_to_Equity']]

    # --- Cash flow ratio (normalised by total assets) ---
    # Harder to manipulate than net income; reliable signal of cash generation.
    cf = cashflow.merge(
        bal[['Ticker', 'Publish Date', 'Total Assets']],
        on=['Ticker', 'Publish Date'], how='left',
    )
    cf['Operating_CF_Ratio'] = cf['Net Cash from Operating Activities'] / cf['Total Assets']
    cf_ratios = cf[['Ticker', 'Publish Date', 'Operating_CF_Ratio']]

    # --- Combine ---
    fund = inc.merge(bal_ratios, on=['Ticker', 'Publish Date'], how='outer')
    fund = fund.merge(cf_ratios,  on=['Ticker', 'Publish Date'], how='outer')
    fund['Publish Date'] = pd.to_datetime(fund['Publish Date'])
    fund = fund.replace([np.inf, -np.inf], np.nan)
    fund = fund.sort_values('Publish Date').reset_index(drop=True)
    return fund


def merge_fundamentals(price_df: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Point-in-time merge: attach the most recently *published* quarterly snapshot
    to each daily price row.

    Using direction='backward' ensures that on any given trading day we only
    attach data that was publicly available — zero look-ahead bias.
    """
    price_df = price_df.sort_values('Date')
    fundamentals = fundamentals.drop(columns='Ticker', errors='ignore').sort_values('Publish Date')

    return pd.merge_asof(
        price_df, fundamentals,
        left_on='Date',
        right_on='Publish Date',
        direction='backward',
    )


def save_processed(df: pd.DataFrame, ticker: str) -> str:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}.csv")
    df.to_csv(out_path, index=False)
    logger.info(f"Saved processed data to {out_path}")
    return out_path


def run(ticker: str) -> pd.DataFrame | None:
    logger.info(f"Running ETL for {ticker}...")
    try:
        raw          = fetch_data(ticker)
        cleaned      = clean_data(raw)
        fundamentals = fetch_fundamentals(ticker)

        # Only merge fundamentals if they were successfully loaded.
        # If fundamentals is None (files missing or ticker not found),
        # we skip the merge and use price data only.
        if fundamentals is not None:
            enriched = merge_fundamentals(cleaned, fundamentals)
        else:
            enriched = cleaned

        featured = engineer_features(enriched)
        save_processed(featured, ticker)

        # Data quality summary: quick sanity check after processing
        t = featured['Target']
        logger.info(
            f"[{ticker}] Done — "
            f"{len(featured)} rows | "
            f"{featured['Date'].min().date()} → {featured['Date'].max().date()} | "
            f"Target (next-day return): mean={t.mean():.4f}  std={t.std():.4f}  "
            f"range=[{t.min():.4f}, {t.max():.4f}]"
        )
        return featured
    except (FileNotFoundError, ValueError) as e:
        # Log the error but return None instead of crashing the whole process,
        # so --all mode can continue with the remaining tickers.
        logger.error(f"ETL failed for {ticker}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL pipeline for a stock ticker.")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol, e.g. AAPL")
    args = parser.parse_args()

    result = run(args.ticker)
    if result is None:
        sys.exit(1)
