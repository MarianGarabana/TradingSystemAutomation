"""
etl.py — Extract, Transform, Load pipeline for SimFin financial data.

Usage:
    python etl/etl.py --ticker AAPL
    python etl/etl.py --all

Improvements over baseline:
- Uses Adj. Close instead of Close for all price-based calculations,
  which accounts for stock splits and dividends for historical accuracy.
- Handles extreme price errors (|return| > 50%) by nullifying the price and
  forward-filling, preserving time-series continuity. Return and Volume_Change
  are then winsorized at the 1st/99th percentile to bound outlier influence.
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

    # Detect price errors: daily returns above 50% are almost certainly data errors
    # or unadjusted stock splits. Instead of dropping the row (which creates a phantom
    # multi-day return in the next pct_change()), we set Adj. Close to NaN and forward-fill.
    # This preserves time-series continuity — all rows stay in the DataFrame with no gaps,
    # so Return, Return_Lag1, Return_Lag2 are never corrupted by artificial multi-day spans.
    daily_return = df['Adj. Close'].pct_change()
    mask = daily_return.abs() > 0.5
    n_errors = int(mask.sum())
    if n_errors > 0:
        logger.warning(
            f"Detected {n_errors} price error(s) with |return| > 50% — "
            "nullified and forward-filled to preserve time-series continuity."
        )
        df.loc[mask, 'Adj. Close'] = np.nan
        df['Adj. Close'] = df['Adj. Close'].ffill()

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

    # Winsorize Return and Volume_Change at the 1st / 99th percentile.
    # Even after forward-filling price errors, earnings-day spikes and thin-volume
    # days can produce extreme percentage changes. Clipping bounds their influence
    # on linear models and lag features while keeping all rows in the dataset.
    for col in ('Return', 'Volume_Change'):
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

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

    # ── Volatility-normalised features for cross-stock comparability ──────────
    # Arithmetic returns (Return above) differ in scale across stocks:
    # a low-vol stock like KO swings ±0.5%/day while TSLA swings ±3%/day.
    # Training a pooled model on raw returns lets high-vol stocks dominate.
    # The three features below put every ticker on the same risk-adjusted scale.

    # Log return: log(P_t / P_{t-1}).  Additive over time, more Gaussian than
    # arithmetic returns, and theoretically correct for multi-period compounding.
    df['Log_Return'] = np.log(df['Adj. Close'] / df['Adj. Close'].shift(1))

    # 20-day rolling realised volatility: standard deviation of daily log returns
    # over the past month.  Captures the current volatility regime for each stock.
    df['Volatility_20'] = df['Log_Return'].rolling(20).std()

    # Volatility-normalised return: log return ÷ recent volatility.
    # Analogous to a daily Z-score — a value of +1 means "rose one standard
    # deviation today".  Directly comparable across KO, TSLA, NVDA, etc.
    df['Return_norm'] = df['Log_Return'] / df['Volatility_20']

    # Lagged normalised returns: give the model memory of recent risk-adjusted moves
    df['Return_norm_Lag1'] = df['Return_norm'].shift(1)
    df['Return_norm_Lag2'] = df['Return_norm'].shift(2)

    # Drop only rows where price-derived columns are NaN (introduced by rolling windows
    # and shifts above).  Fundamental columns (Gross_Margin, etc.) are intentionally
    # excluded here: they can legitimately be NaN for tickers whose revenue structure
    # makes certain ratios undefined (e.g. Visa/Mastercard have no "Gross Profit" line).
    # Those NaN rows are excluded from ML training in train.py via its own dropna(subset=).
    price_required = [
        "Return", "MA5", "MA20", "Volume_Change", "Market_Cap", "Target",
        "RSI", "MACD", "MACD_Signal", "BB_Upper", "BB_Lower",
        "Return_Lag1", "Return_Lag2",
        "Log_Return", "Volatility_20", "Return_norm", "Return_norm_Lag1", "Return_norm_Lag2",
    ]
    df = df.dropna(subset=price_required).reset_index(drop=True)
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


ALL_TICKERS = [
    # Standard pool (26) — price + volatility + fundamentals
    "AAPL", "ABBV", "ADBE", "AMD", "AMZN", "AVGO", "COST", "CRM", "DIS", "GOOG", "INTC",
    "JNJ", "KO", "MCD", "META", "MSFT", "NFLX", "NVDA", "ORCL", "PEP", "PFE",
    "PLTR", "QCOM", "TSLA", "UNH", "WMT",
    # Fallback pool (5) — price + volatility only (no fundamentals)
    "BAC", "GS", "JPM", "MA", "V",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL pipeline for a stock ticker.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ticker", help="Single stock ticker symbol, e.g. AAPL")
    group.add_argument("--all", action="store_true", help="Run ETL for all 31 project tickers")
    args = parser.parse_args()

    tickers = ALL_TICKERS if args.all else [args.ticker]
    failed = []
    for t in tickers:
        result = run(t)
        if result is None:
            failed.append(t)

    if failed:
        logger.error(f"ETL failed for: {failed}")
        sys.exit(1)
