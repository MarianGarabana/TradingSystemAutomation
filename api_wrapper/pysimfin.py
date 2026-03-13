"""
pysimfin.py — A lightweight OOP wrapper around the SimFin API v3.

No frameworks — plain Python with requests.

All endpoints use the SimFin v3 REST API (https://backend.simfin.com/api/v3).
Authentication is done via the Authorization header: "api-key <key>".
Rate limit on the free tier: 2 requests per second — enforced here with a
minimum 0.5 s gap between calls tracked via _last_request_time.

Example usage:
    from api_wrapper.pysimfin import PySimFin
    client = PySimFin(api_key="your_key")
    df_prices = client.get_share_prices("AAPL", "2024-01-01", "2024-01-31")
    df_income = client.get_income_statements("AAPL", period="quarterly")
    df_fin    = client.get_financial_statement("AAPL", "2023-01-01", "2024-01-01")
"""

import os
import time
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

SIMFIN_BASE_URL = "https://backend.simfin.com/api/v3"

# Minimum seconds between requests to respect the 2 req/sec free-tier limit.
# 0.55 s gives a small safety margin over the theoretical 0.5 s minimum.
_MIN_REQUEST_INTERVAL = 0.55


class PySimFin:
    """Wrapper for the SimFin v3 REST API.

    Provides methods to fetch share price data and financial statements
    for publicly traded companies. All responses are returned as
    pandas DataFrames to make downstream analysis straightforward.

    Rate limiting is handled automatically: a timestamp of the last
    request is tracked, and every call sleeps just long enough to
    maintain at most 2 requests per second.
    """

    def __init__(self, api_key: Optional[str] = None):
        # API key can be passed explicitly or loaded from the SIMFIN_API_KEY
        # environment variable / .env file via python-dotenv.
        self.api_key = api_key or os.getenv("SIMFIN_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SimFin API key not found. Set SIMFIN_API_KEY in your .env file."
            )

        # reuse a single requests.Session for connection pooling — avoids
        # re-doing the TCP handshake on every call, which matters when fetching
        # data for 30 tickers in a loop.
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"api-key {self.api_key}"})

        # Track when the last HTTP request was sent so we can enforce the rate
        # limit without sleeping an unconditional 0.5 s on every call.
        self._last_request_time: float = 0.0

    # ── Private helpers ────────────────────────────────────────────────────────

    def _throttle(self) -> None:
        """Sleep if needed to respect the 2 req/sec rate limit.

        We calculate the elapsed time since the last request and only sleep
        the remaining fraction of _MIN_REQUEST_INTERVAL — this way the method
        adds zero latency when the caller is slower than 2 req/sec themselves.
        """
        elapsed = time.time() - self._last_request_time
        if elapsed < _MIN_REQUEST_INTERVAL:
            time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Make a rate-limited GET request to the SimFin API.

        Handles the three most common failure modes:
        - 404: ticker / resource does not exist in SimFin
        - 429: we exceeded the rate limit (shouldn't happen with _throttle,
               but retry once after a full second just in case)
        - other 4xx/5xx: re-raised as requests.HTTPError for the caller

        Returns the parsed JSON response (dict or list).
        """
        self._throttle()
        url = f"{SIMFIN_BASE_URL}/{endpoint}"

        response = self.session.get(url, params=params or {})
        self._last_request_time = time.time()

        if response.status_code == 429:
            # Rate limit hit — wait a full second and retry once before giving up.
            time.sleep(1.0)
            response = self.session.get(url, params=params or {})
            self._last_request_time = time.time()

        if response.status_code == 404:
            # The resource was not found; give a descriptive message rather than
            # a generic HTTPError, so callers know whether to retry or skip.
            raise ValueError(
                f"SimFin API returned 404 for endpoint '{endpoint}' "
                f"with params {params}. Check that the ticker exists."
            )

        # Raise for any other 4xx / 5xx codes (e.g. 401 bad API key, 500 server error).
        response.raise_for_status()
        return response.json()

    # ── Public API methods ─────────────────────────────────────────────────────

    def get_companies(self) -> pd.DataFrame:
        """Return a DataFrame of all companies available in SimFin.

        Each row is one company with columns:
            id, name, ticker, sectorCode, industryName, sectorName, isin

        This is useful for building a ticker → SimFin-ID lookup table if you
        ever need to call ID-based endpoints directly.

        Endpoint: GET /companies/list
        """
        data = self._get("companies/list")

        # The endpoint returns a plain JSON array; pd.DataFrame can ingest it
        # directly without any reshaping.
        df = pd.DataFrame(data)
        return df

    def get_share_prices(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch daily share price data for *ticker* between *start* and *end*.

        Parameters
        ----------
        ticker : str
            Company ticker symbol, e.g. "AAPL".
        start : str
            Start date in YYYY-MM-DD format (inclusive).
        end : str
            End date in YYYY-MM-DD format (inclusive).

        Returns
        -------
        pd.DataFrame
            One row per trading day with columns:
                Date, Open, High, Low, Close, Adj. Close, Volume, Dividend

            Column names mirror the SimFin bulk-download CSV so that the ETL
            and feature-engineering code can work with either data source
            without modification.

        Raises
        ------
        ValueError
            If the ticker is not found in SimFin (404 response).

        Endpoint: GET /companies/prices/compact
        """
        params = {
            "ticker": ticker,
            "start": start,
            "end": end,
        }
        raw = self._get("companies/prices/compact", params)

        # The API returns 200 with an empty list both when the ticker is unknown
        # AND when the requested date range contains no trading days (e.g. a
        # weekend or a future period). Raise ValueError in both cases so the
        # caller can decide whether to fall back to static data or surface an
        # error to the user.
        if not raw:
            raise ValueError(
                f"No price data returned for ticker '{ticker}' between {start} and {end}. "
                "The ticker may not exist in SimFin, or the date range contains no "
                "trading days (weekend, holiday, or future date)."
            )

        payload = raw[0]

        # The API returns data in a compact "columns + rows" format to minimise
        # payload size — no field names repeated for every row.
        # columns:  ["Date", "Dividend Paid", "Common Shares Outstanding",
        #             "Last Closing Price", "Adjusted Closing Price",
        #             "Highest Price", "Lowest Price", "Opening Price",
        #             "Trading Volume"]
        columns = payload["columns"]
        rows = payload["data"]

        if not rows:
            return pd.DataFrame(
                columns=["Date", "Open", "High", "Low", "Close",
                         "Adj. Close", "Volume", "Dividend"]
            )

        df = pd.DataFrame(rows, columns=columns)

        # Rename to the same column names used in the bulk-download CSVs so
        # the rest of the pipeline (ETL, go_live.py) works unchanged.
        df = df.rename(columns={
            "Opening Price":          "Open",
            "Highest Price":          "High",
            "Lowest Price":           "Low",
            "Last Closing Price":     "Close",
            "Adjusted Closing Price": "Adj. Close",
            "Trading Volume":         "Volume",
            "Dividend Paid":          "Dividend",
        })

        # Rename "Common Shares Outstanding" to "Shares Outstanding" so the
        # ETL's engineer_features() can compute Market_Cap = price × shares
        # without any column-name changes in the calling code.
        df = df.rename(columns={"Common Shares Outstanding": "Shares Outstanding"})

        # Convert Date to datetime for downstream sorting and merging.
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        return df[["Date", "Open", "High", "Low", "Close", "Adj. Close",
                   "Volume", "Dividend", "Shares Outstanding"]]

    def get_financial_statement(
        self,
        ticker: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch quarterly financial statement data for *ticker*.

        Fetches income (PL), balance sheet (BS), and cash flow (CF) in three
        separate API calls and merges them into a single wide DataFrame keyed
        on (ticker, Report Date). This convenience method is what the ETL
        needs when enriching price data with fundamental ratios.

        Parameters
        ----------
        ticker : str
            Company ticker symbol, e.g. "AAPL".
        start : str
            Start date in YYYY-MM-DD format.
        end : str
            End date in YYYY-MM-DD format.

        Returns
        -------
        pd.DataFrame
            One row per fiscal quarter with all income, balance, and cash-flow
            columns. Key columns include:
                Report Date, Revenue, Gross Profit, Operating Income (Loss),
                Net Income (Common), Total Assets, Total Equity, Total Debt,
                Cash from Operating Activities

        Endpoint: GET /companies/statements/verbose
        """
        # We need all three statement types to compute the fundamental ratios
        # used as model features (Gross_Margin, Operating_Margin, Net_Margin,
        # Debt_to_Equity, Operating_CF_Ratio).
        statement_types = ["PL", "BS", "CF"]
        frames = []

        for stmt_type in statement_types:
            params = {
                "ticker": ticker,
                "statements": stmt_type,
                # Request all four quarters so we get every quarterly period.
                "period": "Q1,Q2,Q3,Q4",
                "start": start,
                "end": end,
            }
            raw = self._get("companies/statements/verbose", params)

            if not raw:
                continue

            # The response is a list; take the first element (our single ticker).
            payload = raw[0]
            statements = payload.get("statements", [])
            if not statements:
                continue

            # Each element in 'statements' corresponds to one statement type;
            # we requested only one so take index 0.
            data_rows = statements[0].get("data", [])
            if not data_rows:
                continue

            df_stmt = pd.DataFrame(data_rows)

            # Keep only the columns we need to avoid a very wide merge table.
            # Always keep the join key (Report Date) plus the relevant metrics.
            if stmt_type == "PL":
                keep = ["Report Date", "Revenue", "Gross Profit",
                        "Operating Income (Loss)", "Net Income (Common)"]
            elif stmt_type == "BS":
                keep = ["Report Date", "Total Assets", "Total Equity",
                        "Total Debt", "Total Liabilities & Equity"]
            else:  # CF
                keep = ["Report Date", "Cash from Operating Activities"]

            # Only retain columns that actually exist in the response — some
            # tickers may not report all fields.
            keep = [c for c in keep if c in df_stmt.columns]
            df_stmt = df_stmt[keep]
            frames.append(df_stmt)

        if not frames:
            return pd.DataFrame()

        # Merge all three statement frames on Report Date using an outer join
        # so we don't lose periods where one statement has data but another
        # doesn't (e.g. a company that only files PL but not CF).
        merged = frames[0]
        for df_next in frames[1:]:
            merged = merged.merge(df_next, on="Report Date", how="outer")

        merged["Report Date"] = pd.to_datetime(merged["Report Date"])
        merged["Ticker"] = ticker
        merged = merged.sort_values("Report Date").reset_index(drop=True)

        return merged

    # ── Individual statement methods (kept for completeness and direct access) ──

    def get_income_statements(
        self,
        ticker: str,
        period: str = "quarterly",
    ) -> pd.DataFrame:
        """Fetch full income statement (P&L) data for *ticker*.

        Parameters
        ----------
        ticker : str
            Company ticker symbol.
        period : str
            "quarterly" (default) or "annual".

        Returns
        -------
        pd.DataFrame
            One row per fiscal period with all income statement fields.

        Endpoint: GET /companies/statements/verbose (statements=PL)
        """
        # Map user-friendly period name to the SimFin API's period parameter.
        # Quarterly means all four quarters; annual only returns full-year rows.
        simfin_period = "Q1,Q2,Q3,Q4" if period == "quarterly" else "FY"

        params = {
            "ticker": ticker,
            "statements": "PL",
            "period": simfin_period,
        }
        raw = self._get("companies/statements/verbose", params)

        if not raw:
            return pd.DataFrame()

        data_rows = raw[0].get("statements", [{}])[0].get("data", [])
        df = pd.DataFrame(data_rows)
        if "Report Date" in df.columns:
            df["Report Date"] = pd.to_datetime(df["Report Date"])
        return df

    def get_balance_sheets(
        self,
        ticker: str,
        period: str = "quarterly",
    ) -> pd.DataFrame:
        """Fetch full balance sheet data for *ticker*.

        Parameters
        ----------
        ticker : str
            Company ticker symbol.
        period : str
            "quarterly" (default) or "annual".

        Returns
        -------
        pd.DataFrame
            One row per fiscal period with all balance sheet fields.

        Endpoint: GET /companies/statements/verbose (statements=BS)
        """
        simfin_period = "Q1,Q2,Q3,Q4" if period == "quarterly" else "FY"

        params = {
            "ticker": ticker,
            "statements": "BS",
            "period": simfin_period,
        }
        raw = self._get("companies/statements/verbose", params)

        if not raw:
            return pd.DataFrame()

        data_rows = raw[0].get("statements", [{}])[0].get("data", [])
        df = pd.DataFrame(data_rows)
        if "Report Date" in df.columns:
            df["Report Date"] = pd.to_datetime(df["Report Date"])
        return df

    def get_cash_flow_statements(
        self,
        ticker: str,
        period: str = "quarterly",
    ) -> pd.DataFrame:
        """Fetch full cash flow statement data for *ticker*.

        Parameters
        ----------
        ticker : str
            Company ticker symbol.
        period : str
            "quarterly" (default) or "annual".

        Returns
        -------
        pd.DataFrame
            One row per fiscal period with all cash flow fields.

        Endpoint: GET /companies/statements/verbose (statements=CF)
        """
        simfin_period = "Q1,Q2,Q3,Q4" if period == "quarterly" else "FY"

        params = {
            "ticker": ticker,
            "statements": "CF",
            "period": simfin_period,
        }
        raw = self._get("companies/statements/verbose", params)

        if not raw:
            return pd.DataFrame()

        data_rows = raw[0].get("statements", [{}])[0].get("data", [])
        df = pd.DataFrame(data_rows)
        if "Report Date" in df.columns:
            df["Report Date"] = pd.to_datetime(df["Report Date"])
        return df
