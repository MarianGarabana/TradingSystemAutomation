"""
pysimfin.py — A lightweight OOP wrapper around the SimFin API.

No frameworks — plain Python with requests.

Example usage:
    from api_wrapper.pysimfin import PySimFin
    client = PySimFin(api_key="your_key")
    df = client.get_income_statements("AAPL")
"""

import os
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

SIMFIN_BASE_URL = "https://backend.simfin.com/api/v3"


class PySimFin:
    """Wrapper for the SimFin REST API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("SIMFIN_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SimFin API key not found. Set SIMFIN_API_KEY in your .env file."
            )
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"api-key {self.api_key}"})

    def _get(self, endpoint: str, params: dict = None) -> dict:
        """Make a GET request to the SimFin API."""
        url = f"{SIMFIN_BASE_URL}/{endpoint}"
        response = self.session.get(url, params=params or {})
        response.raise_for_status()
        return response.json()

    def get_companies(self) -> pd.DataFrame:
        """Return a DataFrame of all available companies."""
        # TODO: implement
        raise NotImplementedError

    def get_income_statements(self, ticker: str, period: str = "annual") -> pd.DataFrame:
        """Fetch income statement data for a ticker."""
        # TODO: implement
        raise NotImplementedError

    def get_balance_sheets(self, ticker: str, period: str = "annual") -> pd.DataFrame:
        """Fetch balance sheet data for a ticker."""
        # TODO: implement
        raise NotImplementedError

    def get_cash_flow_statements(self, ticker: str, period: str = "annual") -> pd.DataFrame:
        """Fetch cash flow statement data for a ticker."""
        # TODO: implement
        raise NotImplementedError

    def get_share_prices(self, ticker: str) -> pd.DataFrame:
        """Fetch historical share price data for a ticker."""
        # TODO: implement
        raise NotImplementedError
