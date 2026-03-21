# Trading System Automation

A machine learning–based stock trading prediction system built with SimFin financial data.

**Live app:** https://pythongroupassignment.streamlit.app/

## Team

| Name | Role |
|------|------|
| *Jorge Vildoso, Yaxin Wu* | ETL / Data Engineering |
| *Jorge Vildoso, Lorenz Rösguen, Omar Ajlouni* | ML Model |
| *Marian Garabana, Omar Ajlouni* | API Wrapper |
| *Marian Garabana, Marco Ortiz* | Streamlit App |

## Project Overview

End-to-end pipeline for predicting next-day stock signals across 31 large-cap US tickers:

1. Fetches financial data from SimFin (bulk CSVs for training, live API for predictions)
2. Cleans and engineers 22 features covering price, volume, and fundamentals (ETL)
3. Trains two pooled classification models that output BUY / SELL / HOLD signals
4. Serves predictions through a Streamlit web app with backtesting support

## Setup

```bash
# Clone the repo
git clone https://github.com/MarianGarabana/TradingSystemAutomation.git
cd TradingSystemAutomation

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy the environment template and fill in your API key
cp .env.example .env
```

## Running the App

```bash
streamlit run app/Home.py
```

## Project Structure

```
trading-app/
├── README.md
├── requirements.txt
├── AI_USAGE_LOG.md
├── .env.example
├── data/
│   ├── raw/            # git-ignored — SimFin bulk CSVs (download locally)
│   └── processed/      # committed — 31 per-ticker processed CSVs (ETL output, one per ticker)
├── notebooks/
│   ├── etl_exploration.ipynb
│   └── ml_exploration.ipynb
├── etl/
│   └── etl.py
├── model/
│   ├── train.py
│   ├── strategy.py
│   └── trained/
├── api_wrapper/
│   └── pysimfin.py
├── app/
│   ├── Home.py
│   └── pages/
│       ├── go_live.py
│       └── backtesting.py
└── docs/
    └── Trading System Automation_Executive_Summary.pdf
```

## Data

Raw data comes from [SimFin](https://simfin.com/) and lives in `data/raw/`, which is git-ignored and must be downloaded locally. Processed CSVs in `data/processed/` are committed and contain the ETL output (one file per ticker, 22 features + Target).

## AI Usage

See [AI_USAGE_LOG.md](AI_USAGE_LOG.md) for a full log of AI tool usage in this project.
