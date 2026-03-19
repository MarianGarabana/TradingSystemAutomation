# Trading System Automation

A machine learning–based stock trading prediction system built with SimFin financial data.

**Live app:** https://pythongroupassignment.streamlit.app/

## Team

| Name | Role |
|------|------|
| *Marian Garabana* | ETL / Data Engineering |
| *Jorge Vildoso* | ML Model |
| *TBD* | API Wrapper |
| *Marian Garabana* | Streamlit App |

## Project Overview

This project builds an end-to-end pipeline that:
1. Fetches financial data via the SimFin API
2. Cleans and engineers features (ETL)
3. Trains a predictive ML model per ticker
4. Serves predictions through a Streamlit web app

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
│   └── processed/      # committed — 31 per-ticker processed CSVs (ETL output)
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
    └── executive_summary.pdf
```

## Data

Raw data is downloaded from [SimFin](https://simfin.com/) and stored locally in `data/raw/` — this folder is **git-ignored** and must be downloaded locally. Processed CSVs (`data/processed/`) are committed to the repo and contain the ETL output (one CSV per ticker, 22 features + Target).

## AI Usage

See [AI_USAGE_LOG.md](AI_USAGE_LOG.md) for a full log of AI tool usage in this project.
