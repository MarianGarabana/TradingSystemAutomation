# Trading System Automation — Technical Executive Summary

**Master in Big Data and Data Science (MBDS 2026) | March 2026**

- **Live app:** [pythongroupassignment.streamlit.app](https://pythongroupassignment.streamlit.app)
- **Repo:** [github.com/MarianGarabana/TradingSystemAutomation](https://github.com/MarianGarabana/TradingSystemAutomation)

---

## What We Built

An end-to-end automated trading system built entirely in Python. It ingests historical financial data, trains a machine learning model to predict next-day stock direction, and serves live predictions through a Streamlit web app connected to the SimFin API. The system covers 31 US large-cap stocks and is deployed at pythongroupassignment.streamlit.app.

The project has two phases: an **offline phase** (ETL + training, run locally) and an **online phase** (Streamlit app, running on the cloud). The boundary matters because feature calculations must be identical on both sides — any drift breaks predictions.

---

## Repository Structure & Data Flow

| File / Folder | Job | Runs When |
|---|---|---|
| `etl/etl.py` | Cleans raw SimFin CSV, engineers all 22 features, writes processed CSV | Offline |
| `model/strategy.py` | Single source of truth: feature schemas, ticker classification, signal logic | Imported everywhere |
| `model/train.py` | Trains 4 classifiers on pooled data, saves best as `.pkl` | Offline |
| `model/trained/*.pkl` | Serialised models loaded at inference time | App runtime |
| `api_wrapper/pysimfin.py` | OOP wrapper: auth, rate-limiting, error handling | Online (`go_live`) |
| `app/Home.py` | Entry point: metrics, ticker table, tabs | Always loaded |
| `app/pages/go_live.py` | Fetches live data, engineers features, shows BUY/SELL/HOLD | Go Live page |
| `app/pages/backtesting.py` | Portfolio simulation vs. buy-and-hold on historical data | Backtesting page |
| `app/pages/prediction_bet.py` | Interactive bet simulator with leverage & horizon | Prediction Bet page |
| `data/processed/*.csv` | ETL output: one CSV per ticker, all features + Target | Written offline, read online |

### Data Flow

```
SimFin bulk CSV --> etl.py --> data/processed/*.csv --> train.py --> model/trained/*.pkl

SimFin API --> pysimfin.py --> go_live.py --> model/trained/*.pkl --> BUY/SELL/HOLD
```

---

## ETL Pipeline (`etl/etl.py`)

The ETL runs once per ticker and produces a clean, feature-rich CSV. Key steps:

**Price error handler.** Returns above 50% are almost certainly split adjustments. Rather than dropping those rows (which corrupts lag features), we set `Adj. Close` to `NaN` and forward-fill, preserving time-series continuity.

**Feature engineering.** Four feature groups are computed:

- **Price-based:** `MA5`, `MA20`, `RSI` (14-day), `MACD` (EMA12 – EMA26), Bollinger Bands, `Market_Cap`, `Volume_Change`.
- **Volatility-normalised:** `Log_Return` divided by 20-day rolling std (daily Z-score). Lags 1 and 2 are added. This makes AAPL and TSLA comparable despite very different volatility profiles.
- **Fundamental ratios:** `Gross Margin`, `Operating Margin`, `Net Margin`, `Debt-to-Equity`, `Operating CF Ratio` — merged with `pd.merge_asof(direction='backward')` to prevent look-ahead bias (only past-published reports are attached to each row).
- **Target:** next-day price direction via `.shift(-1)`. Only binary (up/down); zero-return rows are dropped at training time.

---

## Strategy Module (`model/strategy.py`)

The most important file architecturally. It defines feature schemas and signal logic that every other file imports, preventing train/serve skew.

**Two schemas.** The standard schema (16 features) is used for 26 tickers. The fallback schema (11 features, no fundamentals) is used for the 5 financial-sector tickers (`BAC`, `GS`, `JPM`, `MA`, `V`), whose income statements use different line items that make the standard ratios undefined.

**Signal logic.** A confidence threshold of 0.52 converts low-conviction model outputs into `HOLD` instead of generating a potentially incorrect `BUY` or `SELL`.

```python
def prediction_to_signal(prediction, confidence=None) -> Signal:
    if confidence is not None and confidence < 0.52:
        return Signal.HOLD
    return Signal.BUY if prediction == 1 else Signal.SELL
```

---

## ML Training (`model/train.py`)

**Per-ticker temporal split.** An 80/20 split is applied to each ticker individually before pooling. This ensures every ticker's most recent 20% of rows land in the test set — preventing alphabetical-order bias that would appear if the split were applied to the combined dataset.

**Class balancing.** The dataset is structurally biased: 52% of days are up, 48% down. Without correction, models learn to predict UP constantly and reach 52% accuracy without learning anything useful. All four classifiers apply balanced weights:

| Model | Balancing | Notes |
|---|---|---|
| Logistic Regression | `class_weight='balanced'` | Wrapped in `StandardScaler` pipeline |
| Random Forest | `class_weight='balanced'` | Scale-invariant, no scaler needed |
| Gradient Boosting | `sample_weight` in `.fit()` | Constructor does not accept `class_weight` |
| LightGBM | `is_unbalance=True` | Wrapped in `try/except` for optional dep. |

**C selection (Logistic Regression).** Regularisation strength is chosen via `TimeSeriesSplit(n_splits=5)` on the training set only, ensuring no future data leaks into earlier folds. `C=0.01` (strong regularisation) won because the five fundamental features are highly correlated (all derived from revenue), making aggressive L2 penalty necessary.

### Results

| Model | Pool | Test Accuracy | Selected |
|---|---|---|---|
| Logistic Regression (C=0.01) | Standard (26 tickers) | 50.1% | ✓ |
| Gradient Boosting | Fallback (5 tickers) | 54.7% | ✓ |
| Random Forest | Standard | 49.8% | – |
| LightGBM | Standard | 49.0% | – |

50.1% looks low but is meaningful: the unbalanced version achieved 52% by predicting UP almost always, with DOWN recall of only 0.35. After balancing, DOWN recall improved to 0.61.

---

## API Wrapper (`api_wrapper/pysimfin.py`)

The `PySimFin` class abstracts authentication, rate limiting, and response parsing. Key design choices:

- **Session reuse.** A single `requests.Session` keeps the TCP connection open across all ticker requests, reducing overhead compared to 31 individual connections.
- **Adaptive throttle.** The `_throttle()` method only sleeps if the last request was less than 0.55s ago — no penalty for naturally slow callers.
- **Column name matching.** `get_share_prices()` renames API response columns to match the SimFin bulk CSV exactly. This means `engineer_features()` works without modification on both data sources.
- **Outer-join merging.** `get_financial_statement()` fetches PL, BS, and CF in separate calls and merges them with `how='outer'` so no periods are lost when one statement has data but another does not.

---

## Web Application (`app/`)

Four Streamlit pages. All data-loading functions use `@st.cache_data` to prevent repeated API calls on every user interaction.

**`Home.py`** — Ticker table built from processed CSVs (only shows tickers with data). Company names are hardcoded so the cloud deployment does not require the raw SimFin file.

**`go_live.py`** — Fetches the last 200 days of prices (200 chosen to survive the 26-row warmup after rolling window computation), runs `engineer_features()`, attaches last-known fundamentals from the processed CSV, and calls the model on a single row (the latest complete trading day). Any API failure falls back silently to the processed CSV, with a badge indicating the active data source.

**`backtesting.py`** — Defaults the date picker to the training cutoff date, ensuring users see out-of-sample results by default. Portfolio simulation multiplies daily returns when the model predicts UP and holds cash (multiplier = 1.0) otherwise. A rolling 30-day accuracy chart shows whether prediction quality is stable over time.

**`prediction_bet.py`** — Users pick a direction (UP/DOWN), investment amount, leverage (1×–10×), and horizon (1 day to 1 year). The page displays the model's prediction alongside the bet and computes the simulated outcome from historical returns. Top-3 feature importances are shown with plain-English labels (e.g., high RSI → "overbought").

---

## Key Engineering Decisions

| Decision | Why It Matters |
|---|---|
| `strategy.py` as single source of truth | Training and inference share one feature list; renaming a feature updates both simultaneously. |
| `engineer_features()` imported by `go_live.py` | Identical code paths at train and serve time; no risk of divergent edge-case handling. |
| Per-ticker temporal split before pooling | Every ticker's recent data lands in the test set, regardless of alphabetical order. |
| API columns renamed to match bulk CSV | `engineer_features()` needs no conditional logic for live vs. historical data. |
| scikit-learn version pinned | A Streamlit Cloud crash taught us that an unpinned dependency installs the latest version, which may reject older `.pkl` files. |
| Silent API fallback to CSV | The app always shows something, even when the API is down or the key is missing. |
| Balanced class weights | Prevents models from achieving 52% accuracy by predicting UP on every single row. |

---

## Conclusions

The system delivers all requirements: a working ETL pipeline, a trained ML model with two pools, an OOP API wrapper, a multi-page web app, and a live cloud deployment. Every component has at least one fallback path.

The ML results are reported honestly. The standard model sits at 50.1% on unseen data — marginally above random but with meaningful DOWN recall (0.61 vs. 0.35 unbalanced). The financial stock model reaches 54.7%. Six model iterations were spent achieving balanced signals rather than chasing headline accuracy.

The main lesson: building a model that works in a notebook and building one that works correctly in a deployed app are very different problems. Feature consistency, API fallbacks, version pinning, and caching required more engineering thought than the model itself — which is the appropriate takeaway for a Python engineering course.
