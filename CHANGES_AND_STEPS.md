# Changes and Steps to Follow

## Recent Changes (2026-03-08)

### Two-model architecture (Jorge)

- `model/strategy.py` is now the single source of truth for feature schemas.
  - `STANDARD_FEATURE_COLS` — 11 features (price + volatility) used for 25 tickers.
  - `FALLBACK_FEATURE_COLS` — same 11 features, used for BAC, GS, JPM, MA, V (banks and payment networks whose income statements are structurally incompatible with margin ratios).
  - 5 fundamental features (`Gross_Margin`, `Operating_Margin`, `Net_Margin`, `Debt_to_Equity`, `Operating_CF_Ratio`) are **commented out** until the API wrapper provides live quarterly data.
- `model/train.py` trains two pooled models: `model_pooled.pkl` (25 standard tickers) and `model_pooled_fallback.pkl` (5 fallback tickers). Run with `python model/train.py --all`.
- `app/pages/go_live.py` and `backtesting.py` load the correct model per ticker automatically via `get_feature_cols(ticker)` and `load_model(ticker)`.

### ETL update (Jorge)

- `etl/etl.py` now produces 5 additional volatility-normalised columns: `Log_Return`, `Volatility_20`, `Return_norm`, `Return_norm_Lag1`, `Return_norm_Lag2`.
- Processed CSVs in `data/processed/` must be regenerated with the updated ETL (see steps below).

### Feature schema fix (Marian)

- `STANDARD_FEATURE_COLS` in `model/strategy.py` had 5 fundamental features that require quarterly SimFin CSV files not available locally. Commented them out so both schemas use the same 11 features. This prevents the app from crashing when running predictions on standard tickers. Restore them once the API wrapper provides live fundamental data.

---

## How to Run the App Locally

Make sure your virtual environment is active and dependencies are installed (`pip install -r requirements.txt`).

**Step 1 — Regenerate processed CSVs** (required — old CSVs are missing the new volatility columns):

```bash
for t in AAPL ABBV ADBE AMD AMZN AVGO BAC CRM DIS GOOG GS INTC JNJ JPM KO MA MCD META MSFT NFLX NVDA ORCL PFE PLTR QCOM TSLA UNH V WMT; do python etl/etl.py --ticker $t; done
```

**Step 2 — Retrain both models:**

```bash
python model/train.py --all
```

**Step 3 — Launch the app:**

```bash
streamlit run app/Home.py
```

---

## What Is Still Missing

| # | What | Where | Assignment weight |
|---|------|--------|-------------------|
| 1 | **API Wrapper** — `pysimfin.py` methods all raise `NotImplementedError`. Need to implement `get_share_prices(ticker, start, end)` and `get_financial_statement(ticker, start, end)`. | `api_wrapper/pysimfin.py` | 20% |
| 2 | **Wire API into Go Live** — page currently reads from static local CSVs. Should fetch fresh data via the wrapper, apply ETL transforms, then predict. A `# TODO` comment marks the exact location. | `app/pages/go_live.py` line 14–23 | Core requirement |
| 3 | **Restore fundamental features** — once the API provides quarterly statements, uncomment the 5 fundamental features in `STANDARD_FEATURE_COLS` and retrain. | `model/strategy.py` | Improves model quality |
| 4 | **Cloud deployment** — app must be publicly accessible via Streamlit Cloud. Add `SIMFIN_API_KEY` as a secret in the Streamlit Cloud dashboard. | Streamlit Cloud | 15% |
| 5 | **Executive Summary** — 2–4 page PDF explaining the approach, ETL, ML model, and web app. Place it in `/docs/executive_summary.pdf`. | `/docs/` | 10% (deliverables) |
