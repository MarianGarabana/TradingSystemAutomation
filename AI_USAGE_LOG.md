# AI Usage Log

This file documents all use of AI tools (ChatGPT, Claude, Copilot, etc.) during this project.

**Policy:** Every non-trivial AI-assisted contribution must be logged here with the prompt, output summary, and what was kept / changed.

---

## Log Format

```
### [Date] — [Team Member] — [Tool]
**Task:** What you were trying to do
**Prompt (summary):** What you asked
**Output summary:** What the AI produced
**What we used:** Which parts we kept and why
**What we changed:** Edits or rejections and why
```

---

## Entries

### 2026-03-06 — Jorge — Claude (claude-sonnet-4-6)
**Task:** Enrich ETL pipeline with fundamental data, Market_Cap feature, and regression target
**Prompt (summary):** Asked Claude to (1) add quarterly fundamental ratios (income, balance, cashflow) to the ETL with a point-in-time merge to avoid look-ahead bias; (2) add Market_Cap as a dynamic price feature; (3) change the Target variable from binary classification (0/1) to a continuous next-day return for regression; (4) update the data dictionary in the notebook.
**Output summary:** Claude added `fetch_fundamentals()`, `_compute_fundamental_features()`, and `merge_fundamentals()` to `etl/etl.py`; updated `engineer_features()` with Market_Cap and regression Target; updated `etl_exploration.ipynb` with the enriched pipeline, feature inventory, and a full data dictionary (17 features: 12 price + 5 fundamental).
**What we used:** All changes kept — fundamental enrichment logic, Market_Cap feature, regression Target definition, and the data dictionary markdown.
**What we changed:** Verified and re-ran ETL for all 5 tickers to regenerate processed CSVs with the full 17-feature schema.

### 2026-03-07 — Marian Garabana — Claude (claude-sonnet-4-6)
**Task:** Redesign Home, Go Live, and Backtesting Streamlit pages
**Prompt (summary):** Asked Claude to redesign `app/Home.py` with a Robinhood-style lime green + black theme, and update `etl/etl.py` to handle missing fundamentals. Asked Claude to help me redesign the Streamlit pages charts (`go_live.py` and `backtesting.py`). Requested that all changes be commented with brief explanations of why and how each part works.
**Output summary:** Claude rewrote `Home.py` and a searchable ticker table loaded from `data/processed/`. It helped me to create the `config.toml` file that helped to change the theme to the colors we wanted to use. Updated `etl/etl.py` so fundamentals are optional (returns `None` instead of crashing). Claude first performed a full gap analysis (ETL done, API wrapper empty, models untrained, both app pages stubs, no deployment, no executive summary). Then it implemented `go_live.py`: model loading that works with or without a .pkl, price chart with MA5/MA20/Bollinger Bands, RSI chart with overbought/oversold zones, and MACD chart with colour-coded histogram. It also fixed `backtesting.py`: ticker + date-range controls, simulation of a $10k portfolio (invested on BUY signals vs. cash on SELL/HOLD), buy-and-hold comparison, cumulative return chart, rolling 30-day accuracy chart, and a signal history table. Both pages degrade when no trained model is present by using an "Always Buy" baseline instead so I was able to see the charts regardless of not having a ML model implemented yet. 
**What we used:** All changes kept. The feature column list (`FEATURE_COLS`), signal logic (imported from `model/strategy.py`), and chart styling (dark theme matching `config.toml`) were kept.
**What we changed:** I requested revisions and guidance to help me implement the changes and explanations and reasonings of the changes implemented by Claude.

### 2026-03-08 — Jorge Vildoso — Claude (claude-sonnet-4-6)
**Task:** Two-model ML architecture covering all 30 tickers + volatility-normalised features
**Prompt (summary):** Asked Claude to (1) expand the ticker universe from 5 to 30 and regenerate processed CSVs; (2) add 5 volatility-normalised features (Log_Return, Volatility_20, Return_norm, Return_norm_Lag1, Return_norm_Lag2) to the ETL and notebook to enable cross-stock pooled training; (3) design a two-model pooled architecture — a standard model (25 tickers, 16 features) and a fallback model (BAC/GS/JPM/MA/V, 11 features, no fundamentals) — to handle tickers where quarterly fundamental data is structurally incompatible; (4) refactor `model/strategy.py` as the single source of truth for feature schemas and ticker classification; (5) rewrite `model/train.py` with an `--all` flag that trains both pooled models; (6) update `go_live.py` and `backtesting.py` to use schema-aware model loading via `get_feature_cols(ticker)`.
**Output summary:** Claude added the 5 vol-norm features to `etl/etl.py` and `etl_exploration.ipynb`; updated `ml_exploration.ipynb` for regression with 16 features and dynamic tickers; rewrote `model/strategy.py` with `STANDARD_FEATURE_COLS`, `FALLBACK_FEATURE_COLS`, `FALLBACK_TICKERS`, `is_fallback_ticker()`, and `get_feature_cols()`; rewrote `model/train.py` with `_train_pooled()`, `train_pooled_standard()`, `train_pooled_fallback()`, and `--all` CLI flag; updated both Streamlit pages to call `load_model(ticker)` and `get_feature_cols(ticker)` at runtime; trained and saved `model_pooled.pkl` (27,696 rows) and `model_pooled_fallback.pkl` (6,075 rows).
**What we used:** All changes kept. The two-model architecture and the `strategy.py` single-source-of-truth pattern were adopted as the production design.
**What we changed:** Reverted an unsolicited graceful NaN fallback (zero-filling) that Claude implemented without being asked — kept the clean two-model separation instead. Decided to defer API wrapper implementation to a later session.

### 2026-03-03 — Marian Garabana — Claude (claude-sonnet-4-6)
**Task:** Scaffold repository structure
**Prompt (summary):** Asked Claude to generate the full folder/file skeleton for the trading system project including README, .gitignore, requirements.txt, placeholder source files, and .env.example.
**Output summary:** Claude created all files and pushed the initial commit to GitHub.
**What we used:** The full scaffold as a starting point.
**What we changed:** Will update team names, fill in real implementation, and remove placeholder comments as we build.
