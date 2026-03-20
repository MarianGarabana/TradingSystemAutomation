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
**What worked well:** What the AI got right on the first try
**What didn't work:** Anything that was wrong, incomplete, or needed iteration
**What we changed:** Edits or rejections and why
**What we learned:** What we understand now that we didn't before
```

---

## Entries

### 2026-03-03 — Marian Garabana — Claude (claude-sonnet-4-6)
**Task:** Scaffold repository structure
**Prompt (summary):** Asked Claude to generate the full folder/file skeleton for the trading system project including README, .gitignore, requirements.txt, placeholder source files, and .env.example.
**Output summary:** Claude created all files and pushed the initial commit to GitHub.
**What worked well:** The generated scaffold matched the project structure described in the assignment instructions — correct folder names (`etl/`, `model/`, `app/`, `api_wrapper/`, `docs/`, `data/`), a sensible `.gitignore` that excluded `.env` and bulk data CSVs, and a complete `requirements.txt` with the main libraries.
**What didn't work:** The README contained placeholder text ("TODO: add team names") and the source files were empty stubs. This was expected and intentional.
**What we changed:** Nothing at this stage — the scaffold was the starting point. Team names, implementation, and real content were added in later sessions.
**What we learned:** AI is very effective at generating boilerplate and project structure from a verbal description. It saved ~30 minutes of manual file creation and ensured we didn't miss standard files like `.gitignore` or `.env.example`.

---

### 2026-03-06 — Jorge Vildoso — Claude (claude-sonnet-4-6)
**Task:** Enrich ETL pipeline with fundamental data, Market_Cap feature, and regression target
**Prompt (summary):** Asked Claude to (1) add quarterly fundamental ratios (income, balance, cashflow) to the ETL with a point-in-time merge to avoid look-ahead bias; (2) add Market_Cap as a dynamic price feature; (3) change the Target variable from binary classification (0/1) to a continuous next-day return for regression; (4) update the data dictionary in the notebook.
**Output summary:** Claude added `fetch_fundamentals()`, `_compute_fundamental_features()`, and `merge_fundamentals()` to `etl/etl.py`; updated `engineer_features()` with Market_Cap and regression Target; updated `etl_exploration.ipynb` with the enriched pipeline, feature inventory, and a full data dictionary (17 features: 12 price + 5 fundamental).
**What worked well:** The point-in-time merge logic using `merge_asof(direction='backward')` was correct on the first attempt — it properly attaches only the most recently published quarterly report to each trading day, avoiding look-ahead bias. The data dictionary structure (grouped by feature type) was clear and reusable.
**What didn't work:** The initial version tried to merge on fiscal year end date instead of Publish Date, which would have introduced look-ahead bias. We caught this during review and Claude corrected it immediately.
**What we changed:** Verified and re-ran ETL for all 5 tickers to regenerate processed CSVs with the full 17-feature schema.
**What we learned:** Financial data is complex and requires careful handling to avoid look-ahead bias. Also, look for finance metrics in order to improve the feature engineering.

---

### 2026-03-07 — Marian Garabana — Claude (claude-sonnet-4-6)
**Task:** Redesign Home, Go Live, and Backtesting Streamlit pages
**Prompt (summary):** Asked Claude to redesign `app/Home.py` with a Robinhood-style lime green + black theme, and update `etl/etl.py` to handle missing fundamentals. Asked Claude to help me redesign the Streamlit pages charts (`go_live.py` and `backtesting.py`). Requested that all changes be commented with brief explanations of why and how each part works.
**Output summary:** Claude rewrote `Home.py` and added a searchable ticker table loaded from `data/processed/`. It created the `config.toml` file to enforce the dark theme. Updated `etl/etl.py` so fundamentals are optional (returns `None` instead of crashing). Implemented `go_live.py` with a price chart (MA5/MA20/Bollinger Bands), RSI chart with overbought/oversold zones, and MACD chart with colour-coded histogram. Fixed `backtesting.py` with a $10k portfolio simulation (invested on BUY signals vs. cash on SELL/HOLD), buy-and-hold comparison, cumulative return chart, rolling 30-day accuracy chart, and a signal history table. Both pages degrade gracefully when no trained model is present by using an "Always Buy" baseline.
**What worked well:** The graceful degradation pattern (fallback to "Always Buy" baseline when no `.pkl` exists) was a good suggestion that let us demo the app visually before the ML model was trained. The chart styling (dark theme, colour-coded histogram bars) worked on the first attempt with no manual CSS tweaks needed.
**What didn't work:** The first version of the backtesting page had an off-by-one error in the signal alignment — predictions for day N were being applied to day N's return instead of day N+1's return, which overstated performance. Claude fixed this on the second iteration.
**What we changed:** Requested revisions and guidance to understand the signal alignment logic. The fix was straightforward once the issue was explained.
**What we learned:** Streamlit's `config.toml` is the correct way to enforce a global theme — trying to override colours with `st.markdown` CSS hacks is fragile. We also learned that signal alignment (predict on day N → trade on day N+1) is easy to get wrong and must be explicitly verified in backtesting code.

---

### 2026-03-08 — Jorge Vildoso — Claude (claude-sonnet-4-6)
**Task:** Two-model ML architecture covering all 30 tickers + volatility-normalised features
**Prompt (summary):** Asked Claude to (1) expand the ticker universe from 5 to 30 and regenerate processed CSVs; (2) add 5 volatility-normalised features (Log_Return, Volatility_20, Return_norm, Return_norm_Lag1, Return_norm_Lag2) to the ETL and notebook to enable cross-stock pooled training; (3) work with a two-model pooled architecture: a standard model (25 tickers, 16 features) and a fallback model (BAC/GS/JPM/MA/V, 11 features, no fundamentals); (4) refactor `model/strategy.py` as the single source of truth for feature schemas and ticker classification; (5) rewrite `model/train.py` with an `--all` flag that trains both pooled models; (6) update `go_live.py` and `backtesting.py` to use schema-aware model loading via `get_feature_cols(ticker)`.
**Output summary:** Claude added the 5 vol-norm features to `etl/etl.py` and `etl_exploration.ipynb`; rewrote `model/strategy.py` with `STANDARD_FEATURE_COLS` and `FALLBACK_FEATURE_COLS`.
**What worked well:** The `strategy.py` single-source-of-truth pattern worked well, having feature schemas defined in one place and imported everywhere eliminated the risk of mismatches between training and inference. The `--all` CLI flag made retraining both pools with one command convenient.
**What didn't work:** Claude added an unsolicited graceful NaN fallback (zero-filling features when the model is loaded) that we explicitly did not want — it would silently produce wrong predictions instead of failing visibly. We caught this during code review.
**What we changed:** Reverted the zero-filling fallback; kept the clean two-model separation instead. Decided to defer API wrapper implementation to a later session.
**What we learned:** When giving AI a multi-part spec, it sometimes adds "defensive" code that solves a problem you haven't asked it to solve and this can conflict with your architecture. We also learned that pooling all tickers in one model dramatically reduces overfitting.

---

### 2026-03-11 — Jorge Vildoso — Claude (claude-sonnet-4-6)
**Task:** Update ML evaluation report with the results from the pooled models
**Prompt (summary):** Asked Claude to go through the stored outputs in the notebook and update `docs/ml_evaluation_report.md` to include the actual numbers from the production models:  performance metrics, model comparison tables, and the settings used for each model (LinearRegression, RandomForest, GradientBoosting).
**Output summary:** Claude added a new section to the report with model comparison tables (R², MAE for each candidate), a summary of how many rows were used for training and testing, and a hyperparameter table for each model showing which settings were explicitly chosen vs. left as defaults.
**What worked well:** Claude pulled the numbers directly from the notebook cell outputs without inventing values, and structured everything in markdown tables that were ready to use. It also spotted that we had not documented which parameters were defaults vs. explicit choices, and added that detail proactively.
**What we learned:** Listing hyperparameters in the report, even the ones left at default, forces you to understand what each setting does and why it was left unchanged.

---

### 2026-03-11 — Jorge Vildoso — Claude (claude-opus-4-6)
**Task:** Obtain expert guidance on evaluating machine learning models for financial market predictions
**Prompt (summary):** Asked Claude to review the file `docs/ml_evaluation_report.md` to understand the model evaluation metrics and provide recommendations from a machine learning and financial perspective.
**Output summary:** Claude suggested adjustments to some model hyperparameters and proposed improvements to parts of the code and evaluation approach in order to improve model performance.
**What worked well:** The recommendations improved the models performance
**What we learned:** It is important to interpret model evaluation results in the context of the underlying business objective. Understanding the metrics allows targeted adjustments to the modeling approach and code, leading to meaningful performance improvements.

---

### 2026-03-11 — Jorge Vildoso — Claude (claude-sonnet-4-6)
**Task:** Improve the training script and add a visual calibration tool
**Prompt (summary):** Asked Claude to make four specific changes to the codebase: (1) in `model/train.py`, replace the hardcoded `LinearRegression` with `Ridge` and add a function `_select_ridge_alpha()` that automatically tests several regularisation values and picks the best one using cross-validation; (2) add a new function `_build_lgbm()` to add LightGBM as a fourth model to compare, with a try/except so the script doesn't crash if LightGBM is not installed; (3) add a function `_log_direction_and_signals()` that prints how often the model predicts the correct direction and how predictions split across the five signal categories (HIGH RISE, LOW RISE, STAY, LOW FALL, HIGH FALL); (4) create a new file `model/calibration.py` that loads the trained models, reconstructs the test data, and generates bar charts showing how predictions compare to actual results. As a reference for the model implementations and cross-validation approach, we used code examples and concepts covered in the Machine Learning II course.
**Output summary:** Claude rewrote the relevant sections of `train.py`, created `model/calibration.py` from scratch, and added `lightgbm>=4.0.0` to `requirements.txt`. Running `python model/train.py --all` showed the automatic alpha selection picking 10 for the standard model and 100 for the fallback. Running `python model/calibration.py` saved two PNG charts to `model/trained/`.
**What worked well:** The new functions integrated cleanly into the existing `_train_pooled()` structure without requiring a full rewrite. The `_build_lgbm()` try/except worked correctly. The calibration script ran on the first attempt with no import errors.
**What didn't work:** After running the scripts we noticed a serious problem in the `_log_direction_and_signals()` output: 100% of predictions were falling into the STAY bin, meaning the model was generating only HOLD signals and never BUY or SELL.
**What we changed:** Nothing in the code for now, the signal threshold issue is documented for the next session. The fix will involve either lowering the minimum allowed alpha value or adjusting the thresholds in `prediction_to_signal()` in `strategy.py`.
**What we learned:** The code for the model works as expected, but the signal threshold issue affects a business objective. This will have to be adressed.

---

### 2026-03-11 — Jorge Vildoso — Claude (claude-sonnet-4-6)
**Task:** Write a full v2 ML evaluation report with all the new results
**Prompt (summary):** Asked Claude to create `docs/ml_evaluation_report_v2.md` using the outputs from the updated `train.py` and `calibration.py` runs — covering all four models, the direction accuracy numbers, the signal bin breakdown, the decile calibration tables, and a before/after comparison against the v1 results.
**Output summary:** Claude created a 10-section report including a 4-model comparison table (Ridge, RandomForest, GradientBoosting, LightGBM), the signal bin breakdown showing 100% STAY, the full decile calibration tables for both model pools, a delta table comparing v1 vs. v2 performance, and an updated limitations section that explicitly lists the HOLD signal problem as the top issue to fix.
**What worked well:** Everything worked as expected. Claude added a v1 vs. v2 comparison table on its own initiative — we hadn't asked for it, but it made the improvement between versions concrete and easy to explain. It also correctly identified the most interesting result from the calibration output: the top-decile predictions for the financial stocks group (BAC, GS, JPM, MA, V) have a meaningfully higher hit rate than the rest.

---

### 2026-03-12 — Jorge Vildoso — Claude (claude-sonnet-4-6)
**Task:** Switch the ML pipeline from regression to binary classification (v4)
**Prompt (summary):** (1) Replace the four regression models with their classifier equivalents; (2) update the signal logic so that the model output directly means BUY (up) or SELL (down), with a HOLD triggered when the model is not confident enough (confidence < 52%) and (3) write a new evaluation report.
**Output summary:** Claude rewrote `model/train.py` to use four classifiers. Updated `model/strategy.py` so `prediction_to_signal()` now takes a confidence score instead of threshold percentiles. Ran `python model/train.py --all` and captured all results. Created and fully filled `docs/ml_evaluation_report_v4.md` with all metrics, comparisons and findings.
**What worked well:** The change was clean and all pages continued working after the switch.
**What we changed:** No code changes beyond the spec. The `thresholds.json` file from v3 was made redundant and is no longer used. The report documents all findings, including the statistical insignificance of the standard pool result.
**What we learned:** Framing the problem as "up or down" (classification) instead of "how much will it move?" (regression) is more honest, it matches the actual decision a trader makes. The previous percentile-based fix was a workaround that made the signals look balanced on paper, but they were based on ranks rather than genuine model conviction. We also learned that stock markets are hard to predict: even with over 24,000 training rows and four different model types, the best result on the broad pool is statistically indistinguishable from random guessing. The financial stocks group (BAC, GS, JPM, MA, V) are more predictable than the tech/consumer group, which is consistent with what the calibration charts showed in v2.

---

### 2026-03-12 — Jorge Vildoso — Claude (claude-sonnet-4-6)
**Task:** Fix the 100% HOLD signal problem across the full trading system
**Prompt (summary):** Provided a detailed spec: replace hardcoded ±0.5%/±2.0% thresholds in `prediction_to_signal()` with percentile-based thresholds computed from the model's own prediction distribution (p75 = BUY threshold, p25 = SELL threshold)
**Output summary:** Claude updated all four files. `strategy.py`: `prediction_to_signal()` now accepts an optional `thresholds` dict (with "buy" and "sell" keys) and falls back to the legacy integer convention if None. `train.py`: added `compute_and_save_thresholds()` — loads both pooled models, runs inference on all usable rows per pool, computes p75/p25 percentiles, saves `{"standard": {...}, "fallback": {...}}` to `model/trained/thresholds.json`; wired into the `--all` CLI branch. `go_live.py` and `backtesting.py`: added `load_thresholds(ticker)` helper and updated inference to pass thresholds to `prediction_to_signal()` / `run_backtest()`. Ran `model/train.py --all` to generate thresholds.json (existing .pkl files reused, no retraining).
**What worked well:** The four-file change was clean and coherent. The percentile approach is self-adapting: it will always generate exactly 25% BUY and 25% SELL signals regardless of the model's prediction range.
**What we changed:** No code changes beyond the spec. The existing `.pkl` files were reused without retraining — `compute_and_save_thresholds()` only runs inference to compute percentiles.
**What we learned:** Percentile-based thresholds are a simple and robust solution for models (like Ridge with high alpha) whose output range is much narrower than any hardcoded threshold. The key insight is that the absolute prediction magnitudes don't necessarily matter, what matters is the relative rank of each prediction within the model's own distribution.

---

### 2026-03-18 — Jorge Vildoso — Claude (claude-sonnet-4-6)
**Task:** Unify the ML iteration history into a single notebook and refine the current comments and markdowns
**Prompt (summary):** Asked Claude to take the four separate ML evaluations and consolidate them into a single `notebooks/ml_exploration.ipynb` that tells the full iteration story. Asked for refine the markdown cells and comments.
**Output summary:** Claude restructured the notebook into 5 sections covering the models journey, with markdown cells explaining what was tried and what changed. The v4 section calls `model/train.py` directly to avoid duplicating production code. A final summary table compares
all four versions side by side. Inline comments explain every non-obvious choice (e.g. why StandardScaler is applied only to LinearRegression, why thresholds are computed from the training set only, why zero-return rows are dropped for binary classification).
**What worked well:** Having all four iterations in one notebook makes the exploration narrative clear and easy to follow for graders. The markdown sections between code blocks explain the "why" behind each change, not just the "what".
**What didn't work:** Some comments and markdowns where too long and detailed.
**What we changed:** Reviewed the comments and markdowns and make some adjustments to make them more concise and clear.
**What we learned:** AI usually gives us a good starting point, but we need to review and refine it to make sure it meets our needs.

---

### 2026-03-18 — Jorge Vildoso — Claude (claude-sonnet-4-6)
**Task:** Use the 5 fundamental features in the standard model, add ABBV to the training pool and fix the live inference path
**Prompt (summary):** We noticed that the standard model was supposed to use 16 features (price + volatility + 5 fundamental ratios) while the fallback model uses 11 (no fundamentals), but in practice both were using the same 11 features because the fundamental columns were commented out in `strategy.py`. We asked Claude to do a deep review of all affected files and propose a plan to fix everything end to end.
**Output summary:** Claude read `strategy.py`, `train.py`, `etl.py`, `go_live.py`, and the processed CSVs to map the full picture. It identified four things that needed to change: (1) uncomment the 5 fundamental features in `STANDARD_FEATURE_COLS` in `strategy.py`; (2) add ABBV to `ALL_TICKERS` in `etl.py` and regenerate its CSV with fundamentals; (3) fix a hidden crash in the live API path of `go_live.py`; (4) update `train.py`.
**What worked well:** Claude spotted the live inference crash before we hit it in production. Propose a clear plan we could review and implement.
**What didn't work:** Nothing failed.
**What we changed:** Accepted all four code changes as proposed. The solution for the live inference path (attach last-known fundamentals from CSV) was chosen over the alternative (fetch fundamentals live from the API) because it is simpler and requires no extra API calls.
**What we learned:** Changing a feature schema is not just a one-line edit — it touches every place the model is loaded and used. The most value contribution of AI was to trace all the parts of the code that use the features and find all the places that needed to be updated.

---

### 2026-03-19 — Jorge Vildoso — Claude (claude-sonnet-4-6)
**Task:** Diagnose and fix a production crash on the live Streamlit app
**Prompt (summary):** Shared a screenshot of the app crashing with a `ValueError` inside sklearn's prediction code. Asked Claude to analyse the error and propose a fix.
**Output summary:** Claude traced the full call stack from `go_live.py` down into sklearn's internal validation function. It identified the root cause: our `requirements.txt` had `scikit-learn>=1.3.0` with no upper bound, so Streamlit Cloud was free to install the latest version of sklearn — which turned out to be stricter about validating input data when the model had been pickled on a different version locally.
**What worked well:** The diagnosis was fast and accurate. Claude followed the traceback step by step and landed on the real cause without going in circles.
**What didn't work:** Nothing.
**What we changed:** Accepted all proposed changes as-is. Three files updated: `requirements.txt`, `go_live.py`, `backtesting.py`.
**What we learned:** Pinning only a minimum version in `requirements.txt` is risky for deployed apps — cloud platforms will always install the latest compatible version, which may behave differently from what you tested locally.

---

### 2026-03-19 — Jorge Vildoso — Claude (claude-sonnet-4-6)
**Task:** Apply balanced class weights to all the models reviewed to solve the UP bias problem
**Prompt (summary):** The models showed strong UP bias. Asked Claude to add `class_weight='balanced'` to RF, `sample_weight=compute_sample_weight('balanced')` to GBC's `.fit()` call, `is_unbalance=True` to LGBM and execute the training according to the code review in Machine Learning class.
**Output summary:** Claude modified `train.py` (4 targeted edits: import, RF constructor, LGBM constructor, GBR fit call), ran `python model/train.py --all` to produce new `.pkl` files.
**What worked well:** Claude's execution was precise, it identified that GBC requires a different mechanism (`sample_weight` in `.fit()`) vs the other classifiers (constructor parameter), which is a non-obvious API difference.
**What didn't work:** Nothing failed.
**What we changed:** Accepted all changes as proposed. The training confirmed: standard winner switched from GBC → LogReg(C=0.01, balanced).
**What we learned:** Balanced weights are the correct default for any classification problem with a structural class imbalance in the base rate.

---

### 2026-03-20 — Marian Garabana — Claude (claude-sonnet-4-6)
**Task:** Implement the prediction page (`app/pages/prediction_bet.py`)
**Prompt (summary):** Asked Claude to build the prediction bet page for the Streamlit app. A page where users can simulate a bet on a stock based on the model's prediction and see whether the outcome was correct.
**Output summary:** Claude designed the full skeleton of the page: layout structure, UI sections (ticker selector, bet input, prediction display, outcome reveal), and the wiring to the existing model inference and API fetch logic. Claude also diagnosed and fixed several bugs in the SimFin API fetch path that caused incorrect or missing data to be returned during live inference.
**What worked well:** The design skeleton was solid on the first attempt: the section layout, component ordering, and the flow from ticker selection to prediction to outcome matched what was needed without major rework. The API bug fixes were accurate and targeted.
**What didn't work:** Some API fetch calls were returning wrong or no data due to parameter mismatches in the wrapper. Claude identified and fixed these.
**What we changed:** Every proposed change was reviewed and approved before execution, no code was applied without consent. Some minor UI tweaks were made after the skeleton was in place to match the existing app theme and improve the design principles of the application overall. 
**What we learned:** Having a clear page skeleton before filling in logic makes it much easier to design in a more structured and organized way.
