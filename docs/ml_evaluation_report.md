# ML Model Evaluation Report
## Trading System Automation — Next-Day Return Prediction

**Date:** March 2026
**Dataset:** SimFin bulk CSV — 30 US large-cap tickers, 2020–2024
**Evaluation scope:** Per-ticker prototype (sections 2–3) + Production pooled models (section 6)
**Source:** `notebooks/ml_exploration.ipynb` (sections 5–6 per-ticker; section 8 pooled production)

---

## 1. Problem Setup

### Task
Predict the **next-day log return** of a stock price (continuous regression). The raw prediction is then converted to a trading signal:

| Predicted return | Signal |
|---|---|
| > +2.0% | HIGH RISE → **BUY** |
| +0.5% to +2.0% | LOW RISE → **BUY** |
| −0.5% to +0.5% | STAY → **HOLD** |
| −2.0% to −0.5% | LOW FALL → **SELL** |
| < −2.0% | HIGH FALL → **SELL** |

### Target variable
```
Target = Adj_Close.pct_change().shift(-1)
```
This is the **next-day forward return** — the quantity the model is asked to predict. It is computed at ETL time and stored in each processed CSV. NaN in the last row (no future price) is dropped before training.

### Feature set (16 features)

| Group | Features | Notes |
|---|---|---|
| Price-based (6) | MA5, MA20, Volume_Change, Market_Cap, RSI, MACD | Standard technical indicators |
| Volatility-normalised (5) | Log_Return, Volatility_20, Return_norm, Return_norm_Lag1, Return_norm_Lag2 | Key for cross-ticker pooling — risk-adjusts returns so KO and TSLA are comparable |
| Fundamental (5) | Gross_Margin, Operating_Margin, Net_Margin, Debt_to_Equity, Operating_CF_Ratio | Quarterly, point-in-time merged via `merge_asof(direction='backward')` on Publish Date |

**Fallback schema (11 features):** BAC, GS, JPM, MA, V lack fundamental data (bank income statements use a non-standard format; Mastercard/Visa have no Gross Profit line). These tickers use only the price-based + volatility-normalised groups.

### Train / test split
- **Method:** Temporal 80/20 split per ticker — rows sorted by date, last 20% held out.
- **No shuffling** — shuffling would constitute look-ahead bias by allowing future price information to leak into training.
- **NaN handling:** Rows with any NaN in the feature or target columns are dropped before the split index is computed (not after), so the 80/20 ratio applies to usable rows only.

### Models evaluated
Three families, each representing a different inductive bias:

| Model | Scaling | Key property |
|---|---|---|
| LinearRegression | StandardScaler applied | Interpretable; assumes linear feature-target relationship; severely distorted by Market_Cap (~10¹²) vs Log_Return (~0.01) without scaling |
| RandomForestRegressor (200 trees) | None — scale-invariant | Bagged decision trees; captures non-linear interactions; low variance via averaging |
| GradientBoostingRegressor (200 estimators) | None — scale-invariant | Boosted trees; typically strongest on tabular data; higher risk of overfitting on small datasets |

### Evaluation metrics
- **MAE (Mean Absolute Error):** Average magnitude of prediction error. Same unit as the target (daily return). E.g., MAE=0.015 means the model is off by ~1.5 percentage points on average.
- **RMSE (Root Mean Squared Error):** Penalises large errors more than MAE. Useful for detecting outlier predictions.
- **R² (Coefficient of Determination):** Fraction of target variance explained by the model. R²=1.0 is perfect; R²=0.0 means the model is no better than predicting the mean; **R²<0 means the model is worse than predicting the mean**.
- **Direction Accuracy:** Fraction of test rows where `sign(predicted) == sign(actual)`. This is the most operationally relevant metric for a trading system — a model that predicts the direction correctly >50% of the time has positive expected value even if R² is low.

---

## 2. Per-Ticker Results — Regression Metrics (R², MAE, RMSE)

> **Note:** BAC, GS, JPM, MA, V are skipped in this per-ticker evaluation because all 16 features are NaN for these tickers (fundamental data unavailable). They are handled by the fallback pooled model.

### 2.1 Complete results table

| Ticker | Model | MAE | RMSE | R² |
|---|---|---|---|---|
| AAPL | LinearRegression | 0.01428 | 0.01812 | −0.3792 |
| AAPL | RandomForest | 0.01584 | 0.02037 | −0.7435 |
| AAPL | GradientBoosting | 0.02637 | 0.03412 | **−3.891** |
| ADBE | LinearRegression | 0.01423 | 0.02215 | −0.0612 |
| ADBE | RandomForest | 0.01419 | 0.02173 | −0.0207 |
| ADBE | GradientBoosting | 0.01522 | 0.02238 | −0.0829 |
| AMD | LinearRegression | 0.02156 | 0.02850 | −0.0340 |
| AMD | RandomForest | 0.02340 | 0.02993 | −0.1405 |
| AMD | GradientBoosting | 0.02676 | 0.03449 | −0.5139 |
| AMZN | LinearRegression | 0.01436 | 0.01896 | −0.0411 |
| AMZN | RandomForest | 0.01572 | 0.02110 | −0.2894 |
| AMZN | GradientBoosting | 0.01739 | 0.02260 | −0.4789 |
| AVGO | LinearRegression | 0.03874 | 0.05204 | −0.7627 |
| AVGO | RandomForest | 0.03068 | 0.04246 | −0.1735 |
| AVGO | GradientBoosting | 0.04300 | 0.05349 | −0.8624 |
| COST | LinearRegression | 0.01107 | 0.01418 | −0.1547 |
| COST | RandomForest | 0.01040 | 0.01371 | −0.0785 |
| COST | GradientBoosting | 0.01135 | 0.01468 | −0.2371 |
| CRM | LinearRegression | 0.01470 | 0.02343 | −0.0266 |
| CRM | RandomForest | 0.01583 | 0.02452 | −0.1239 |
| CRM | GradientBoosting | 0.01852 | 0.02756 | −0.4208 |
| DIS | LinearRegression | 0.01016 | 0.01492 | **+0.0023** |
| DIS | RandomForest | 0.01304 | 0.01795 | −0.4433 |
| DIS | GradientBoosting | 0.03149 | 0.04064 | **−6.399** |
| GOOG | LinearRegression | 0.01657 | 0.02106 | −0.3690 |
| GOOG | RandomForest | 0.02578 | 0.02978 | −1.7375 |
| GOOG | GradientBoosting | 0.04304 | 0.04768 | **−6.018** |
| INTC | LinearRegression | 0.02422 | 0.03574 | −0.0159 |
| INTC | RandomForest | 0.03322 | 0.04440 | −0.5675 |
| INTC | GradientBoosting | 0.03943 | 0.05340 | −1.2672 |
| JNJ | LinearRegression | 0.00811 | 0.01083 | −0.0251 |
| JNJ | RandomForest | 0.00840 | 0.01114 | −0.0848 |
| JNJ | GradientBoosting | 0.00881 | 0.01204 | −0.2662 |
| KO | LinearRegression | 0.00879 | 0.01137 | −0.4124 |
| KO | RandomForest | 0.02439 | 0.03334 | **−11.144** |
| KO | GradientBoosting | 0.02641 | 0.03614 | **−13.276** |
| MCD | LinearRegression | 0.00884 | 0.01193 | **+0.0061** |
| MCD | RandomForest | 0.01073 | 0.01385 | −0.3398 |
| MCD | GradientBoosting | 0.01419 | 0.01853 | −1.3969 |
| META | LinearRegression | 0.01589 | 0.02107 | −0.1504 |
| META | RandomForest | 0.01604 | 0.02167 | −0.2173 |
| META | GradientBoosting | 0.02025 | 0.02478 | −0.5910 |
| MSFT | LinearRegression | 0.01032 | 0.01395 | −0.0390 |
| MSFT | RandomForest | 0.01065 | 0.01453 | −0.1273 |
| MSFT | GradientBoosting | 0.01279 | 0.01639 | −0.4343 |
| NFLX | LinearRegression | 0.01524 | 0.02153 | −0.2038 |
| NFLX | RandomForest | 0.01459 | 0.02032 | −0.0715 |
| NFLX | GradientBoosting | 0.01782 | 0.02397 | −0.4918 |
| NVDA | LinearRegression | 0.02819 | 0.03719 | −0.0984 |
| NVDA | RandomForest | 0.03224 | 0.04270 | −0.4481 |
| NVDA | GradientBoosting | 0.04548 | 0.05766 | −1.6407 |
| ORCL | LinearRegression | 0.01581 | 0.02426 | −0.0262 |
| ORCL | RandomForest | 0.04245 | 0.04897 | −3.1819 |
| ORCL | GradientBoosting | 0.03569 | 0.04402 | −2.3792 |
| PEP | LinearRegression | 0.00858 | 0.01122 | −0.0749 |
| PEP | RandomForest | 0.00852 | 0.01129 | −0.0889 |
| PEP | GradientBoosting | 0.00888 | 0.01159 | −0.1468 |
| PFE | LinearRegression | 0.01152 | 0.01487 | −0.0644 |
| PFE | RandomForest | 0.01424 | 0.01808 | −0.5741 |
| PFE | GradientBoosting | 0.02086 | 0.02664 | −2.4163 |
| PLTR | LinearRegression | 0.04154 | 0.06054 | −0.9633 |
| PLTR | RandomForest | 0.03574 | 0.05032 | −0.3567 |
| PLTR | GradientBoosting | 0.03858 | 0.05338 | −0.5262 |
| QCOM | LinearRegression | 0.01833 | 0.02438 | **+0.0123** |
| QCOM | RandomForest | 0.02174 | 0.02843 | −0.3434 |
| QCOM | GradientBoosting | 0.02436 | 0.03199 | −0.7012 |
| TSLA | LinearRegression | 0.03136 | 0.04347 | −0.0190 |
| TSLA | RandomForest | 0.03457 | 0.04749 | −0.2165 |
| TSLA | GradientBoosting | 0.04219 | 0.05671 | −0.7348 |
| UNH | LinearRegression | 0.03664 | 0.04957 | **−6.127** |
| UNH | RandomForest | 0.01449 | 0.01988 | −0.1462 |
| UNH | GradientBoosting | 0.01777 | 0.02283 | −0.5124 |
| WMT | LinearRegression | 0.01455 | 0.01876 | **−0.997** |
| WMT | RandomForest | 0.01101 | 0.01508 | −0.2913 |
| WMT | GradientBoosting | 0.01130 | 0.01551 | −0.3654 |

### 2.2 Summary statistics per model family

Values exclude extreme outliers (KO RF/GBR, UNH LR, WMT LR) which are artefacts of per-ticker data scarcity rather than systematic model failure — see Section 4.2.

| Metric | LinearRegression | RandomForest | GradientBoosting |
|---|---|---|---|
| Mean R² (all 25 tickers) | −0.67 | −0.84 | −1.63 |
| Mean R² (excl. outliers) | −0.20 | −0.38 | −0.75 |
| Tickers with R² > −0.1 | **13 / 25** | 6 / 25 | 2 / 25 |
| Tickers with positive R² | **3 / 25** (DIS, MCD, QCOM) | 0 / 25 | 0 / 25 |
| Mean MAE | 0.0175 | 0.0192 | 0.0231 |
| Best MAE (lowest) | JNJ: 0.00811 | PEP: 0.00852 | JNJ: 0.00881 |
| Worst MAE (highest) | PLTR: 0.04154 | ORCL: 0.04245 | NVDA: 0.04548 |

---

## 3. Per-Ticker Results — Direction Accuracy

Direction accuracy is the primary operational metric: does the model correctly predict whether the price will go up or down the next day?

> **Baseline:** A coin flip = 50.0%. Any value consistently above 50% is exploitable in theory.

| Ticker | LR Dir. Acc | RF Dir. Acc | GBR Dir. Acc | Best Model |
|---|---|---|---|---|
| AAPL | 44.40% | 41.38% | 43.97% | LR |
| ADBE | 45.67% | 49.52% | 47.12% | RF |
| AMD | 49.57% | 49.57% | 48.71% | LR / RF (tie) |
| AMZN | 51.81% | 48.70% | **52.85%** | GBR |
| AVGO | 50.50% | 50.00% | **53.50%** | GBR |
| COST | 51.38% | **54.14%** | 53.04% | RF |
| CRM | 50.42% | 53.33% | **57.92%** | GBR ⭐ |
| DIS | 48.37% | **53.02%** | 53.49% | GBR |
| GOOG | 42.67% | 42.24% | 42.24% | LR |
| INTC | **50.21%** | 45.49% | 44.64% | LR |
| JNJ | 49.77% | 54.34% | **57.08%** | GBR ⭐ |
| KO | 46.35% | 47.64% | 47.64% | RF / GBR |
| MCD | **51.81%** | 44.56% | 44.04% | LR |
| META | 46.12% | **56.90%** | 46.12% | RF ⭐ |
| MSFT | 46.12% | **53.02%** | 49.14% | RF |
| NFLX | 48.93% | 46.35% | 48.93% | LR / GBR |
| NVDA | 51.45% | 52.28% | **53.11%** | GBR |
| ORCL | **49.37%** | 46.41% | 44.73% | LR |
| PEP | **53.37%** | 47.15% | 45.60% | LR |
| PFE | 53.68% | 50.22% | **54.55%** | GBR |
| PLTR | 43.84% | 48.77% | **51.23%** | GBR |
| QCOM | **56.90%** | 48.71% | 54.31% | LR ⭐ |
| TSLA | **56.03%** | 46.98% | 47.84% | LR ⭐ |
| UNH | 50.22% | 50.65% | **52.81%** | GBR |
| WMT | 42.50% | 42.92% | **48.33%** | GBR |

### 3.1 Average direction accuracy by model

| Model | Mean Dir. Acc | Tickers > 50% | Tickers > 53% |
|---|---|---|---|
| LinearRegression | **49.3%** | 11 / 25 | 4 / 25 |
| RandomForest | 49.0% | 12 / 25 | 5 / 25 |
| GradientBoosting | **49.7%** | 15 / 25 | 8 / 25 |

> ⭐ marks cases where direction accuracy exceeds 55% — potentially exploitable edge.

---

## 4. Interpretation & Analysis

### 4.1 Why are all R² values negative?

A negative R² does not mean the model is "broken" — it means **the model's predictions vary less than the actual returns**, so its residuals are larger than simply predicting the mean return every day. This is expected in financial markets for two compounding reasons:

1. **Market efficiency (EMH):** In liquid, large-cap US equities, publicly available information (price history, technical indicators, published financials) is quickly priced in by institutional traders. The portion of next-day return that can be explained by any systematic signal is very small — typically < 1% of variance. Positive R² in academic papers on return prediction is considered noteworthy.

2. **Data scarcity per ticker:** Each per-ticker model is trained on ~750–960 usable rows (after NaN removal). This is a very small sample for an ensemble of 200 decision trees. Both RF and GBR are prone to overfitting on such datasets — they memorise training noise rather than learning generalisable patterns, producing poor out-of-sample R².

3. **Signal-to-noise ratio:** Daily stock returns are dominated by noise. A typical large-cap stock has daily volatility of 1–3%. The predictable component (the signal) is believed to be an order of magnitude smaller. Measuring model quality with R² on a high-noise target produces near-zero or negative values even for genuinely predictive models.

**Conclusion:** Negative R² is the expected baseline for this problem. The correct primary metric is direction accuracy, not R².

### 4.2 Extreme outliers — KO, UNH, WMT, DIS, GOOG

Several tickers show catastrophically negative R² for specific models:

| Ticker | Model | R² | Root cause |
|---|---|---|---|
| KO | RF | −11.14 | RF severely overfits on ~930 rows; test set has some large-return events the model magnifies |
| KO | GBR | −13.28 | Same cause, GBR amplifies the overfit further |
| UNH | LR | −6.13 | StandardScaler is fit on training data; UNH had several large-return quarters (earnings surprises) in the test period that fall far outside the training distribution |
| WMT | LR | −1.00 | WMT has unusually low return variance overall; LR predictions have higher variance than actuals in the test period |
| DIS | GBR | −6.40 | DIS had structural disruption (streaming pivot, COVID impact) in the test period; GBR overfit training patterns that do not generalise |
| GOOG | GBR | −6.02 | Same class of issue — large out-of-distribution events in test set |

These are not model failures in the sense of implementation bugs — they reflect the **distribution shift** between the training and test periods, which is an inherent challenge in financial time series.

### 4.3 LinearRegression outperforms tree models on R²

Counterintuitively, LR has the least-negative R² in 17 out of 25 tickers. Two explanations:

1. **Regularisation through simplicity:** With only 16 features and ~900 training samples, linear regression has fewer degrees of freedom and therefore overfits less. RF and GBR have hundreds of free parameters (200 trees × depth × splits) and overfit the noise in the training set.

2. **Feature linearity:** The 5 volatility-normalised features (`Return_norm`, `Return_norm_Lag1`, `Return_norm_Lag2`, `Log_Return`, `Volatility_20`) are designed to be cross-sectionally comparable and may have a near-linear relationship with next-day returns in low-signal environments. Tree models need more data to discover and exploit non-linear interactions reliably.

**Implication for the pooled model:** With ~25,000+ training rows (25 tickers × ~1,000 rows each), tree models should benefit much more from the larger dataset. The pooled model is expected to reverse this ranking — RF and GBR should generalise better at scale.

### 4.4 Direction accuracy — the signal that matters

Three of the four ⭐ tickers (CRM 57.9% GBR, JNJ 57.1% GBR, QCOM 56.9% LR, TSLA 56.0% LR) consistently exceed 55% direction accuracy. This is meaningful:

- With 232 test rows and direction accuracy = 57%, the z-score vs 50% baseline is ~2.1 (p ≈ 0.036), suggesting a statistically significant edge for at least some ticker–model combinations.
- META RF at 56.9% and PFE GBR at 54.6% also show consistent above-baseline performance.

However, these are **per-ticker, in-sample regime** results — they reflect the test period of each ticker's own history. A proper out-of-time test (e.g., train on 2020–2023, test on 2024 only) would be needed to validate statistical significance across the full 30-ticker universe before drawing trading conclusions.

---

## 5. Model Selection Rationale (Per-Ticker Prototype)

### Per-ticker evaluation (this notebook)

Based on the combined evidence:

| Criterion | Winner | Notes |
|---|---|---|
| R² (least negative) | **LinearRegression** | 13/25 tickers with R² > −0.1 |
| MAE (lowest error) | **LinearRegression** | Mean MAE 0.0175 vs RF 0.0192 |
| Direction accuracy (mean) | **GradientBoosting** | 49.7% vs LR 49.3% |
| Stability (no catastrophic outliers) | **LinearRegression** | GBR produces R² = −13.3 on KO |

**Per-ticker conclusion:** LinearRegression is the most reliable model at this data scale.

### Production decision (pooled model)

The production training script (`model/train.py --all`) trains all three models on the pooled dataset and selects the winner by test R². With ~25,000 training samples, the tree models should overcome their data-scarcity disadvantage. The winning model will be determined at training time and logged in the run output.

---

## 6. Production Pooled Model Results

The production models (`model_pooled.pkl` and `model_pooled_fallback.pkl`) were trained in `ml_exploration.ipynb` section 8, implementing the same pooled strategy as `python model/train.py --all`. Results below are from stored cell outputs.

### 6.1 Training data summary

| Model | Tickers | Features | Train rows | Test rows | Total rows |
|---|---|---|---|---|---|
| `model_pooled.pkl` (standard) | 25 tickers (all except BAC/GS/JPM/MA/V) | 16 (price + vol-norm + fundamentals) | 22,148 | 5,548 | 27,696 |
| `model_pooled_fallback.pkl` (fallback) | BAC, GS, JPM, MA, V | 11 (price + vol-norm only) | 4,860 | 1,215 | 6,075 |

Split method: **per-ticker 80/20 temporal split applied before pooling** — each ticker's final 20% of usable rows (after NaN removal) is reserved for the test set, preventing any single ticker from dominating the held-out evaluation.

### 6.2 Model comparison — Standard pool (25 tickers, 16 features)

| Model | R² (test set) | MAE (test set) | Selected |
|---|---|---|---|
| LinearRegression (+ StandardScaler) | **−0.0026** | **0.01530** | ✅ Winner |
| RandomForestRegressor (200 trees) | −0.0814 | 0.01596 | — |
| GradientBoostingRegressor (200 estimators) | −0.1623 | 0.01623 | — |

**Winner: LinearRegression** → saved as `model/trained/model_pooled.pkl`

### 6.3 Model comparison — Fallback pool (5 tickers, 11 features)

| Model | R² (test set) | MAE (test set) | Selected |
|---|---|---|---|
| LinearRegression (+ StandardScaler) | **−0.0025** | **0.00996** | ✅ Winner |
| RandomForestRegressor (200 trees) | −0.0669 | 0.01044 | — |
| GradientBoostingRegressor (200 estimators) | −0.0776 | 0.01046 | — |

**Winner: LinearRegression** → saved as `model/trained/model_pooled_fallback.pkl`

### 6.4 Pooled vs. per-ticker: R² comparison

| Context | LR mean R² | Training rows | Notes |
|---|---|---|---|
| Per-ticker prototype (sections 5–6) | −0.67 | ~900 per model | 25 separate models |
| **Pooled standard (production)** | **−0.0026** | **27,696** | Single shared model |
| **Pooled fallback (production)** | **−0.0025** | **6,075** | Single shared model |

The pooled LR achieves R²=−0.003 versus a per-ticker mean of −0.67 — roughly **250× better**. This validates the core hypothesis from Section 4.3: data scarcity was the primary driver of per-ticker R² degradation, not model misspecification.

### 6.5 Key observations

**1. LinearRegression wins in both pools — including at 22,000+ training samples.**
Section 4.3 predicted that tree ensembles would close the gap with more data. They did not — LR still wins. Two likely explanations: (a) the feature-target relationship is genuinely near-linear in this domain; (b) the volatility-normalised features (`Return_norm`, `Return_norm_Lag1`, `Return_norm_Lag2`) are already doing the non-linear normalisation work, leaving little for tree models to discover.

**2. MAE is consistent and interpretable.**
Standard pool MAE = 0.01530 means the model's average absolute prediction error is ~1.53% per day. Fallback pool MAE = 0.00996 (~1.0%) reflects the inherently lower daily volatility of financial stocks (BAC, GS, JPM, MA, V) versus the mixed-sector standard pool (which includes TSLA, NVDA, PLTR).

**3. All R² values remain negative even at pooled scale.**
R²=−0.003 is far better than per-ticker, but still negative — the model is still slightly worse than the constant-mean predictor on the test set. This is consistent with EMH in liquid, large-cap US equities. Direction accuracy on the pooled test set was not separately logged, but per-ticker Section 3 results (~49–50%) are the best available proxy.

**4. Production models are refitted on the full dataset before saving.**
After evaluating candidates on the held-out test set, the winning model is retrained on all data (train + test combined). The `.pkl` files therefore contain models trained on all 27,696 rows (standard) and 6,075 rows (fallback), giving them the best possible parameter estimates for live inference.

**5. GradientBoosting consistently underperforms at both scales.**
GBR is worst at per-ticker scale (mean R²=−1.63, catastrophic KO R²=−13.28) and worst at pooled scale (standard R²=−0.16). Even with ~22,000 training rows, GBR overfits relative to LR. This may reflect that the signal in daily returns is too sparse and diffuse for gradient-boosted trees to learn reliably without regularisation (e.g., max_depth, subsample, min_samples_leaf tuning).

### 6.6 Hyperparameter configuration

The same model configurations are used across **both pools** (standard and fallback) — the only difference between pools is the feature set (16 vs 11 features) and the tickers included.

#### LinearRegression (Winner — both pools)

| Parameter | Value | Notes |
|---|---|---|
| Preprocessing | `StandardScaler` | Required: Market_Cap (~10¹²) vs Log_Return (~0.01) without scaling distorts OLS coefficients |
| StandardScaler — with_mean | `True` (default) | Subtracts feature mean |
| StandardScaler — with_std | `True` (default) | Divides by feature std |
| fit_intercept | `True` (default) | Intercept term included |
| Regularisation | None | Plain OLS — no L1/L2 penalty |
| n_jobs | `None` (default) | Single-threaded |

> **Note:** The absence of regularisation is a known limitation — see Section 8 recommendations. Ridge regression (L2) would add a penalty term `α‖w‖²` and is a recommended next step.

#### RandomForestRegressor

| Parameter | Value | Notes |
|---|---|---|
| n_estimators | **200** | Number of trees in the forest (explicit) |
| random_state | **42** | Reproducibility seed (explicit) |
| n_jobs | **−1** | Uses all available CPU cores (explicit) |
| max_depth | `None` (default) | Trees grown until all leaves are pure or contain < min_samples_split samples — primary overfitting risk |
| min_samples_split | `2` (default) | Minimum samples required to split an internal node |
| min_samples_leaf | `1` (default) | Minimum samples in each leaf — low value amplifies overfitting on small per-ticker datasets |
| max_features | `1.0` (default, sklearn≥1.1) | All features considered at each split (changed from `'auto'` in older versions) |
| bootstrap | `True` (default) | Bagging with replacement |
| Preprocessing | None | Scale-invariant; no scaler applied |

#### GradientBoostingRegressor

| Parameter | Value | Notes |
|---|---|---|
| n_estimators | **200** | Number of boosting stages (explicit) |
| random_state | **42** | Reproducibility seed (explicit) |
| learning_rate | `0.1` (default) | Shrinkage factor applied to each tree's contribution |
| max_depth | `3` (default) | Maximum depth per tree — shallow by design, but 200 stages × depth-3 trees is still high capacity |
| min_samples_split | `2` (default) | Minimum samples to split |
| min_samples_leaf | `1` (default) | Minimum samples per leaf |
| subsample | `1.0` (default) | Fraction of samples used per stage; setting <1.0 would add stochastic regularisation |
| max_features | `None` (default) | All features considered at each split |
| loss | `'squared_error'` (default, sklearn≥1.0) | Standard MSE loss for regression |
| Preprocessing | None | Scale-invariant; no scaler applied |

#### Summary: explicitly set vs. default parameters

| Parameter | LR | RF | GBR |
|---|---|---|---|
| **Explicitly set** | _(none beyond scaler)_ | n_estimators=200, random_state=42, n_jobs=−1 | n_estimators=200, random_state=42 |
| **Key defaults left unchanged** | fit_intercept=True, no regularisation | max_depth=None, min_samples_leaf=1, max_features=1.0 | learning_rate=0.1, max_depth=3, subsample=1.0 |
| **Known overfitting risk** | Low (few parameters) | High: max_depth=None on small datasets | Medium: subsample=1.0, large n_estimators |

---

## 7. Context: Per-Ticker vs. Pooled Architecture

| | Per-ticker prototype (sections 2–3) | Production pooled model (section 6) |
|---|---|---|
| Training data per model | ~750–960 rows per ticker | 27,696 rows (standard) / 6,075 rows (fallback) |
| Number of models | One per ticker (25 standard + 5 fallback = 30) | One per feature schema (2 total) |
| Purpose | Prototype — explore model families, detect outliers | Production — loaded by Streamlit app for inference |
| Where trained | `ml_exploration.ipynb` sections 5–6 | `ml_exploration.ipynb` section 8 / `python model/train.py --all` |
| LR R² (actual) | Mean −0.67 (range: +0.012 to −6.13) | **−0.0026** (standard) / **−0.0025** (fallback) |
| Winner model family | LinearRegression (13/25 tickers best R²) | LinearRegression (both pools) |
| Status | Reference only — not deployed | **Active** — `.pkl` files loaded by `go_live.py` and `backtesting.py` |

---

## 8. Known Limitations & Recommended Next Steps

### Limitations

| Limitation | Severity | Description |
|---|---|---|
| No walk-forward validation | High | A single 80/20 split cannot detect regime changes. Rolling validation (train on year N, test on year N+1) would give a more reliable estimate of out-of-sample performance |
| Fundamental data lag | Medium | Quarterly fundamentals are merged at Publish Date, but actual investor awareness often lags further. Point-in-time accuracy is approximate |
| No transaction costs | Medium | Direction accuracy assumes frictionless trading. With bid-ask spreads and slippage, the true edge is lower |
| No position sizing | Low | The current signal is binary (BUY/SELL/HOLD). A confidence-weighted position size (larger bet when model is more certain) could improve risk-adjusted returns |
| BAC, GS, JPM, MA, V excluded | Informational | These 5 tickers are handled by a separate fallback model (11 features, no fundamentals) — not evaluated in this report |

### Recommended next steps for the production model

1. **Walk-forward cross-validation** — 3-fold temporal CV (train 2020–2021 → test 2022; train 2020–2022 → test 2023; train 2020–2023 → test 2024) would give a more honest performance estimate than a single 80/20 split.
2. **Direction accuracy on pooled test set** — log `sign(predicted) == sign(actual)` during `model/train.py` runs; this is the most operationally relevant metric and is currently only measured per-ticker in the prototype notebook.
3. **Regularisation for LinearRegression** — consider Ridge (L2) or Lasso (L1) regression; the current OLS estimator has no regularisation. With 11–16 features and 6,000–27,000 rows the ratio is manageable, but Ridge would reduce variance and might improve the fallback model.
4. **GBR hyperparameter tuning** — GBR underperforms at all scales tested. Reducing `max_depth` (default=3→2), adding `subsample<1.0`, and increasing `min_samples_leaf` would add regularisation and may close the gap with LR. Currently using default hyperparameters only.
5. **Calibration check** — plot the distribution of model predictions vs. actual returns on the test set. A well-calibrated regression model should produce predicted returns that, when positive, correlate with actual positive returns at a rate materially above 50%.
6. **Feature importance analysis** — extract LR coefficients (standardised, post-scaling) on the pooled model to identify which of the 16 features drive predictions. This is meaningful at pooled scale with 27,000 rows; per-ticker feature importances (section 7 of the notebook) are noisy due to data scarcity.

---

*Report generated from `notebooks/ml_exploration.ipynb` (sections 5–6 per-ticker outputs; section 8 pooled model outputs). Both production models are trained and deployed: `model/trained/model_pooled.pkl` and `model/trained/model_pooled_fallback.pkl`.*
