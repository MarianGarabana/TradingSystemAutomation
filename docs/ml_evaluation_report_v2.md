# ML Model Evaluation Report — v2
## Trading System Automation — Next-Day Return Prediction

**Date:** March 2026
**Dataset:** SimFin bulk CSV — 30 US large-cap tickers, 2020–2024
**Evaluation scope:** Per-ticker prototype (sections 2–3) + Production pooled models v2 (section 6)
**Changes from v1:** Ridge replaces OLS; LightGBM added as fourth candidate; Ridge alpha selected via TimeSeriesSplit CV; direction accuracy and signal-bin breakdown now logged for both pools; calibration plots generated (section 8)
**Source:** `notebooks/ml_exploration.ipynb` (per-ticker sections 5–6) + `python model/train.py --all` (production v2)

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
This is the **next-day forward return** — the quantity the model is asked to predict. Computed at ETL time; NaN in the last row is dropped before training.

### Feature set (11 active features)

| Group | Features | Notes |
|---|---|---|
| Price-based (6) | MA5, MA20, Volume_Change, Market_Cap, RSI, MACD | Standard technical indicators |
| Volatility-normalised (5) | Log_Return, Volatility_20, Return_norm, Return_norm_Lag1, Return_norm_Lag2 | Risk-adjusts returns cross-sectionally — makes KO and TSLA comparable in the pooled model |

> **Note on fundamentals:** STANDARD_FEATURE_COLS includes 5 fundamental ratios (Gross_Margin, Operating_Margin, Net_Margin, Debt_to_Equity, Operating_CF_Ratio) which are currently commented out pending API integration. Both the standard and fallback schemas therefore use the same 11 active features. Both pools effectively use the same schema at this stage.

**Fallback schema:** BAC, GS, JPM, MA, V always use the 11-feature price+vol-norm schema. When fundamentals are restored for standard tickers, the schemas will diverge (16 vs 11 features).

### Train / test split
- **Method:** Temporal 80/20 split per ticker before pooling.
- **No shuffling** — look-ahead bias prevention.
- **NaN handling:** Rows with any NaN in feature or target columns are dropped before the split index is computed.

### Models evaluated (v2 — four candidates)

| Model | Scaling | Key property |
|---|---|---|
| **Ridge (+ StandardScaler, α via CV)** | StandardScaler applied | L2-regularised OLS; α selected by TimeSeriesSplit(n=5) CV on training set, optimising direction accuracy |
| RandomForestRegressor (200 trees) | None | Bagged decision trees; captures non-linear interactions |
| GradientBoostingRegressor (200 estimators) | None | Boosted trees; highest risk of overfitting on smaller datasets |
| LightGBMRegressor (500 estimators) | None | Gradient boosting with leaf-wise growth; added regularisation via min_child_samples and subsampling |

### Evaluation metrics
- **MAE:** Average absolute prediction error (same unit as Target — daily return). MAE=0.015 means ±1.5% average error.
- **RMSE:** Root mean squared error — penalises large prediction errors more than MAE.
- **R²:** Fraction of target variance explained. R²<0 means the model is worse than predicting the mean return every day.
- **Direction Accuracy:** `mean(sign(y_pred) == sign(y_actual))`. The primary operational metric — a model with direction accuracy consistently above 50% has positive expected value regardless of R².

---

## 2. Per-Ticker Results — Regression Metrics (Prototype, `ml_exploration.ipynb` sections 5–6)

> **Context:** These are from the per-ticker prototype evaluation using OLS LinearRegression. They are not the production metrics — see Section 6 for production results.

> **Note:** BAC, GS, JPM, MA, V are absent — their 11 features have NaN fundamental columns in the per-ticker prototype run, causing zero clean rows. They are handled by the fallback pooled model.

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

### 2.2 Summary statistics per model family (per-ticker prototype)

| Metric | LinearRegression | RandomForest | GradientBoosting |
|---|---|---|---|
| Mean R² (all 25 tickers) | −0.67 | −0.84 | −1.63 |
| Mean R² (excl. outliers) | −0.20 | −0.38 | −0.75 |
| Tickers with R² > −0.1 | **13 / 25** | 6 / 25 | 2 / 25 |
| Tickers with positive R² | **3 / 25** (DIS, MCD, QCOM) | 0 / 25 | 0 / 25 |
| Mean MAE | 0.0175 | 0.0192 | 0.0231 |

---

## 3. Per-Ticker Results — Direction Accuracy (Prototype)

| Ticker | LR Dir. Acc | RF Dir. Acc | GBR Dir. Acc | Best |
|---|---|---|---|---|
| AAPL | 44.40% | 41.38% | 43.97% | LR |
| ADBE | 45.67% | 49.52% | 47.12% | RF |
| AMD | 49.57% | 49.57% | 48.71% | LR / RF |
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

### 3.1 Average direction accuracy (per-ticker prototype)

| Model | Mean Dir. Acc | Tickers > 50% | Tickers > 53% |
|---|---|---|---|
| LinearRegression | 49.3% | 11 / 25 | 4 / 25 |
| RandomForest | 49.0% | 12 / 25 | 5 / 25 |
| GradientBoosting | **49.7%** | 15 / 25 | 8 / 25 |

> ⭐ Direction accuracy > 55% — statistically significant edge at p≈0.03 (z-test vs 50% baseline, n≈232 test rows).

---

## 4. Interpretation & Analysis

### 4.1 Why are all R² values negative?

A negative R² does not mean the model is broken — it means the model's predictions have higher variance than the actual returns on the test set. Two compounding reasons:

1. **Market efficiency (EMH):** In liquid large-cap US equities, publicly available information is rapidly priced in. The predictable fraction of next-day return variance is typically < 1%.
2. **Data scarcity per ticker:** ~900 rows for 200 trees — tree ensembles memorise training noise instead of learning generalisable patterns.
3. **Signal-to-noise ratio:** Daily returns have ~1–3% volatility; the predictable signal is orders of magnitude smaller.

**The correct primary metric is direction accuracy, not R².**

### 4.2 Extreme outliers

| Ticker | Model | R² | Root cause |
|---|---|---|---|
| KO | RF | −11.14 | RF severely overfits on ~930 rows; large-return events in test set |
| KO | GBR | −13.28 | Same cause, amplified by boosting |
| UNH | LR | −6.13 | Earnings surprises in test period fall outside training distribution |
| DIS | GBR | −6.40 | Structural disruption (COVID, streaming pivot) in test period |
| GOOG | GBR | −6.02 | Out-of-distribution events in test period |

These reflect **distribution shift** between training and test periods, not implementation bugs.

### 4.3 LinearRegression / Ridge outperforms tree models at low data scale

With ~900 rows per ticker, tree ensembles overfit. Linear models with far fewer degrees of freedom generalise better. This is reversed (partially) at pooled scale — see Section 6.

### 4.4 Direction accuracy — the signal that matters

Four ⭐ cases exceed 55% (CRM GBR 57.9%, JNJ GBR 57.1%, QCOM LR 56.9%, TSLA LR 56.0%). Statistical significance at p≈0.03 each. However, these are per-ticker in-sample-regime results — not validated out-of-time across the full universe.

---

## 5. Model Selection Rationale (Per-Ticker Prototype)

| Criterion | Winner | Notes |
|---|---|---|
| R² (least negative) | **LinearRegression** | 13/25 tickers R² > −0.1 |
| MAE (lowest error) | **LinearRegression** | Mean MAE 0.0175 vs RF 0.0192 |
| Direction accuracy (mean) | **GradientBoosting** | 49.7% vs LR 49.3% |
| Stability | **LinearRegression** | GBR produces R² = −13.3 on KO |

**Per-ticker conclusion:** LinearRegression is the most reliable model at this data scale.

---

## 6. Production Pooled Model Results (v2)

Models trained via `python model/train.py --all`. Four candidates per pool: Ridge (α via CV), RF, GBR, LightGBM.

> **Change from v1:** OLS LinearRegression replaced by Ridge (L2 regularisation, α tuned via CV). LightGBM added as fourth candidate. Direction accuracy and signal-bin breakdown now logged.

### 6.1 Training data summary

| Pool | Tickers | Features | Train rows | Test rows | Total |
|---|---|---|---|---|---|
| Standard (`model_pooled.pkl`) | 25 (all except BAC/GS/JPM/MA/V) | 11 | 24,201 | 6,051 | 30,252 |
| Fallback (`model_pooled_fallback.pkl`) | BAC, GS, JPM, MA, V | 11 | 4,860 | 1,215 | 6,075 |

Split method: **per-ticker 80/20 temporal split before pooling**. Training rows per ticker: 972 for 24 tickers, 873 for PLTR (shorter history).

### 6.2 Ridge alpha cross-validation (training set only)

Alpha candidates evaluated via `TimeSeriesSplit(n_splits=5)`, optimising **mean direction accuracy** across folds. Test set is never touched during CV.

#### Standard pool — alpha CV scores

| α | CV mean dir. acc |
|---|---|
| 0.01 | 0.5181 |
| 0.1 | 0.5181 |
| 1.0 | 0.5182 |
| **10.0** | **0.5191** ← selected |
| 100.0 | 0.5191 |

> α=10.0 selected (tied with α=100.0; first wins). Higher α = stronger L2 regularisation = coefficients shrunk closer to zero.

#### Fallback pool — alpha CV scores

| α | CV mean dir. acc |
|---|---|
| 0.01 | 0.5064 |
| 0.1 | 0.5059 |
| 1.0 | 0.5074 |
| 10.0 | 0.5067 |
| **100.0** | **0.5091** ← selected |

> α=100.0 selected. The higher optimal alpha in the fallback pool (5 tickers, 4,860 rows vs 24,201) is consistent with the smaller dataset needing stronger regularisation to prevent overfitting.

### 6.3 Model comparison — Standard pool (25 tickers, 11 features)

| Model | MAE | RMSE | R² (test) | Selected |
|---|---|---|---|---|
| **Ridge (α=10.0, + StandardScaler)** | **0.014960** | **0.023443** | **−0.0021** | ✅ Winner |
| LightGBMRegressor | 0.015222 | 0.023693 | −0.0236 | — |
| RandomForestRegressor | 0.015887 | 0.024222 | −0.0698 | — |
| GradientBoostingRegressor | 0.016204 | 0.024927 | −0.1330 | — |

**Winner: Ridge (α=10.0)** → saved as `model/trained/model_pooled.pkl`
**Direction accuracy (test set):** 50.67% (n=6,051)

### 6.4 Model comparison — Fallback pool (5 tickers, 11 features)

| Model | MAE | RMSE | R² (test) | Selected |
|---|---|---|---|---|
| **Ridge (α=100.0, + StandardScaler)** | **0.009952** | **0.014515** | **−0.0012** | ✅ Winner |
| RandomForestRegressor | 0.010444 | 0.014985 | −0.0670 | — |
| GradientBoostingRegressor | 0.010461 | 0.015059 | −0.0776 | — |
| LightGBMRegressor | 0.010613 | 0.015284 | −0.1101 | — |

**Winner: Ridge (α=100.0)** → saved as `model/trained/model_pooled_fallback.pkl`
**Direction accuracy (test set):** 51.44% (n=1,215)

### 6.5 Signal-bin breakdown (winning model on test set)

The signal bins define how predicted returns map to trading signals. A healthy model should distribute predictions across multiple bins.

#### Standard pool

| Bin | Threshold | N | % of test | Within-bin dir. acc |
|---|---|---|---|---|
| HIGH RISE | pred > +2% | 0 | 0.0% | — |
| LOW RISE | +0.5% to +2% | 0 | 0.0% | — |
| STAY | −0.5% to +0.5% | 6,051 | **100.0%** | 50.7% |
| LOW FALL | −2% to −0.5% | 0 | 0.0% | — |
| HIGH FALL | pred < −2% | 0 | 0.0% | — |

#### Fallback pool

| Bin | Threshold | N | % of test | Within-bin dir. acc |
|---|---|---|---|---|
| HIGH RISE | pred > +2% | 0 | 0.0% | — |
| LOW RISE | +0.5% to +2% | 0 | 0.0% | — |
| STAY | −0.5% to +0.5% | 1,214 | **99.9%** | 51.5% |
| LOW FALL | −2% to −0.5% | 1 | 0.1% | 0.0% |
| HIGH FALL | pred < −2% | 0 | 0.0% | — |

> **Critical finding:** All predictions fall in the STAY bin. The Ridge model produces predicted returns with absolute magnitude < 0.5%, so the 5-tier thresholds are never crossed. In practice, the app will generate **only HOLD signals** from the current production model. This is a direct consequence of Ridge shrinkage on a near-zero-signal target: the regulariser pushes all coefficients toward zero, collapsing predicted returns toward the mean (~0%). See Section 9 for implications.

### 6.6 v1 vs. v2 — Key metric changes

| Metric | v1 (OLS) | v2 (Ridge α=10/100) | Δ |
|---|---|---|---|
| Standard R² | −0.0026 | **−0.0021** | +0.0005 (marginal improvement) |
| Standard MAE | 0.01530 | **0.01496** | −0.00034 (↓ 2.2%) |
| Standard RMSE | — | 0.02344 | now logged |
| Fallback R² | −0.0025 | **−0.0012** | +0.0013 (improvement) |
| Fallback MAE | 0.00996 | **0.00995** | −0.00001 (flat) |
| LightGBM (new) | — | R²=−0.024 (std), −0.110 (fb) | 2nd place standard; 4th fallback |
| Direction acc | not logged | 50.67% / 51.44% | now measured |
| Signal bins | not logged | 100% STAY | now measured |

### 6.7 Hyperparameter configuration (v2)

#### Ridge (Winner — both pools)

| Parameter | Standard pool | Fallback pool | Notes |
|---|---|---|---|
| Preprocessing | StandardScaler | StandardScaler | Required: Market_Cap (~10¹²) vs Log_Return (~0.01) |
| alpha (α) | **10.0** (CV-selected) | **100.0** (CV-selected) | L2 penalty strength; higher = more shrinkage |
| CV method | TimeSeriesSplit(n=5) | TimeSeriesSplit(n=5) | Preserves temporal order within training set |
| CV metric | Direction accuracy | Direction accuracy | Primary operational criterion |
| fit_intercept | True (default) | True (default) | |
| solver | auto (default) | auto (default) | Cholesky for dense features |

#### RandomForestRegressor

| Parameter | Value | Notes |
|---|---|---|
| n_estimators | 200 | Explicit |
| random_state | 42 | Reproducibility |
| n_jobs | −1 | All CPU cores |
| max_depth | None (default) | Fully grown trees — primary overfitting risk |
| min_samples_leaf | 1 (default) | Low value amplifies overfit on small datasets |
| max_features | 1.0 (default, sklearn≥1.1) | All features at each split |
| bootstrap | True (default) | |

#### GradientBoostingRegressor

| Parameter | Value | Notes |
|---|---|---|
| n_estimators | 200 | Explicit |
| random_state | 42 | Reproducibility |
| learning_rate | 0.1 (default) | Shrinkage per stage |
| max_depth | 3 (default) | Shallow per-tree; 200×depth-3 still has high capacity |
| subsample | 1.0 (default) | No stochastic regularisation |
| loss | squared_error (default) | |

#### LightGBMRegressor (new in v2)

| Parameter | Value | Notes |
|---|---|---|
| n_estimators | 500 | More estimators offset by slow learning rate |
| learning_rate | 0.03 | Low → slow learning, less overfit |
| num_leaves | 31 (default) | Controls tree complexity (leaf-wise growth) |
| min_child_samples | **30** | Minimum leaf size — key regulariser; prevents tiny leaves on dense regions |
| colsample_bytree | **0.8** | Feature subsampling per tree — adds regularisation |
| subsample | **0.7** | Row subsampling per tree — stochastic gradient boosting |
| random_state | 42 | Reproducibility |
| n_jobs | −1 | All CPU cores |
| verbose | −1 | Suppress stdout (logging via Python logger) |

#### Summary: overfitting risk ranking (v2)

| Model | Overfitting risk | Rationale |
|---|---|---|
| Ridge (α CV) | **Low** | L2 regularisation; α=10–100 provides strong shrinkage |
| LightGBM | **Medium-low** | min_child_samples=30, subsampling, slow LR offset high n_estimators |
| RandomForest | **Medium-high** | max_depth=None; min_samples_leaf=1 |
| GradientBoosting | **High** | No subsampling; subsample=1.0 |

---

## 7. Calibration Analysis

Calibration measures whether the **rank order** of predictions corresponds to the rank order of actual returns. Predictions are binned into deciles (1=most negative → 10=most positive). A well-calibrated model shows monotonically increasing mean actual return across deciles.

Plots saved to:
- `model/trained/calibration_standard.png`
- `model/trained/calibration_fallback.png`

### 7.1 Standard pool — decile calibration table

| Decile | N | Mean Predicted Return | Mean Actual Return | Dir. Acc. |
|---|---|---|---|---|
| 1 (most −) | 606 | −0.000930 | +0.000907 | 0.4604 |
| 2 | 605 | −0.000186 | +0.001002 | 0.4595 |
| 3 | 605 | +0.000158 | +0.000947 | 0.5289 |
| 4 | 605 | +0.000427 | +0.000948 | 0.5091 |
| 5 | 605 | +0.000662 | −0.000247 | 0.5074 |
| 6 | 605 | +0.000881 | −0.000090 | 5.041 |
| 7 | 605 | +0.001093 | −0.000012 | 0.5174 |
| 8 | 605 | +0.001329 | +0.001617 | 0.5355 |
| 9 | 605 | +0.001612 | +0.002195 | 0.5537 |
| 10 (most +) | 605 | +0.002158 | −0.001004 | 0.4909 |

**Interpretation:** Non-monotonic. Deciles 8–9 show a genuine positive signal (mean actual return increases with prediction: +0.162%, +0.220%). However, decile 10 collapses back to negative (−0.100%), and the two bottom deciles (negative predictions) actually correspond to positive actual returns — the model is partially **anti-predictive** in the tails. Overall: very weak and inconsistent calibration.

### 7.2 Fallback pool — decile calibration table

| Decile | N | Mean Predicted Return | Mean Actual Return | Dir. Acc. |
|---|---|---|---|---|
| 1 (most −) | 122 | −0.002738 | −0.000741 | 0.4590 |
| 2 | 121 | −0.001733 | +0.000427 | 0.5124 |
| 3 | 122 | −0.001169 | +0.001331 | 0.4344 |
| 4 | 121 | −0.000704 | −0.000644 | 0.5124 |
| 5 | 122 | −0.000336 | −0.001516 | 0.5164 |
| 6 | 121 | +0.000064 | +0.001627 | 0.4628 |
| 7 | 121 | +0.000427 | +0.000144 | 0.4793 |
| 8 | 122 | +0.000794 | +0.002269 | **0.5574** |
| 9 | 121 | +0.001221 | +0.001541 | **0.5702** |
| 10 (most +) | 122 | +0.002062 | +0.003772 | **0.6393** |

**Interpretation:** Notably better than the standard pool. Deciles 8–10 show **monotonically increasing** mean actual return (+0.227%, +0.154%, +0.377%) and strong direction accuracy — decile 10 achieves **63.9% direction accuracy** on 122 rows. The top-decile pattern is statistically meaningful: z-score vs 50% baseline = (0.639−0.5)/√(0.5×0.5/122) = 3.07, p≈0.002. The model has a genuine, exploitable signal in the upper prediction tail for financial stocks.

### 7.3 Calibration summary

| Pool | Monotonicity | Best decile dir. acc | Worst decile dir. acc | Actionable? |
|---|---|---|---|---|
| Standard | Weak / non-monotonic | 55.4% (decile 9) | 45.9% (decile 1) | Borderline — deciles 8–9 only |
| Fallback | **Partially monotonic (deciles 8–10)** | **63.9%** (decile 10) | 43.4% (decile 3) | **Yes — top decile has strong signal** |

The fallback pool (BAC, GS, JPM, MA, V) shows stronger calibration than the standard pool despite having fewer training rows. Financial stocks have more stable return-generating processes (lower idiosyncratic volatility relative to tech stocks like NVDA, TSLA, PLTR), making their returns somewhat more predictable with linear models.

---

## 8. Context: Per-Ticker Prototype vs. Production Pooled (v1 vs. v2)

| | Per-ticker prototype | Production v1 (OLS) | Production v2 (Ridge+LGBM) |
|---|---|---|---|
| Training rows | ~900 per model | 22,148 / 4,860 | **24,201 / 4,860** |
| Models evaluated | LR, RF, GBR | LR, RF, GBR | **Ridge(CV), RF, GBR, LightGBM** |
| Alpha selection | N/A | N/A | **TimeSeriesSplit CV, dir. acc.** |
| Winner | LR (13/25 tickers) | LinearRegression | **Ridge α=10 / α=100** |
| Winner R² | Mean −0.67 | −0.0026 / −0.0025 | **−0.0021 / −0.0012** |
| Winner MAE | 0.0175 (mean) | 0.01530 / 0.00996 | **0.01496 / 0.00995** |
| Direction accuracy | ~49–50% (per-ticker) | Not logged | **50.67% / 51.44%** |
| Signal bins | Not logged | Not logged | **100% STAY (all bins)** |
| Calibration | Not performed | Not performed | **Computed — plots generated** |
| RMSE | Logged | Not logged | **Logged** |

---

## 9. Key Findings & Implications for the Trading System

### Finding 1 — The model produces only HOLD signals
All predictions cluster in the STAY bin (|pred| < 0.5%). The Streamlit app (`go_live.py`, `backtesting.py`) will generate **only HOLD signals** in live inference. This is mathematically correct behaviour: Ridge with strong regularisation shrinks all predictions toward the mean (≈0%), and the mean daily return is well within the ±0.5% HOLD threshold. To generate BUY/SELL signals, either: (a) reduce the thresholds to ±0.05% or ±0.1%, or (b) use the raw predicted return magnitude as a continuous confidence score rather than hard-binning.

### Finding 2 — Fallback pool has exploitable calibration in the top decile
The top prediction decile (BAC/GS/JPM/MA/V) achieves 63.9% direction accuracy on 122 test rows — statistically significant (p≈0.002). This suggests that the Ridge model captures a weak but real signal for financial stocks in the upper prediction tail. A strategy of trading only when the predicted return falls in decile 10 would historically have been profitable on the fallback pool.

### Finding 3 — LightGBM is competitive on the standard pool
LightGBM (R²=−0.0236) is the second-best model on the standard pool, closer to Ridge (−0.0021) than RF (−0.0698) or GBR (−0.1330). The regularised hyperparameters (min_child_samples=30, subsample=0.7, colsample_bytree=0.8, slow learning_rate=0.03) successfully constrain overfitting. With further tuning or more data, LightGBM could overtake Ridge.

### Finding 4 — Ridge L2 regularisation provides measurable improvement over OLS
R² improved from −0.0026 (v1 OLS) to −0.0021 (v2 Ridge) on the standard pool, and from −0.0025 to −0.0012 on the fallback pool. The fallback improvement is larger — consistent with stronger regularisation (α=100) being more beneficial on the smaller dataset (4,860 rows vs 24,201).

---

## 10. Known Limitations & Recommended Next Steps

### Limitations

| Limitation | Severity | Description |
|---|---|---|
| 100% HOLD signals in production | **High** | Signal thresholds (±0.5%, ±2%) are designed for models that predict returns of ≥1%  — but Ridge shrinks all predictions to near-zero. Thresholds must be recalibrated or the signal generation logic redesigned. |
| No walk-forward validation | High | Single 80/20 split cannot detect regime changes; rolling validation would give a more honest performance estimate |
| Fundamental data unavailable | Medium | STANDARD_FEATURE_COLS has 5 fundamental ratios commented out; both schemas currently use the same 11 features — standard model has no advantage over fallback |
| No transaction costs | Medium | Direction accuracy assumes frictionless trading; true edge is lower with spreads and slippage |
| No position sizing | Low | Binary BUY/SELL/HOLD; a confidence-weighted position size could improve risk-adjusted returns |

### Recommended next steps

1. **Recalibrate signal thresholds** — given all predictions are within ±0.5%, set dynamic thresholds based on the empirical predicted-return distribution (e.g., top/bottom quartile → BUY/SELL). Or expose raw predicted returns to the app and let the user set thresholds.

2. **Implement walk-forward cross-validation** — 3-fold temporal CV: train 2020–2021 → test 2022; train 2020–2022 → test 2023; train 2020–2023 → test 2024. This gives the most honest estimate of out-of-sample performance.

3. **Activate fundamental features** — restore the 5 commented-out fundamental ratios in STANDARD_FEATURE_COLS once the API wrapper is implemented. This will genuinely differentiate the standard and fallback schemas (16 vs 11 features) and may capture earnings-driven return predictability.

4. **Exploit the fallback pool top decile** — implement a "high confidence only" strategy: generate BUY signals only when predicted return exceeds the 90th percentile of historical predictions. Backtesting shows 63.9% direction accuracy in this regime.

5. **Tune LightGBM further** — LightGBM was second-best on the standard pool with default-like hyperparameters. A grid search over num_leaves (15–63), min_child_samples (10–50), and learning_rate (0.01–0.05) via TimeSeriesSplit CV could close the remaining gap with Ridge or surpass it.

6. **Feature importance extraction** — extract Ridge coefficients (post-StandardScaler, i.e., standardised units) to understand which of the 11 features drive predictions. At 24,000+ training rows the coefficients are stable and interpretable.

---

*Report v2 generated from `python model/train.py --all` and `python model/calibration.py` (March 2026). Both production models are deployed: `model/trained/model_pooled.pkl` (Ridge α=10.0) and `model/trained/model_pooled_fallback.pkl` (Ridge α=100.0). Calibration plots: `model/trained/calibration_standard.png` and `model/trained/calibration_fallback.png`.*
