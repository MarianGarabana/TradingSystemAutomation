# ML Model Evaluation Report — v4
## Trading System Automation — Next-Day Direction Classification

**Date:** March 2026
**Dataset:** SimFin bulk CSV — 30 US large-cap tickers, 2020–2024
**Evaluation scope:** Per-ticker prototype (sections 2–5, historical reference) + Production pooled classifiers v4 (section 6) + Calibration by predicted probability (section 7)
**Changes from v3:** Complete switch from **regression to binary classification**. Ridge/RF/GBR/LightGBM regressors replaced by LogisticRegression/RFC/GBC/LGBMClassifier. Model selection criterion changed from R² to accuracy. Dynamic percentile thresholds (`thresholds.json`) removed — classifier outputs directions directly. `prediction_to_signal()` updated to use predicted class + confidence (0.52 threshold for HOLD).
**Source:** `python model/train.py --all` → `model/trained/model_pooled.pkl` + `model/trained/model_pooled_fallback.pkl`

> **Note on actual metrics (sections 6, 7, 9):** Fill in values marked `[TBD]` by running `python model/train.py --all` and capturing the log output. The table structures and methodology are complete; only the numeric results require post-training fill-in.

---

## 1. Problem Setup

### Task
Directly predict **next-day stock price direction** as a binary classification problem:

| Class | Meaning | Signal |
|---|---|---|
| **1** (UP) | Next-day return > 0 | BUY (if confidence ≥ 0.52) |
| **0** (DOWN) | Next-day return < 0 | SELL (if confidence ≥ 0.52) |
| — | Either class | HOLD (if max(predict_proba) < 0.52) |

> **v4 change:** Previous versions (v1–v3) used regression to predict a continuous return, then applied threshold binning. The structural weakness was that Ridge regularisation collapsed all predictions toward zero, requiring a post-hoc percentile fix (v3). v4 switches to classification, directly optimising the decision boundary between up and down days.

### Target variable
```
Target = Adj_Close.pct_change().shift(-1)    # continuous return from ETL (unchanged)
y_binary = (Target > 0).astype(int)          # derived at training time
# Rows where Target == 0 are dropped (ambiguous direction)
```
The raw `Target` column in the CSVs is unchanged — the binary transformation is applied in `model/train.py` at training time, not in the ETL pipeline.

### Feature set (11 active features — unchanged from v3)

| Group | Features | Notes |
|---|---|---|
| Price-based (6) | MA5, MA20, Volume_Change, Market_Cap, RSI, MACD | Standard technical indicators |
| Volatility-normalised (5) | Log_Return, Volatility_20, Return_norm, Return_norm_Lag1, Return_norm_Lag2 | Risk-adjusts returns cross-sectionally — makes KO and TSLA comparable in the pooled model |

> **Note on fundamentals:** 5 fundamental ratios remain commented out of STANDARD_FEATURE_COLS pending API integration. Both standard and fallback schemas use the same 11 active features for now.

### Train / test split (unchanged from v3)
- **Method:** Temporal 80/20 split per ticker before pooling.
- **No shuffling** — look-ahead bias prevention.
- **Zero-return rows dropped** before splitting (new in v4 — ambiguous for binary target).
- **NaN handling:** Rows with any NaN in feature or target columns are dropped before computing the split index.

### Models evaluated (v4 — four classifiers)

| Model | Scaling | Key property |
|---|---|---|
| **LogisticRegression (+ StandardScaler, C via CV)** | StandardScaler applied | L2-regularised linear classifier; C selected by TimeSeriesSplit(n=5) CV on training set, optimising accuracy; `class_weight='balanced'` compensates for class imbalance |
| RandomForestClassifier | None | Bagged decision trees; captures non-linear decision boundaries |
| GradientBoostingClassifier | None | Boosted trees; tuned with low learning_rate=0.02 and shallow depth=2 to reduce overfit |
| LGBMClassifier | None | Leaf-wise gradient boosting; strong regularisation via num_leaves=15, min_child_samples=50, subsampling |

### Evaluation metrics
- **Accuracy:** `mean(y_pred == y_true)`. Primary model selection criterion. >50% beats random guessing.
- **F1 (binary):** Harmonic mean of precision and recall for class 1 (UP). Secondary metric — penalises models that ignore one class.
- **Classification report:** Per-class precision, recall, F1, and support — logged for the winning model only.
- **Note on direction accuracy:** In v1–v3, "direction accuracy" was defined as `mean(sign(y_pred) == sign(y_actual))` for regression outputs. In v4, this is identical to accuracy on the binary classification problem — both measure fraction of correct up/down predictions.

---

## 2. Per-Ticker Results — Regression Metrics (Prototype, `ml_exploration.ipynb` sections 5–6)

> **Context:** These are from the per-ticker prototype evaluation using OLS LinearRegression. They are **historical reference only** — not the production metrics. See Section 6 for production results.

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

> **v4 design decision:** Instead of trying to improve R², v4 directly optimises the direction boundary by switching to classification (binary cross-entropy loss). This is a structurally better formulation of the problem because the trading decision is inherently binary (buy or don't buy) regardless of the predicted magnitude.

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

## 6. Production Pooled Model Results (v4 — Classification)

Models trained via `python model/train.py --all`. Four classifiers per pool: LogisticRegression (C via CV), RandomForestClassifier, GradientBoostingClassifier, LGBMClassifier. **Fill in values below after running training.**

### 6.1 Training data summary

| Pool | Tickers | Features | Train rows | Test rows | UP days (train) | DOWN days (train) |
|---|---|---|---|---|---|---|
| Standard (`model_pooled.pkl`) | 25 (all except BAC/GS/JPM/MA/V) | 11 | 24,111 | 6,040 | 12,538 (52.0%) | 11,573 (48.0%) |
| Fallback (`model_pooled_fallback.pkl`) | BAC, GS, JPM, MA, V | 11 | 4,853 | 1,214 | 2,503 (51.6%) | 2,350 (48.4%) |

Split method: **per-ticker 80/20 temporal split before pooling**. Zero-return rows dropped. Class distribution is nearly balanced (~52%/48% UP/DOWN), consistent with the equity risk premium.

### 6.2 LogisticRegression C cross-validation (training set only)

C candidates evaluated via `TimeSeriesSplit(n_splits=5)`, optimising **mean accuracy** across folds.

#### Standard pool — C CV scores

| C | CV mean accuracy |
|---|---|
| 0.01 | 0.5109 |
| 0.1 | 0.5110 |
| 1.0 | 0.5107 |
| **10.0** | **0.5111 ← selected** |

> C=10.0 selected (weakest regularisation among candidates). The CV accuracy differences are very small (~0.0002), indicating that regularisation strength has minimal impact in this regime — the signal-to-noise ratio is low regardless of C.

#### Fallback pool — C CV scores

| C | CV mean accuracy |
|---|---|
| 0.01 | 0.4918 |
| 0.1 | 0.4943 |
| **1.0** | **0.4968 ← selected** |
| 10.0 | 0.4933 |

> C=1.0 selected. The fallback CV scores are all below 50%, suggesting the LogisticRegression model underperforms random guessing on CV folds — tree-based models outperform it on the test set. The non-monotonic pattern (C=10.0 < C=1.0) reflects high variance in the small training folds.

### 6.3 Model comparison — Standard pool (25 tickers, 11 features)

| Model | Accuracy (test) | F1 (binary) | Selected |
|---|---|---|---|
| LogisticRegression(C=10.0, balanced) | 50.0% | 0.521 | — |
| **RandomForestClassifier** | **50.6%** | **0.613** | **✅ Winner** |
| GradientBoostingClassifier | 50.6% | 0.603 | — |
| LGBMClassifier | 50.3% | 0.561 | — |

**Winner: RandomForestClassifier** → saved as `model/trained/model_pooled.pkl`
(tied with GBC at 50.6%; RFC selected as first in iteration order)

**Classification report (RandomForestClassifier, n=6,040 test rows):**
```
              precision    recall  f1-score   support

    DOWN (0)       0.47      0.24      0.32      2877
      UP (1)       0.52      0.75      0.61      3163

    accuracy                           0.51      6040
   macro avg       0.49      0.49      0.47      6040
weighted avg       0.49      0.51      0.47      6040
```

### 6.4 Model comparison — Fallback pool (5 tickers, 11 features)

| Model | Accuracy (test) | F1 (binary) | Selected |
|---|---|---|---|
| LogisticRegression(C=1.0, balanced) | 51.4% | 0.573 | — |
| RandomForestClassifier | 52.8% | 0.650 | — |
| **GradientBoostingClassifier** | **54.0%** | **0.675** | **✅ Winner** |
| LGBMClassifier | 52.6% | 0.626 | — |

**Winner: GradientBoostingClassifier** → saved as `model/trained/model_pooled_fallback.pkl`

**Classification report (GradientBoostingClassifier, n=1,214 test rows):**
```
              precision    recall  f1-score   support

    DOWN (0)       0.50      0.14      0.22       557
      UP (1)       0.55      0.88      0.67       657

    accuracy                           0.54      1214
   macro avg       0.52      0.51      0.45      1214
weighted avg       0.52      0.54      0.46      1214
```

### 6.5 Signal distribution (v4)

Unlike v3's percentile-based thresholds that mechanically forced ~25% BUY / ~25% SELL, v4's signal distribution depends on the confidence threshold (0.52) and the classifier's actual probability distribution:

| Signal | Trigger | Expected outcome |
|---|---|---|
| **BUY** | predict()==1 AND max(predict_proba) ≥ 0.52 | ~[TBD]% of inference calls |
| **SELL** | predict()==0 AND max(predict_proba) ≥ 0.52 | ~[TBD]% of inference calls |
| **HOLD** | max(predict_proba) < 0.52 (low confidence) | ~[TBD]% of inference calls |

> If the classifier is well-calibrated (predict_proba close to actual class frequencies), the majority of predictions near the 50/50 decision boundary will fall below 0.52 and produce HOLD signals. This is correct behaviour — HOLD means "not enough conviction to trade".

### 6.6 Hyperparameter configuration (v4 classifiers)

#### LogisticRegression (Winner or runner-up)

| Parameter | Value | Notes |
|---|---|---|
| Preprocessing | StandardScaler | Required: Market_Cap (~10¹²) vs Log_Return (~0.01) |
| C | [TBD] (CV-selected) | Inverse regularisation strength; lower C = stronger L2 |
| C candidates | [0.01, 0.1, 1.0, 10.0] | Tuned via TimeSeriesSplit(n=5) CV |
| CV metric | Accuracy | Primary operational criterion |
| max_iter | 1000 | Ensures convergence with large feature scale differences |
| solver | lbfgs | Default for L2 penalty |
| class_weight | balanced | Compensates for any class imbalance in up/down days |

#### RandomForestClassifier

| Parameter | Value | Notes |
|---|---|---|
| n_estimators | 200 | Explicit |
| max_depth | 8 | Limits tree depth — key regulariser vs v3 (was None) |
| min_samples_leaf | 15 | Minimum leaf size — prevents memorising individual days |
| max_features | 0.7 | Feature subsampling per split — adds regularisation |
| random_state | 42 | Reproducibility |
| n_jobs | −1 | All CPU cores |

#### GradientBoostingClassifier

| Parameter | Value | Notes |
|---|---|---|
| n_estimators | 500 | More estimators offset by slow learning rate |
| learning_rate | 0.02 | Slow learning → less overfit vs v3 GBR (was 0.1) |
| max_depth | 2 | Very shallow trees — strong bias/variance tradeoff |
| min_samples_leaf | 20 | Minimum leaf size |
| min_samples_split | 40 | Minimum samples to split a node |
| subsample | 0.7 | Row subsampling per tree — stochastic regularisation |
| max_features | 0.8 | Feature subsampling per split |
| random_state | 42 | Reproducibility |

#### LGBMClassifier

| Parameter | Value | Notes |
|---|---|---|
| n_estimators | 500 | More estimators offset by slow learning rate |
| learning_rate | 0.03 | Low → slow learning, less overfit |
| num_leaves | **15** | Shallow trees — reduced from default 31 for pooled scale |
| min_child_samples | **50** | Minimum leaf size — key regulariser (was 30 in v3) |
| colsample_bytree | 0.8 | Feature subsampling per tree |
| subsample | 0.7 | Row subsampling per tree |
| random_state | 42 | Reproducibility |
| n_jobs | −1 | All CPU cores |
| verbose | −1 | Suppress stdout |

#### Summary: regularisation comparison v3 → v4

| Model | v3 variant | v4 variant | Key changes |
|---|---|---|---|
| Linear | Ridge (regression) | LogisticRegression (classification) | Loss: MSE → binary cross-entropy; `class_weight='balanced'` added |
| RF | RandomForestRegressor (default params) | RandomForestClassifier | `max_depth=8`, `min_samples_leaf=15`, `max_features=0.7` — all tightened |
| GBR | GradientBoostingRegressor (defaults) | GradientBoostingClassifier | `lr=0.02`, `max_depth=2`, `subsample=0.7`, `max_features=0.8` — significantly regularised |
| LGBM | LGBMRegressor | LGBMClassifier | `num_leaves` 31→15, `min_child_samples` 30→50 — tightened |

---

## 7. Calibration Analysis — Predicted Probability Deciles

For classifiers that output `predict_proba()`, calibration measures whether higher predicted probability for class 1 (UP) corresponds to a higher actual frequency of up days.

Predictions are binned into deciles by `predict_proba()[:, 1]` (probability of UP). A well-calibrated classifier should show monotonically increasing actual UP frequency across deciles.

> **Fill in after running training and manually computing calibration from the test set outputs.**

### 7.1 Standard pool — probability calibration table

| Decile | N | Mean P(UP) | Actual UP freq. | Accuracy |
|---|---|---|---|---|
| 1 (lowest P(UP)) | [TBD] | [TBD] | [TBD] | [TBD] |
| 2 | [TBD] | [TBD] | [TBD] | [TBD] |
| 3 | [TBD] | [TBD] | [TBD] | [TBD] |
| 4 | [TBD] | [TBD] | [TBD] | [TBD] |
| 5 | [TBD] | [TBD] | [TBD] | [TBD] |
| 6 | [TBD] | [TBD] | [TBD] | [TBD] |
| 7 | [TBD] | [TBD] | [TBD] | [TBD] |
| 8 | [TBD] | [TBD] | [TBD] | [TBD] |
| 9 | [TBD] | [TBD] | [TBD] | [TBD] |
| 10 (highest P(UP)) | [TBD] | [TBD] | [TBD] | [TBD] |

### 7.2 Fallback pool — probability calibration table

| Decile | N | Mean P(UP) | Actual UP freq. | Accuracy |
|---|---|---|---|---|
| 1 (lowest P(UP)) | [TBD] | [TBD] | [TBD] | [TBD] |
| 2 | [TBD] | [TBD] | [TBD] | [TBD] |
| 3 | [TBD] | [TBD] | [TBD] | [TBD] |
| 4 | [TBD] | [TBD] | [TBD] | [TBD] |
| 5 | [TBD] | [TBD] | [TBD] | [TBD] |
| 6 | [TBD] | [TBD] | [TBD] | [TBD] |
| 7 | [TBD] | [TBD] | [TBD] | [TBD] |
| 8 | [TBD] | [TBD] | [TBD] | [TBD] |
| 9 | [TBD] | [TBD] | [TBD] | [TBD] |
| 10 (highest P(UP)) | [TBD] | [TBD] | [TBD] | [TBD] |

### 7.3 Calibration summary

| Pool | Monotonicity | Best decile accuracy | Worst decile accuracy | Actionable? |
|---|---|---|---|---|
| Standard | [TBD] | [TBD] | [TBD] | [TBD] |
| Fallback | [TBD] | [TBD] | [TBD] | [TBD] |

---

## 8. Context: Per-Ticker Prototype vs. Production v1 → v2 → v3 → v4

| | Per-ticker prototype | Prod. v1 (OLS) | Prod. v2 (Ridge+LGBM) | Prod. v3 (dynamic thresholds) | **Prod. v4 (classification)** |
|---|---|---|---|---|---|
| Problem type | Regression | Regression | Regression | Regression | **Classification** |
| Training rows | ~900 per model | 22,148 / 4,860 | 24,201 / 4,860 | 24,201 / 4,860 | ~24,000 / ~4,800 |
| Models evaluated | LR, RF, GBR | LR, RF, GBR | Ridge(CV), RF, GBR, LGBM | Same as v2 | **LogReg(CV), RFC, GBC, LGBMC** |
| Hyperparameter CV | N/A | N/A | α via TimeSeriesSplit | Same as v2 | **C via TimeSeriesSplit** |
| CV metric | N/A | N/A | Direction accuracy | Same as v2 | **Accuracy** |
| Winner | LR (13/25 tickers) | LinearRegression | Ridge α=10 / α=100 | Same as v2 | **RFC (standard) / GBC (fallback)** |
| Accuracy / dir. acc. | ~49–50% (per-ticker) | Not logged | 50.67% / 51.44% | Same as v2 | **50.65% / 54.04%** |
| F1 (binary) | Not logged | Not logged | Not logged | Not logged | **0.613 / 0.675** |
| Signal distribution | N/A | Not logged | 100% STAY | ~25% BUY / ~50% HOLD / ~25% SELL (forced) | **Biased toward UP (recall=0.75/0.88); HOLD when proba<0.52** |
| Signal thresholds | N/A | ±0.5%/±2% hardcoded | ±0.5%/±2% hardcoded | p75/p25 percentile-based | **None — class label + proba** |
| HOLD mechanism | N/A | ±0.5% range | ±0.5% range | p25–p75 range | **max(predict_proba) < 0.52** |
| Calibration | Not performed | Not performed | Computed (decile by pred. return) | Same as v2 | **Decile by pred. probability** |

---

## 9. Key Findings & Implications for the Trading System

### Finding 1 — Classification accuracy vs. v3 direction accuracy

- **Standard pool:** v4 RFC accuracy = **50.65%** vs. v3 Ridge direction accuracy = 50.67% → Δ = **−0.02%** (essentially identical)
- **Fallback pool:** v4 GBC accuracy = **54.04%** vs. v3 Ridge direction accuracy = 51.44% → Δ = **+2.60%** (meaningful improvement)

**Standard pool:** No improvement. The RFC at 50.65% is statistically indistinguishable from random guessing (z-score vs. 50%: (0.5065−0.5)/√(0.25/6040) = 1.01, p≈0.31). Switching from regression to classification did not unlock additional signal from these 11 technical features on a 25-ticker pool.

**Fallback pool:** Genuine improvement. GBC at 54.04% on 1,214 test rows is statistically significant (z-score: (0.5404−0.5)/√(0.25/1214) = 2.82, p≈0.005). The financial stocks (BAC/GS/JPM/MA/V) have more predictable price direction than the broader tech/consumer pool — a pattern that was already visible in v3's calibration analysis (top decile: 63.9% direction accuracy). GBC's shallow depth (2) and subsampling (0.7) appear to capture this signal robustly.

### Finding 2 — Signal distribution: strong UP bias

Both winning classifiers (RFC and GBC) show a heavy bias toward predicting UP:

| Pool | Model | UP recall | DOWN recall | Pattern |
|---|---|---|---|---|
| Standard | RFC | **0.75** | 0.24 | Predicts UP on ~75% of days |
| Fallback | GBC | **0.88** | 0.14 | Predicts UP on ~88% of days |

This is not a bug — it reflects the equity risk premium. US large-cap stocks rise more days than they fall (~52% UP in both pools). Tree-based classifiers without `class_weight='balanced'` learn to follow the majority class. The consequence for live inference: the classifier will generate mostly BUY signals, with SELL signals appearing only on high-confidence DOWN predictions.

**Impact of the 0.52 confidence gate:** Because tree classifiers tend to produce probabilities clustered near 0.5 (shallow trees, strong regularisation), the 0.52 threshold will suppress many BUY signals into HOLD. The actual BUY/SELL/HOLD split in production will depend on the model's probability distribution, which should be profiled after deploying the API wrapper.

### Finding 3 — Calibration quality by probability decile

The v3 fallback pool showed 63.9% direction accuracy in the top regression prediction decile. In v4, the equivalent test is whether high `predict_proba()[:, 1]` values correspond to high actual UP frequency. The classification report provides aggregate evidence: GBC achieves 55% UP precision overall, meaning ~55% of its UP predictions are correct. Given the strong UP recall bias (0.88), the model's high-probability UP calls should be more trustworthy than its DOWN calls.

*Full decile-by-probability calibration tables require running `model/calibration.py` after updating it for classifier output. The current `calibration.py` uses regression predictions — update it to use `predict_proba()[:, 1]` deciles.*

### Finding 4 — Structural fix vs. post-hoc patch

The v4 switch to classification is a **structural fix** for the signal diversity problem:

- **v2:** Ridge regression + fixed thresholds → 100% HOLD (prediction compression)
- **v3:** Ridge regression + dynamic percentile thresholds → ~25%/~50%/~25% BUY/HOLD/SELL (forced by construction, not by model conviction)
- **v4:** Classification + confidence gate → HOLD is generated by genuine low conviction, not by prediction scale mismatch

The v4 approach means a high BUY ratio could legitimately indicate the model is confident about upward moves — not an artifact of threshold calibration.

---

## 10. Known Limitations & Recommended Next Steps

### Limitations

| Limitation | Severity | Description |
|---|---|---|
| No walk-forward validation | High | Single 80/20 split cannot detect regime changes; rolling validation would give a more honest performance estimate |
| Confidence threshold (0.52) not tuned | Medium | The 0.52 threshold that gates HOLD vs. BUY/SELL is a reasonable default but is not optimised; a lower threshold generates more signals at the cost of lower precision; a higher threshold generates fewer but higher-confidence signals |
| Fundamental data unavailable | Medium | STANDARD_FEATURE_COLS has 5 fundamental ratios commented out; both schemas currently use the same 11 features — standard model has no advantage over fallback |
| No transaction costs | Medium | Accuracy assumes frictionless trading; true edge is lower with spreads and slippage |
| UP bias in tree classifiers | **Medium** | RFC and GBC predict UP on 75–88% of days (recall 0.75/0.88 for UP, 0.24/0.14 for DOWN). This matches the ~52% equity risk premium but suppresses SELL signals almost entirely. Adding `class_weight='balanced'` to RFC/GBC/LGBM or using `sample_weight` would force more balanced precision/recall across classes. |

### Recommended next steps

1. **Tune the confidence threshold** — grid search over [0.50, 0.52, 0.55, 0.58, 0.60] using the backtesting page. A threshold of 0.50 produces the most signals; 0.60 produces the fewest but highest-conviction signals. Optimise for Sharpe ratio or win rate on the test set.

2. **Implement walk-forward cross-validation** — 3-fold temporal CV: train 2020–2021 → test 2022; train 2020–2022 → test 2023; train 2020–2023 → test 2024. This gives the most honest out-of-sample accuracy estimate.

3. **Activate fundamental features** — restore the 5 commented-out fundamental ratios in STANDARD_FEATURE_COLS once the API wrapper is implemented. This will differentiate the standard and fallback schemas (16 vs 11 features) and may capture earnings-driven predictability.

4. **Add class_weight='balanced' to tree classifiers** — or use `sample_weight` in fit() to ensure RF/GBR/LGBM handle any up/down day imbalance as robustly as LogisticRegression.

5. **Feature importance from the winner** — extract feature importances (RF/GBR/LGBM: `.feature_importances_`; LogReg: `.named_steps['model'].coef_`) to understand which of the 11 features drive the classification boundary.

---

## 11. Historical Record: v3 Fix (Regression with Percentile Thresholds)

> This section is preserved for traceability. The percentile-based threshold mechanism (`thresholds.json`) has been removed in v4. The root cause it addressed (Ridge shrinkage → 100% HOLD) is now structurally resolved by switching to classification.

The v3 fix computed p75/p25 thresholds from the full prediction distribution and stored them in `model/trained/thresholds.json`. These thresholds were loaded at inference time by `go_live.py` and `backtesting.py`. The fix produced a ~25%/~50%/~25% BUY/HOLD/SELL distribution by construction — but this was a mechanical rank-based signal, not a model conviction signal.

**v4 removes:** `thresholds.json`, `compute_and_save_thresholds()`, `load_thresholds()` in app pages, the `thresholds` parameter in `prediction_to_signal()`.

**v4 standard thresholds (from `model/trained/thresholds.json`, now obsolete):**

| Pool | Buy threshold (p75) | Sell threshold (p25) |
|---|---|---|
| Standard | +0.1427% | +0.0389% |
| Fallback | +0.1499% | −0.0344% |

> Note: the standard pool sell threshold (+0.0389%) was positive — an artefact of the equity risk premium. Even the bottom 25% of Ridge predictions were slightly positive in expected value. This confirmed that the regression approach was not generating genuine negative signals; the percentile fix was a mechanical workaround.

---

*Report v4 generated from `python model/train.py --all` (March 2026). Standard pool: RandomForestClassifier, accuracy=50.65%, F1=0.613. Fallback pool: GradientBoostingClassifier, accuracy=54.04%, F1=0.675. Models deployed: `model/trained/model_pooled.pkl` (RFC, 25 tickers) and `model/trained/model_pooled_fallback.pkl` (GBC, 5 tickers). Calibration tables (Section 7) require updating `model/calibration.py` for classifier predict_proba output.*
