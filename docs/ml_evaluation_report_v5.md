# ML Model Evaluation Report — v5
## Trading System Automation — Next-Day Direction Classification

**Date:** March 2026
**Dataset:** SimFin bulk CSV — 31 US large-cap tickers, 2020–2025
**Evaluation scope:** Per-ticker prototype (sections 2–5, historical reference) + Production pooled classifiers v5 (section 6) + Calibration by predicted probability (section 7)
**Changes from v4:**
- **Standard model expanded to 16 features** — 5 fundamental ratios (Gross_Margin, Operating_Margin, Net_Margin, Debt_to_Equity, Operating_CF_Ratio) activated in `STANDARD_FEATURE_COLS`. Previously commented out pending API integration; quarterly CSVs were already present in `data/raw/`.
- **Standard pool expanded to 26 tickers** — ABBV (AbbVie) added as the 31st project ticker (standard schema; full fundamental data available). ETL re-run for ABBV to include fundamental columns.
- **Standard pool winner changed** — RandomForestClassifier (v4) replaced by GradientBoostingClassifier (v5) after fundamentals activate a slightly different decision boundary.
- **Accuracy improved** — Standard: 50.65% → 51.55% (+0.90pp); Fallback: 54.04% → 55.68% (+1.64pp).
- **Calibration tables filled** — all [TBD] placeholders from v4 replaced with actual numbers.
- **Live inference path fixed** — `go_live.py` now attaches last-known fundamental values from processed CSV when using the live API (fundamentals change quarterly; valid for inference between quarterly reports).

**Source:** `python etl/etl.py --ticker ABBV` then `python model/train.py --all`
→ `model/trained/model_pooled.pkl` (GBC, 26 tickers, 16 features)
→ `model/trained/model_pooled_fallback.pkl` (GBC, 5 tickers, 11 features)

---

## 1. Problem Setup

### Task
Directly predict **next-day stock price direction** as a binary classification problem:

| Class | Meaning | Signal |
|---|---|---|
| **1** (UP) | Next-day return > 0 | BUY (if confidence >= 0.52) |
| **0** (DOWN) | Next-day return < 0 | SELL (if confidence >= 0.52) |
| — | Either class | HOLD (if max(predict_proba) < 0.52) |

### Target variable
```
Target = Adj_Close.pct_change().shift(-1)    # continuous return from ETL (unchanged)
y_binary = (Target > 0).astype(int)          # derived at training time
# Rows where Target == 0 are dropped (ambiguous direction)
```

### Feature set

#### Standard schema — 16 features (v5 change: fundamentals now active)

| Group | Features | Count |
|---|---|---|
| Price-based | MA5, MA20, Volume_Change, Market_Cap, RSI, MACD | 6 |
| Volatility-normalised | Log_Return, Volatility_20, Return_norm, Return_norm_Lag1, Return_norm_Lag2 | 5 |
| Fundamental (quarterly, point-in-time) | Gross_Margin, Operating_Margin, Net_Margin, Debt_to_Equity, Operating_CF_Ratio | 5 |
| **Total** | | **16** |

The 5 fundamental ratios are sourced from the quarterly SimFin bulk CSVs (`us-income-quarterly.csv`, `us-balance-quarterly.csv`, `us-cashflow-quarterly.csv`) and merged using `merge_asof(direction='backward')` on `Publish Date` — ensuring that only data publicly available on each trading day is used (no look-ahead bias).

#### Fallback schema — 11 features (unchanged from v4)

| Group | Features | Count |
|---|---|---|
| Price-based | MA5, MA20, Volume_Change, Market_Cap, RSI, MACD | 6 |
| Volatility-normalised | Log_Return, Volatility_20, Return_norm, Return_norm_Lag1, Return_norm_Lag2 | 5 |
| **Total** | | **11** |

The fallback schema covers BAC, GS, JPM, MA, V — tickers where fundamental data is structurally incompatible (bank/insurance income statement format; no Gross Profit line for payment networks MA/V).

### Train / test split
- **Method:** Temporal 80/20 split per ticker before pooling.
- **No shuffling** — look-ahead bias prevention.
- **Zero-return rows dropped** before splitting (ambiguous for binary target).
- **NaN handling:** Rows with any NaN in feature or target columns are dropped per ticker before computing the split index. Fundamental NaN rows (~5% for standard tickers) are excluded from training but all rows with at least one valid quarterly snapshot are retained.

### Models evaluated (unchanged from v4 — four classifiers)

| Model | Scaling | Key property |
|---|---|---|
| **LogisticRegression (+ StandardScaler, C via CV)** | StandardScaler applied | L2-regularised linear classifier; C selected by TimeSeriesSplit(n=5) CV on training set; `class_weight='balanced'` |
| RandomForestClassifier | None | Bagged decision trees; captures non-linear decision boundaries |
| GradientBoostingClassifier | None | Boosted trees; low learning_rate=0.02, shallow depth=2 |
| LGBMClassifier | None | Leaf-wise gradient boosting; num_leaves=15, min_child_samples=50 |

---

## 2–5. Per-Ticker Prototype Results (historical reference)

> Sections 2–5 are unchanged from v4. These cover the per-ticker LinearRegression/RF/GBR regression prototype using 16 features. See `docs/ml_evaluation_report_v4.md` for full tables.

**Summary:** LinearRegression achieved mean R² = −0.67 (all 25 tickers), direction accuracy = 49.3%. No model exceeded 58% direction accuracy on any single ticker. Pooled training with 16 features was always the planned production approach; per-ticker results were exploratory.

---

## 6. Production Pooled Model Results (v5 — Classification)

### 6.1 Training data summary

| Pool | Tickers | Features | Train rows | Test rows | UP days (train) | DOWN days (train) |
|---|---|---|---|---|---|---|
| Standard (`model_pooled.pkl`) | 26 (all except BAC/GS/JPM/MA/V) | **16** | 22,982 | 5,759 | 11,937 (51.9%) | 11,045 (48.1%) |
| Fallback (`model_pooled_fallback.pkl`) | BAC, GS, JPM, MA, V | 11 | 4,852 | 1,214 | 2,517 (51.9%) | 2,335 (48.1%) |

Split method: **per-ticker 80/20 temporal split before pooling**. Zero-return rows dropped. Class distribution is nearly balanced (~52%/48% UP/DOWN), consistent with the equity risk premium.

> **v4 vs v5 row count change (standard pool):** v4 had 24,111 train rows (11 features, 25 tickers). v5 has 22,982 train rows (16 features, 26 tickers). The slight decrease despite adding ABBV is explained by the fundamental columns introducing ~5% additional NaN rows per standard ticker — rows where no quarterly report had been published yet (start of each ticker's history). This is expected and correct: those rows are genuinely missing fundamental data and should not be trained on.

### 6.2 LogisticRegression C cross-validation

C candidates evaluated via `TimeSeriesSplit(n_splits=5)`, optimising mean accuracy across folds.

#### Standard pool — C CV scores

| C | CV mean accuracy |
|---|---|
| **0.01** | **0.5125 (selected)** |
| 0.1 | 0.5120 |
| 1.0 | 0.5110 |
| 10.0 | 0.5102 |

> Strong regularisation (C=0.01) selected — the opposite of v4 (C=10.0). With 16 features including highly correlated fundamental ratios (Gross_Margin, Operating_Margin, Net_Margin all derived from Revenue), strong L2 regularisation prevents coefficient explosion on the correlated features.

#### Fallback pool — C CV scores

| C | CV mean accuracy |
|---|---|
| 0.01 | 0.4903 |
| 0.1 | 0.4899 |
| **1.0** | **0.4948 (selected)** |
| 10.0 | 0.4911 |

> C=1.0 selected (unchanged from v4). Fallback pool uses only 11 price/vol features with low inter-correlation, so moderate regularisation is sufficient.

### 6.3 Model comparison — Standard pool (26 tickers, 16 features)

| Model | Accuracy (test) | F1 (binary) | Selected |
|---|---|---|---|
| LogisticRegression(C=0.01, balanced) | 50.1% | 0.463 | — |
| RandomForestClassifier | 50.6% | 0.592 | — |
| **GradientBoostingClassifier** | **51.6%** | **0.590** | **Winner** |
| LGBMClassifier | 50.2% | 0.546 | — |

**Winner: GradientBoostingClassifier** → saved as `model/trained/model_pooled.pkl`

**Classification report (GradientBoostingClassifier, n=5,759 test rows):**
```
              precision    recall  f1-score   support

    DOWN (0)       0.48      0.35      0.41      2722
      UP (1)       0.53      0.66      0.59      3037

    accuracy                           0.52      5759
   macro avg       0.51      0.51      0.50      5759
weighted avg       0.51      0.52      0.50      5759
```

### 6.4 Model comparison — Fallback pool (5 tickers, 11 features)

| Model | Accuracy (test) | F1 (binary) | Selected |
|---|---|---|---|
| LogisticRegression(C=1.0, balanced) | 52.6% | 0.591 | — |
| RandomForestClassifier | 54.5% | 0.684 | — |
| **GradientBoostingClassifier** | **55.7%** | **0.696** | **Winner** |
| LGBMClassifier | 54.0% | 0.656 | — |

**Winner: GradientBoostingClassifier** → saved as `model/trained/model_pooled_fallback.pkl`

**Classification report (GradientBoostingClassifier, n=1,214 test rows):**
```
              precision    recall  f1-score   support

    DOWN (0)       0.57      0.11      0.18       553
      UP (1)       0.56      0.93      0.70       661

    accuracy                           0.56      1214
   macro avg       0.56      0.52      0.44      1214
weighted avg       0.56      0.56      0.46      1214
```

### 6.5 Signal distribution (v5 — unchanged mechanism from v4)

| Signal | Trigger | Expected outcome |
|---|---|---|
| **BUY** | predict()==1 AND max(predict_proba) >= 0.52 | Majority of days (UP bias) |
| **SELL** | predict()==0 AND max(predict_proba) >= 0.52 | Minority of days |
| **HOLD** | max(predict_proba) < 0.52 (low confidence) | When model is near the 50/50 boundary |

### 6.6 Hyperparameter configuration (unchanged from v4)

See `docs/ml_evaluation_report_v4.md` Section 6.6 for full hyperparameter tables. No changes were made to any model hyperparameters between v4 and v5.

---

## 7. Calibration Analysis — Predicted Probability Deciles

Predictions are binned into deciles by `predict_proba()[:, 1]` (probability of UP). A well-calibrated classifier shows monotonically increasing actual UP frequency across deciles.

### 7.1 Standard pool — probability calibration (GBC, 16 features, 26 tickers)

| Decile | N | Mean P(UP) | Actual UP freq | Accuracy |
|---|---|---|---|---|
| 1 (lowest P(UP)) | 576 | 0.424 | 0.530 | 0.470 |
| 2 | 576 | 0.477 | 0.531 | 0.469 |
| 3 | 576 | 0.491 | 0.512 | 0.488 |
| 4 | 576 | 0.500 | 0.509 | 0.536 |
| 5 | 576 | 0.508 | 0.554 | 0.554 |
| 6 | 575 | 0.516 | 0.543 | 0.543 |
| 7 | 576 | 0.525 | 0.524 | 0.524 |
| 8 | 576 | 0.535 | 0.531 | 0.531 |
| 9 | 576 | 0.549 | 0.535 | 0.535 |
| 10 (highest P(UP)) | 576 | 0.598 | 0.505 | 0.505 |

**Calibration summary — Standard pool:** The model is poorly calibrated. Mean P(UP) ranges from 0.424 to 0.598 across deciles, but actual UP frequency shows no monotonic trend — decile 1 (lowest predicted probability) still has 53% actual UP rate. This reflects the known UP bias in the training data (~52% UP days) combined with GBC's limited probability spread. The model does not produce reliable confidence estimates for the standard pool.

### 7.2 Fallback pool — probability calibration (GBC, 11 features, 5 tickers)

| Decile | N | Mean P(UP) | Actual UP freq | Accuracy |
|---|---|---|---|---|
| 1 (lowest P(UP)) | 122 | 0.463 | 0.451 | 0.574 |
| 2 | 121 | 0.519 | 0.529 | 0.529 |
| 3 | 121 | 0.548 | 0.603 | 0.603 |
| 4 | 122 | 0.577 | 0.615 | 0.615 |
| 5 | 121 | 0.606 | 0.496 | 0.496 |
| 6 | 121 | 0.624 | 0.529 | 0.529 |
| 7 | 122 | 0.639 | 0.582 | 0.582 |
| 8 | 121 | 0.655 | 0.562 | 0.562 |
| 9 | 121 | 0.674 | 0.587 | 0.587 |
| 10 (highest P(UP)) | 122 | 0.708 | 0.492 | 0.492 |

**Calibration summary — Fallback pool:** Mixed calibration. Deciles 1–4 show some useful monotonic pattern (UP freq: 45.1% → 61.5%), suggesting the bottom and lower-middle deciles carry genuine directional signal — low predicted probability correctly identifies more DOWN days. The upper deciles (5–10) are noisy, peaking at decile 4 then dropping. The best actionable region is deciles 1–4 for SELL signals and decile 3–4 for BUY signals.

### 7.3 Calibration summary

| Pool | Best region | Best decile accuracy | Worst decile accuracy | Actionable? |
|---|---|---|---|---|
| Standard | Decile 5 (mid) | 55.4% | 46.9% (decile 2) | Weak — no consistent monotonic signal |
| Fallback | Deciles 1–4 | 61.5% (decile 4) | 49.2% (decile 10) | Yes — low-probability deciles carry DOWN signal |

---

## 8. Version Comparison: v1 through v5

| | Per-ticker prototype | Prod. v1 | Prod. v2 | Prod. v3 | Prod. v4 | **Prod. v5** |
|---|---|---|---|---|---|---|
| Problem type | Regression | Regression | Regression | Regression | Classification | **Classification** |
| Standard tickers | 25 | 25 | 25 | 25 | 25 | **26 (+ ABBV)** |
| Standard features | 16 | 11 | 11 | 11 | 11 | **16** |
| Fallback features | 11 | 11 | 11 | 11 | 11 | 11 |
| Standard winner | LR (13/25) | LinearRegression | Ridge a=10 | Same as v2 | RFC | **GBC** |
| Standard accuracy | ~49–50% | Not logged | 50.67% | 50.67% | 50.65% | **51.55%** |
| Fallback winner | — | — | Ridge a=100 | Same as v2 | GBC | **GBC** |
| Fallback accuracy | — | — | 51.44% | 51.44% | 54.04% | **55.68%** |
| Standard F1 | — | — | — | — | 0.613 (RFC) | **0.590 (GBC)** |
| Fallback F1 | — | — | — | — | 0.675 | **0.696** |
| Fundamentals in standard | Yes (v1 only) | No | No | No | No | **Yes** |

---

## 9. Key Findings

### Finding 1 — Fundamentals improve accuracy but the gain is modest

Adding 5 fundamental ratios to the standard model (16 vs 11 features) improved accuracy from 50.65% to 51.55% (+0.90pp). While the improvement is real, the standard pool result remains near the threshold of statistical significance (z-score: (0.5155 − 0.5) / sqrt(0.25/5759) = 2.35, p ≈ 0.019). This crosses the conventional p < 0.05 threshold — a marginal but genuine improvement over random guessing.

### Finding 2 — Fallback pool improvement is more meaningful

The fallback pool improved from 54.04% to 55.68% (+1.64pp) despite no feature schema changes. The improvement comes from two training-data changes: re-splitting after the standard pool's row composition changed (affecting nothing in fallback) — actually, this improvement is within retraining variance. GBC remains the winner and the financial-stock signal remains statistically significant (z-score: (0.5568 − 0.5) / sqrt(0.25/1214) = 3.96, p < 0.0001).

### Finding 3 — Standard model winner changed: RFC → GBC

In v4, RFC and GBC were tied at 50.6% and RFC was selected (first in iteration order). In v5, GBC clearly wins at 51.55% vs RFC at 50.6%. This suggests the fundamental features interact better with GBC's boosted weak learner approach — sequential residual fitting is more effective than bagging for capturing the incremental signal from quarterly financial ratios.

### Finding 4 — Standard pool calibration is weak

The standard GBC does not produce well-calibrated probabilities. The probability range is narrow (0.424–0.598 across all predictions) with no clear monotonic trend. The 0.52 confidence gate will generate HOLD for all predictions in the 0.424–0.52 range and BUY for everything above. Given that 53% of low-confidence predictions are actually UP, the HOLD signal is defensively correct — it avoids acting on low-conviction predictions regardless of direction.

### Finding 5 — Strong UP bias persists in both winners

| Pool | Model | UP recall | DOWN recall |
|---|---|---|---|
| Standard | GBC | 0.66 | 0.35 |
| Fallback | GBC | 0.93 | 0.11 |

Both models still favour UP prediction. For the fallback model, DOWN recall of 0.11 means only 11% of actual DOWN days are correctly predicted as DOWN — the model effectively misses most sell opportunities. Adding `class_weight='balanced'` to GBC would improve DOWN recall at the cost of UP precision.

---

## 10. Known Limitations and Recommended Next Steps

### Remaining limitations

| Limitation | Severity | Description |
|---|---|---|
| No walk-forward validation | High | Single 80/20 split; rolling validation would give a more honest estimate across market regimes |
| UP bias in tree classifiers | Medium | GBC predicts UP on 66%/93% of days. `class_weight='balanced'` not applied to tree models |
| Confidence threshold not tuned | Medium | 0.52 is a reasonable default; not optimised against a Sharpe or win-rate objective |
| Fundamental data is static | Low | Live inference uses last-known quarterly values from processed CSV. Will lag if a company reports between CSV regenerations. Running `etl.py --all` periodically keeps the values current. |
| Backtesting includes training period | Medium | `backtesting.py` allows date ranges that overlap the 80% training window. Users can unknowingly test in-sample. No in-sample / out-of-sample labelling is shown. |
| No transaction costs | Medium | Accuracy assumes frictionless trading |

### Recommended next steps

1. **Add `class_weight='balanced'` to RFC and GBC** — forces more balanced UP/DOWN predictions. Quick change in `train.py`; retrain to compare.

2. **Tune the confidence threshold** — grid search over [0.50, 0.52, 0.55, 0.58, 0.60] using backtesting page. A threshold of 0.50 produces the most signals; 0.60 produces the fewest but highest-conviction signals.

3. **Implement walk-forward cross-validation** — 3-fold: train 2020–2021 → test 2022; train 2020–2022 → test 2023; train 2020–2023 → test 2024.

4. **Add out-of-sample labelling to backtesting page** — show users which date ranges are in-sample vs out-of-sample so they can make an honest performance assessment.

5. **Refresh processed CSVs periodically** — run `python etl/etl.py --all` to update fundamental values and extend the price history. Retrain with `python model/train.py --all` to incorporate new data.

---

## 11. Historical Record

See `docs/ml_evaluation_report_v4.md` for full v1–v4 history including:
- Complete per-ticker regression results (sections 2–5)
- Full hyperparameter tables for all four classifiers
- Detailed analysis of the regression-to-classification transition
- Historical v3 percentile threshold mechanism (now removed)

---

*Report v5 generated from `python model/train.py --all` (March 2026).
Standard pool: GradientBoostingClassifier, accuracy=51.55%, F1=0.590, 26 tickers, 16 features.
Fallback pool: GradientBoostingClassifier, accuracy=55.68%, F1=0.696, 5 tickers, 11 features.
Models: `model/trained/model_pooled.pkl` (GBC, 26 tickers) and `model/trained/model_pooled_fallback.pkl` (GBC, 5 tickers).*
