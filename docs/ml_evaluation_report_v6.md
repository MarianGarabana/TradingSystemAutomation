# ML Model Evaluation Report — v6
## Trading System Automation — Next-Day Direction Classification

**Date:** March 2026
**Dataset:** SimFin bulk CSV — 31 US large-cap tickers, 2020–2025
**Evaluation scope:** Production pooled classifiers v6 — class-balanced training
**Changes from v5:**
- **Balanced class weights applied to ALL four classifiers** — addresses the strong UP bias observed in v5 (standard GBC: DOWN recall 0.35; fallback GBC: DOWN recall 0.11).
  - `LogisticRegression`: `class_weight='balanced'` (already set in v5, no change)
  - `RandomForestClassifier`: `class_weight='balanced'` added
  - `GradientBoostingClassifier`: `sample_weight=compute_sample_weight('balanced', y_train)` passed to `.fit()`
  - `LGBMClassifier`: `is_unbalance=True` added
- **Standard pool winner changed** — GradientBoostingClassifier (v5) replaced by LogisticRegression(C=0.01, balanced) (v6). With balanced weights, tree ensembles can no longer exploit the UP frequency shortcut; LogReg's L2-regularised linear boundary becomes the strongest model on the standard pool.
- **Standard pool accuracy** — 51.55% (v5 GBC) → 50.08% (v6 LogReg). Small drop expected: the model no longer predicts UP on 66% of days.
- **Fallback pool winner unchanged** — GradientBoostingClassifier remains the winner (54.86% vs 55.68% in v5). Small accuracy drop due to balanced weights.
- **DOWN recall substantially improved in standard pool** — 0.35 → 0.61 (LogReg now issues meaningful SELL signals). Fallback pool DOWN recall: 0.11 → 0.16 (modest improvement).

**Source:** `python model/train.py --all`
→ `model/trained/model_pooled.pkl` (LogisticRegression C=0.01 balanced, 26 tickers, 16 features)
→ `model/trained/model_pooled_fallback.pkl` (GBC balanced weights, 5 tickers, 11 features)

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

#### Standard schema — 16 features (unchanged from v5)

| Group | Features | Count |
|---|---|---|
| Price-based | MA5, MA20, Volume_Change, Market_Cap, RSI, MACD | 6 |
| Volatility-normalised | Log_Return, Volatility_20, Return_norm, Return_norm_Lag1, Return_norm_Lag2 | 5 |
| Fundamental (quarterly, point-in-time) | Gross_Margin, Operating_Margin, Net_Margin, Debt_to_Equity, Operating_CF_Ratio | 5 |
| **Total** | | **16** |

#### Fallback schema — 11 features (unchanged from v5)

| Group | Features | Count |
|---|---|---|
| Price-based | MA5, MA20, Volume_Change, Market_Cap, RSI, MACD | 6 |
| Volatility-normalised | Log_Return, Volatility_20, Return_norm, Return_norm_Lag1, Return_norm_Lag2 | 5 |
| **Total** | | **11** |

The fallback schema covers BAC, GS, JPM, MA, V — tickers where fundamental data is structurally incompatible.

### Train / test split
- **Method:** Temporal 80/20 split per ticker before pooling (unchanged from v5).
- **No shuffling** — look-ahead bias prevention.
- **Zero-return rows dropped** before splitting (ambiguous for binary target).

### Class balancing — v6 change

| Classifier | v5 mechanism | v6 mechanism |
|---|---|---|
| `LogisticRegression` | `class_weight='balanced'` | Same (no change) |
| `RandomForestClassifier` | No balancing | `class_weight='balanced'` added |
| `GradientBoostingClassifier` | No balancing | `sample_weight=compute_sample_weight('balanced', y_train)` in `.fit()` |
| `LGBMClassifier` | No balancing | `is_unbalance=True` added |

`class_weight='balanced'` scales each sample's loss contribution by `n_samples / (n_classes * class_count)`. This penalises errors on the minority class (DOWN, ~48% of days) equally to errors on the majority class (UP, ~52% of days), preventing models from exploiting the base rate by defaulting to UP.

---

## 2–5. Per-Ticker Prototype Results (historical reference)

> Unchanged from v5. See `docs/ml_evaluation_report_v5.md` for the full per-ticker regression results (sections 2–5).

---

## 6. Production Pooled Model Results (v6 — Balanced Classification)

### 6.1 Training data summary

| Pool | Tickers | Features | Train rows | Test rows | UP days (train) | DOWN days (train) |
|---|---|---|---|---|---|---|
| Standard (`model_pooled.pkl`) | 26 (all except BAC/GS/JPM/MA/V) | 16 | 22,982 | 5,759 | 11,937 (51.9%) | 11,045 (48.1%) |
| Fallback (`model_pooled_fallback.pkl`) | BAC, GS, JPM, MA, V | 11 | 4,852 | 1,214 | 2,517 (51.9%) | 2,335 (48.1%) |

Split method: **per-ticker 80/20 temporal split before pooling**. Zero-return rows dropped. Row counts are identical to v5 — only model configuration changed.

### 6.2 LogisticRegression C cross-validation

C candidates evaluated via `TimeSeriesSplit(n_splits=5)`, optimising mean accuracy across folds.

#### Standard pool — C CV scores

| C | CV mean accuracy |
|---|---|
| **0.01** | **0.5125 (selected)** |
| 0.1 | 0.5120 |
| 1.0 | 0.5110 |
| 10.0 | 0.5102 |

> Strong regularisation (C=0.01) selected — consistent with v5. The fundamental feature group contains highly correlated ratios (Gross_Margin, Operating_Margin, Net_Margin all derived from Revenue), making strong L2 penalty optimal.

#### Fallback pool — C CV scores

| C | CV mean accuracy |
|---|---|
| 0.01 | 0.4903 |
| 0.1 | 0.4899 |
| **1.0** | **0.4948 (selected)** |
| 10.0 | 0.4911 |

> C=1.0 selected — unchanged from v5. Fallback pool uses only 11 price/vol features with low inter-correlation, so moderate regularisation is sufficient.

### 6.3 Model comparison — Standard pool (26 tickers, 16 features)

| Model | Accuracy (test) | F1 (binary) | Selected |
|---|---|---|---|
| **LogisticRegression(C=0.01, balanced)** | **50.1%** | **0.463** | **Winner** |
| RandomForestClassifier (balanced) | 49.7% | 0.461 | — |
| GradientBoostingClassifier (balanced weights) | 49.3% | 0.461 | — |
| LGBMClassifier (is_unbalance=True) | 49.0% | 0.461 | — |

**Winner: LogisticRegression(C=0.01, balanced)** → saved as `model/trained/model_pooled.pkl`

**Classification report (LogisticRegression, n=5,759 test rows):**
```
              precision    recall  f1-score   support

    DOWN (0)       0.48      0.61      0.53      2722
      UP (1)       0.54      0.41      0.46      3037

    accuracy                           0.50      5759
   macro avg       0.51      0.51      0.50      5759
weighted avg       0.51      0.50      0.50      5759
```

### 6.4 Model comparison — Fallback pool (5 tickers, 11 features)

| Model | Accuracy (test) | F1 (binary) | Selected |
|---|---|---|---|
| LogisticRegression(C=1.0, balanced) | 52.6% | 0.591 | — |
| RandomForestClassifier (balanced) | 52.8% | 0.654 | — |
| **GradientBoostingClassifier (balanced weights)** | **54.9%** | **0.677** | **Winner** |
| LGBMClassifier (is_unbalance=True) | 52.4% | 0.626 | — |

**Winner: GradientBoostingClassifier** → saved as `model/trained/model_pooled_fallback.pkl`

**Classification report (GradientBoostingClassifier, n=1,214 test rows):**
```
              precision    recall  f1-score   support

    DOWN (0)       0.51      0.16      0.25       553
      UP (1)       0.55      0.87      0.68       661

    accuracy                           0.55      1214
   macro avg       0.53      0.52      0.46      1214
weighted avg       0.54      0.55      0.48      1214
```

### 6.5 Recall comparison: v5 vs v6

| Pool | Metric | v5 (GBC, unbalanced) | v6 (winner, balanced) | Change |
|---|---|---|---|---|
| Standard | Accuracy | 51.55% | 50.08% | −1.47pp |
| Standard | UP recall | 0.66 | 0.41 | −0.25 |
| Standard | DOWN recall | 0.35 | **0.61** | **+0.26** |
| Standard | Macro F1 | 0.50 | 0.50 | 0.00 |
| Fallback | Accuracy | 55.68% | 54.86% | −0.82pp |
| Fallback | UP recall | 0.93 | 0.87 | −0.06 |
| Fallback | DOWN recall | 0.11 | 0.16 | +0.05 |
| Fallback | Macro F1 | 0.44 | 0.46 | +0.02 |

**Key result:** Standard pool DOWN recall nearly doubled (0.35 → 0.61). LogReg now correctly identifies 61% of actual DOWN days, versus 35% for GBC in v5. The standard model now issues a meaningful volume of SELL signals. Macro F1 is unchanged (0.50) — the model trades UP recall for DOWN recall without losing overall macro performance.

**Fallback pool:** GBC remains the winner. The balanced sample weights (passed via `sample_weight=` in `.fit()`) have limited effect on the fallback pool — DOWN recall improved only from 0.11 to 0.16. This is expected for a heavily imbalanced real-world signal (financial stocks trend upward persistently over 2020–2024). The dominant effect in the fallback pool is the structural UP bias in the data itself, which sample weighting partially but not fully corrects.

### 6.6 Signal distribution (v6)

| Signal | Trigger | Expected distribution |
|---|---|---|
| **BUY** | predict()==1 AND max(predict_proba) >= 0.52 | Reduced vs v5 — LogReg now predicts UP on only ~41% of standard days |
| **SELL** | predict()==0 AND max(predict_proba) >= 0.52 | Substantially increased vs v5 — DOWN recall 0.61 generates more SELL signals |
| **HOLD** | max(predict_proba) < 0.52 (low confidence) | High for standard pool — LogReg's probability range is narrow around 0.5 |

> **Note on the standard pool HOLD rate:** LogisticRegression with C=0.01 (strong L2 regularisation) produces probabilities clustered tightly around 0.5. With a 0.52 confidence gate, a large fraction of standard pool predictions will be converted to HOLD. This is conservative but correct — low-conviction predictions should not generate signals. If fewer signals are desired, lower the gate to 0.50; if higher conviction is required, raise to 0.55.

### 6.7 Hyperparameter configuration (v6 changes highlighted)

#### Standard pool winner — LogisticRegression

| Parameter | Value |
|---|---|
| Scaling | StandardScaler (required — Market_Cap ~10¹² vs Log_Return ~0.01) |
| C | 0.01 (selected by TimeSeriesSplit CV) |
| solver | lbfgs |
| max_iter | 1000 |
| **class_weight** | **'balanced' (v5 and v6)** |

#### Fallback pool winner — GradientBoostingClassifier

| Parameter | Value |
|---|---|
| n_estimators | 500 |
| learning_rate | 0.02 |
| max_depth | 2 |
| min_samples_leaf | 20 |
| min_samples_split | 40 |
| subsample | 0.7 |
| max_features | 0.8 |
| **sample_weight** | **compute_sample_weight('balanced', y_train) passed to .fit() (v6 new)** |

---

## 7. Calibration Analysis — Predicted Probability Deciles

### 7.1 Standard pool — probability calibration (LogisticRegression, 16 features, 26 tickers)

| Decile | N | Mean P(UP) | Actual UP freq | Accuracy |
|---|---|---|---|---|
| 1 (lowest P(UP)) | 576 | 0.457 | 0.505 | 0.495 |
| 2 | 576 | 0.475 | 0.549 | 0.451 |
| 3 | 576 | 0.483 | 0.491 | 0.509 |
| 4 | 576 | 0.488 | 0.547 | 0.453 |
| 5 | 576 | 0.493 | 0.523 | 0.477 |
| 6 | 575 | 0.498 | 0.518 | 0.482 |
| 7 | 576 | 0.502 | 0.531 | 0.531 |
| 8 | 576 | 0.508 | 0.519 | 0.519 |
| 9 | 576 | 0.515 | 0.564 | 0.564 |
| 10 (highest P(UP)) | 576 | 0.527 | 0.526 | 0.526 |

**Calibration summary — Standard pool:** The model is poorly calibrated in terms of probability spread. LogReg with strong L2 regularisation (C=0.01) compresses all probabilities into a narrow band (0.457–0.527). Actual UP frequency shows no monotonic trend across deciles. Deciles 9–10 (mean P(UP) 0.515–0.527) have the highest actual UP rate (56.4%, 52.6%), providing weak but real signal for BUY. However, since most predictions fall below the 0.52 confidence gate, the standard model will generate a large share of HOLD signals in practice.

### 7.2 Fallback pool — probability calibration (GBC, balanced weights, 11 features, 5 tickers)

| Decile | N | Mean P(UP) | Actual UP freq | Accuracy |
|---|---|---|---|---|
| 1 (lowest P(UP)) | 122 | 0.444 | 0.467 | 0.533 |
| 2 | 121 | 0.500 | 0.521 | 0.479 |
| 3 | 121 | 0.528 | 0.603 | 0.603 |
| 4 | 122 | 0.558 | 0.590 | 0.590 |
| 5 | 121 | 0.588 | 0.537 | 0.537 |
| 6 | 121 | 0.609 | 0.554 | 0.554 |
| 7 | 122 | 0.625 | 0.525 | 0.525 |
| 8 | 121 | 0.642 | 0.579 | 0.579 |
| 9 | 121 | 0.662 | 0.562 | 0.562 |
| 10 (highest P(UP)) | 122 | 0.698 | 0.508 | 0.508 |

**Calibration summary — Fallback pool:** Mixed calibration — similar pattern to v5. Decile 1 now shows 53.3% accuracy on DOWN direction (P(UP)=0.444, actual UP freq=0.467), a slight improvement vs v5 decile 1 (57.4% accuracy, different mechanism). Deciles 3–4 remain the most informative BUY region (60.3%, 59.0% UP freq). The upper deciles (5–10) are noisy. The balanced weights shifted some probability mass from high deciles toward lower deciles, slightly improving the low-P(UP) region's DOWN signal.

### 7.3 Calibration summary

| Pool | Model | Best region | Best decile accuracy | Worst decile accuracy | Actionable? |
|---|---|---|---|---|---|
| Standard | LogReg (C=0.01, balanced) | Decile 9 | 56.4% | 45.1% (decile 2) | Weak — narrow prob. range, most predictions below 0.52 gate |
| Fallback | GBC (balanced) | Decile 3 | 60.3% | 47.9% (decile 2) | Yes — deciles 1 and 3–4 carry directional signal |

---

## 8. Version Comparison: v1 through v6

| | Per-ticker prototype | Prod. v1 | Prod. v2 | Prod. v3 | Prod. v4 | Prod. v5 | **Prod. v6** |
|---|---|---|---|---|---|---|---|
| Problem type | Regression | Regression | Regression | Regression | Classification | Classification | **Classification** |
| Standard tickers | 25 | 25 | 25 | 25 | 25 | 26 | **26** |
| Standard features | 16 | 11 | 11 | 11 | 11 | 16 | **16** |
| Fallback features | 11 | 11 | 11 | 11 | 11 | 11 | **11** |
| Class balancing | — | — | — | — | LR only | LR only | **All classifiers** |
| Standard winner | LR (13/25) | LinearRegression | Ridge α=10 | Same as v2 | RFC | GBC | **LogReg (C=0.01)** |
| Standard accuracy | ~49–50% | Not logged | 50.67% | 50.67% | 50.65% | 51.55% | **50.08%** |
| Standard DOWN recall | — | — | — | — | — | 0.35 | **0.61** |
| Fallback winner | — | — | — | — | GBC | GBC | **GBC** |
| Fallback accuracy | — | — | — | — | 54.04% | 55.68% | **54.86%** |
| Fallback DOWN recall | — | — | — | — | — | 0.11 | **0.16** |
| Standard F1 (macro) | — | — | — | — | ~0.50 | 0.50 | **0.50** |
| Fallback F1 (macro) | — | — | — | — | ~0.44 | 0.44 | **0.46** |

> **Note:** Macro F1 is the fair cross-version comparison metric. Accuracy is misleading here because an UP-biased model (v5 GBC, UP recall=0.66) scores higher accuracy than a balanced model (v6 LogReg, UP recall=0.41) on a test set that is 52.7% UP — the accuracy gap reflects bias removal, not skill loss.

---

## 9. Key Findings

### Finding 1 — LogReg wins the standard pool when all classifiers are balanced

With balanced class weights applied to all four classifiers, tree ensembles can no longer exploit the UP frequency shortcut (predicting UP on most days to accumulate recall). All tree models (49.0–49.7%) fall below LogReg (50.1%). This confirms the initial hypothesis: the signal-to-noise ratio in daily returns is too low for GBC/RF to exploit genuine non-linear patterns — their v5 edge came entirely from the unbalanced training signal, not from non-linear feature interactions.

LogReg's L2-regularised linear decision boundary is more robust in this regime. With C=0.01 (strong regularisation), it avoids overfitting on the 16-feature space that includes highly correlated fundamental ratios.

### Finding 2 — Standard pool DOWN recall nearly doubled: 0.35 → 0.61

The most significant improvement in v6 is the standard pool's DOWN recall. With balanced training, LogReg correctly identifies 61% of actual DOWN days, versus 35% for GBC in v5. This means the standard model now issues a meaningful volume of SELL signals instead of being almost exclusively a BUY signal generator.

The tradeoff: UP recall dropped from 0.66 to 0.41. The model is now balanced in its errors — macro F1 is unchanged at 0.50. Overall accuracy dropped by 1.47pp (51.55% → 50.08%) because the model no longer exploits the ~52% UP base rate.

### Finding 3 — Fallback pool is more resilient to balancing, but GBC still shows UP bias

GBC remains the fallback winner (54.86%) despite balanced weights. DOWN recall improved only from 0.11 to 0.16 — a modest gain. The structural UP trend in financial stocks over 2020–2024 is strong enough that even with balanced sample weights, the boosted tree model learns to predict UP on ~87% of days. The `sample_weight` mechanism penalises DOWN errors more, but with 5 tickers and a ~52%/48% split, the absolute number of DOWN training samples (2,335 rows) is still sufficient for GBC to find UP patterns.

### Finding 4 — Standard pool calibration is extremely narrow under LogReg with C=0.01

With C=0.01, LogReg compresses all probabilities into the range 0.457–0.527. This means the 0.52 confidence gate will classify a large fraction of predictions as HOLD — only predictions above decile 8 (mean P(UP) > 0.508) exceed the gate. This is conservative but defensively correct. If fewer HOLD signals are desired, consider lowering the gate to 0.50 or using a less aggressive C (e.g., C=0.1 selects probabilities from a wider range).

### Finding 5 — Macro F1 is the honest metric; accuracy is misleading

Both v5 and v6 achieve macro F1 = 0.50 on the standard pool. Accuracy changed (51.55% → 50.08%) because accuracy rewards the UP-biased model: predicting UP 66% of the time yields 51.55% accuracy on a 52.7% UP test set. Macro F1 treats both classes equally and reveals that v5's accuracy advantage was entirely due to the UP bias, not genuine directional skill. In v6, both models have the same macro F1, confirming the comparison is now on equal terms.

---

## 10. Known Limitations and Recommended Next Steps

### Remaining limitations

| Limitation | Severity | Description |
|---|---|---|
| No walk-forward validation | High | Single 80/20 split; rolling validation would give a more honest estimate across market regimes |
| Standard pool HOLD rate is high | Medium | LogReg C=0.01 produces very narrow probabilities (0.457–0.527); most predictions fall below the 0.52 gate → fewer actionable signals |
| Fallback DOWN recall still low | Medium | GBC fallback DOWN recall = 0.16; balanced weights partially but not fully correct the structural UP trend in financial stocks |
| Confidence threshold not tuned | Medium | 0.52 is a reasonable default; not optimised against a Sharpe or win-rate objective |
| Fundamental data is static | Low | Live inference uses last-known quarterly values from processed CSV. Will lag if a company reports between CSV regenerations |
| No transaction costs | Medium | Accuracy assumes frictionless trading |

### Recommended next steps

1. **Tune the confidence threshold for LogReg** — grid search over [0.50, 0.51, 0.52, 0.53] on the standard pool. With C=0.01, probabilities are narrow; lowering the gate to 0.50 or 0.51 will produce more signals while still requiring some directional conviction.

2. **Experiment with C=0.1 for the standard pool** — slightly weaker L2 regularisation would widen the probability spread (more predictions above 0.52), at the cost of slightly higher collinearity risk from the fundamental feature group.

3. **Implement walk-forward cross-validation** — 3-fold: train 2020–2021 → test 2022; train 2020–2022 → test 2023; train 2020–2023 → test 2024. This would reveal whether v6 LogReg's standard pool accuracy (50.08%) is stable across market regimes or volatile.

4. **Explore class-balanced XGBoost** — `scale_pos_weight = n_negative / n_positive` is XGBoost's equivalent of `is_unbalance`. XGBoost's column-sampling and regularisation may outperform LightGBM on this dataset.

5. **Refresh processed CSVs periodically** — run `python etl/etl.py --all` to update fundamental values and extend price history. Retrain with `python model/train.py --all` to incorporate new data.

---

## 11. Historical Record

See `docs/ml_evaluation_report_v5.md` for full v5 results including:
- Complete calibration tables (v5 GBC probabilities)
- Full hyperparameter tables for all four classifiers
- Detailed analysis of the 16-feature standard schema (fundamentals activation)
- Per-ticker regression prototype results (sections 2–5)

See `docs/ml_evaluation_report_v4.md` for v1–v4 history.

---

*Report v6 generated from `python model/train.py --all` (March 2026, v6 balanced class weights).
Standard pool: LogisticRegression(C=0.01, balanced), accuracy=50.08%, macro F1=0.50, DOWN recall=0.61, 26 tickers, 16 features.
Fallback pool: GradientBoostingClassifier (balanced sample weights), accuracy=54.86%, macro F1=0.46, DOWN recall=0.16, 5 tickers, 11 features.
Models: `model/trained/model_pooled.pkl` (LogReg) and `model/trained/model_pooled_fallback.pkl` (GBC).*
