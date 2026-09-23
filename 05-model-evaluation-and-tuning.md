# 05 — Model Evaluation & Tuning

> Back to [index](README.md) · Prev: [04 Unsupervised](04-unsupervised-learning.md) · Next: [06 Feature Engineering](06-feature-engineering.md)

The module that separates practitioners from hobbyists: **an unmeasured model is an untrained model.**

## 1. The bias–variance tradeoff

`Expected Error = Bias² + Variance + Irreducible Noise`

- **Bias:** error from wrong assumptions (underfitting) — linear model on moons data.
- **Variance:** error from sensitivity to training samples (overfitting) — deep trees.
- **Irreducible:** label noise, missing features.
- **Total error is U-shaped vs capacity** — the sweet spot is validation-selected.
- Modern caveat: overparameterized nets can have *low bias and low variance* simultaneously ("double descent") — classical picture still the right diagnostic tool.
- **Diagnosis:** train score low → high bias → bigger model/features; train high, val low → high variance → more data, regularization, simpler model.

## 2. Metrics

### Regression
- **MAE** — average |error|, robust to outliers.
- **MSE/RMSE** — penalize big errors (same units as target for RMSE).
- **R²** — variance explained; 0 = predict-the-mean quality.
- Pick by business cost: symmetric → RMSE; outliers → MAE; asymmetric costs → custom loss.

### Binary classification
| Metric | Meaning | Use when |
|---|---|---|
| Accuracy | correct / total | balanced classes only |
| Precision | TP/(TP+FP) | false alarms costly (spam: don't lose real mail) |
| Recall (sensitivity) | TP/(TP+FN) | misses costly (cancer screening) |
| F1 | harmonic mean P & R | single number, imbalanced |
| **PR-AUC** | P-R curve area | **severe imbalance — prefer over ROC-AUC** |
| ROC-AUC | P(random pos > random neg) | ranking quality, balanced-ish |
| Log loss | penalized confidence | probabilistic quality |
| Specificity | TN/(TN+FP) | false-positive budget |

- **Threshold ≠ 0.5 always.** Tune the operating threshold on validation to your cost matrix / precision-recall requirement. Calibration matters first ([03](03-supervised-learning-algorithms.md)).

### Multiclass
Macro avg (each class equal) vs weighted avg (each sample equal) vs micro (global counts). **Confusion matrix** is the error-analysis artifact — read it.

### Ranking / retrieval / generation
NDCG, MAP, MRR (ranking); precision@k, recall@k, hit-rate (recommenders); BLEU/ROUGE (old MT/summarization), **BLEURT/BERTScore/LLM-as-judge** (modern generation eval — [19](19-llm-application-engineering.md)).

## 3. Cross-validation

- **K-fold** (5 or 10 standard): each point validated exactly once → honest variance estimate.
- **Stratified K-fold** for classification; **Group K-fold** for grouped entities; **TimeSeriesSplit** for temporal data.
- **Leave-one-out:** N folds — low bias, high variance, expensive.
- **Repeated K-fold** for small data stability.
- **Nested CV:** inner loop tunes, outer loop estimates — the unbiased way to report tuned performance.
- **Golden rule:** never tune on the test set. Test set is touched exactly once, at the end.

## 4. Regularization (fighting overfitting)

- **L1/L2 penalties** on weights; **early stopping**; **dropout**; **data augmentation**; **weight tying**; **reducing capacity**; **more data**.
- Classical ML: shrinkage, pruning trees, min_samples_leaf.
- Details in [08](08-training-and-regularizing-networks.md) for deep nets.

## 5. Hyperparameter tuning

| Strategy | How | Cost | Note |
|---|---|---|---|
| Grid search | exhaustive over grid | explodes with dims | wastes budget on bad combos |
| Random search | random draws | **better per hour** (Bergstra & Bengio) | default first step |
| Bayesian (TPE/GP) | model the objective | best sample-efficiency | Optuna, Ray Tune, SMAC |
| Successive halving/Hyperband | kill bad runs early | huge savings | combine with BO (ASHA) |

- **Rules:** define search space & budget first; seed everything; tune the highest-impact hyperparams first (capacity, learning rate, regularization strength); always re-evaluate the winner with fresh CV.
- Optuna example pattern: `study = optuna.create_study(direction="maximize"); study.optimize(obj, n_trials=100)`.

## 6. Imbalanced data (rare events)

1. **Use the right metric** (PR-AUC, F1, recall@precision).
2. **Class weights** (`class_weight="balanced"`) — usually the cleanest fix.
3. Resampling: oversample minority (SMOTE for tabular), undersample majority (loses info).
4. Threshold tuning post-training.
5. Anomaly-detection framing; one-class learning.
6. **Never** oversample before splitting — synthetic points leak across folds.

## 7. Error analysis — where real gains come from

- Slice metrics by segment (device, language, user type, length).
- Confusion pairs (class A always mistaken for B → labeling problem or missing feature).
- Review the 50 worst failures by loss — usually reveals label bugs, leakage, or a missing feature.
- **Data-centric improvement** (clean labels, add edge cases) often beats model swaps.

## Mastery Checklist

- [ ] Decomposes error into bias² + variance + noise and diagnoses from learning curves
- [ ] Chooses metrics for a given business cost (precision vs recall vs PR-AUC)
- [ ] Tunes classification thresholds deliberately, not at 0.5 by default
- [ ] Runs appropriate CV for grouped/temporal data and keeps test untouched
- [ ] Compares grid vs random vs Bayesian search and picks by budget
- [ ] Handles imbalance with weights/resampling/thresholds without leaking
