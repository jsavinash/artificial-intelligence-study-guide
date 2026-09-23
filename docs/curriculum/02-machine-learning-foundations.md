# 02 — Machine Learning Foundations

> Back to [index](../../README.md) · Prev: [01 Python](01-python-and-data-tooling.md) · Next: [03 Supervised Learning](03-supervised-learning-algorithms.md)

## 1. What machine learning is

**Definition (Mitchell):** a program learns from experience *E* with respect to task *T* and performance measure *P*, if its performance on T improves with E.

Contrast with classic programming: you write **rules**; in ML you provide **data + loss + search procedure**, and the machine finds the rules.

**Why it wins:** tasks where rules are unknowable (detecting spam nuances, predicting customer behavior) or where the world drifts and you need continuous adaptation.

### The three classic families

| Family | Supervision | Example |
|---|---|---|
| **Supervised** | labeled (x → y) | fraud detection, house prices |
| **Unsupervised** | unlabeled structure | customer segmentation, topic discovery |
| **Reinforcement** | delayed reward via environment | game playing, robotics, RLHF |

Plus: **self-supervised** (labels manufactured from the data itself — the engine behind LLMs; see [11](11-transformers-and-foundation-models.md)) and **semi/weakly supervised** (few labels, many unlabeled).

## 2. Types of supervised tasks

- **Regression:** continuous target (price, temperature).
- **Binary/multiclass classification:** categorical target.
- **Ranking/retrieval:** ordered relevance (search, recommenders).
- **Sequence labeling:** per-token labels (NER, part-of-speech).
- **Generation:** produce new sequences/images (LLMs, diffusion; see [12](12-generative-models.md)).

## 3. The ML workflow (the process that never changes)

1. **Problem framing** — what decision does the model feed? What's the metric of business success? (Accuracy often ≠ success.)
2. **Data collection & labeling** — provenance, consent, label quality (label noise caps achievable performance).
3. **Exploratory data analysis (EDA)** — distributions, missingness, class balance, leakage hunting, correlations.
4. **Split:** train / validation / test — *or* time-based splits for temporal data (never shuffle the future).
5. **Baseline** — trivial solution first (majority class, linear model, "predict yesterday's value"). No baseline → no idea if you're learning anything.
6. **Feature engineering / preprocessing** ([06](06-feature-engineering.md)).
7. **Model selection & training** ([03](03-supervised-learning-algorithms.md), [07](07-deep-learning-fundamentals.md)).
8. **Evaluation on validation** ([05](05-model-evaluation-and-tuning.md)).
9. **Error analysis** — read the failures; this is where real gains come from.
10. **Final test evaluation — once.**
11. **Deploy, monitor, retrain** ([21](21-mlops-and-production-ml.md)).

## 4. Data splitting — details that ruin projects

- **Train/val/test** typical sizes: 60/20/20, 70/15/15, or 80/10/10 with big data.
- **Stratify** classification splits to preserve class ratios.
- **Group splits:** all rows from one patient/user stay in one split — otherwise identity leaks.
- **Time-series:** split chronologically; validate on future windows (walk-forward).
- **Leakage:** any feature that wouldn't exist at prediction time (post-outcome data, target-derived aggregates, global preprocessing). Detection: does performance collapse in production? Prevention: fit everything inside the pipeline on training folds only.

## 5. Core concepts you must own

- **Features vs labels;** independent variables X, target y.
- **Parametric vs non-parametric:** fixed parameter count (linear regression) vs growing with data (KNN, trees).
- **Underfitting vs overfitting** — capacity too low vs memorizing noise. See bias–variance in [05](05-model-evaluation-and-tuning.md).
- **Generalization gap:** train performance minus test performance — your overfitting dashboard.
- **Inductive bias:** what assumptions make a model succeed? CNNs assume locality; transformers assume permutation + learned position; LSTMs assume long-range gates.
- **No Free Lunch theorem:** no universally best algorithm — performance is always relative to a data distribution. Hence: try several, measure.
- **Stationarity shift:** the world at deployment ≠ the world at training (covariate shift, label shift, concept drift — [21](21-mlops-and-production-ml.md)).

## 6. Supervised vs self-supervised vs unsupervised — the modern picture

```
Labels expensive?          → unsupervised / self-supervised pretrain
                             then fine-tune with few labels
Label-rich?                → standard supervised
Sequential decisions?      → reinforcement learning
```

**Self-supervised learning** is the single most important idea of the last decade: BERT masks tokens and predicts them; GPT predicts the next token; SimCLR crops images and learns agreement. The "labels" come free from the data's own structure. This is how all foundation models are built ([11](11-transformers-and-foundation-models.md), [14](14-large-language-models.md)).

## Mastery Checklist

- [ ] Names the three classic families + self-supervised, with an example of each
- [ ] Recites the ML workflow from problem framing to monitoring
- [ ] Explains why preprocessing must happen inside the CV pipeline (leakage)
- [ ] Chooses the right split strategy for cross-sectional, grouped, and temporal data
- [ ] Defines under/overfitting, generalization gap, inductive bias, No Free Lunch
- [ ] Always builds a baseline before training anything serious
