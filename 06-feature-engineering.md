# 06 — Feature Engineering

> Back to [index](README.md) · Prev: [05 Evaluation](05-model-evaluation-and-tuning.md) · Next: [07 Deep Learning Fundamentals](07-deep-learning-fundamentals.md)

**Feature engineering = domain knowledge turned into numbers.** For classical ML it's often worth more than the algorithm choice. (Deep nets learn their own features — that's the paradigm shift of [07](07-deep-learning-fundamentals.md).)

## 1. Golden rules

1. Fit **all transforms on training data only**, inside a Pipeline ([01](01-python-and-data-tooling.md)).
2. Hunt **target leakage**: any feature that encodes the label or its future (payment flag on churn dataset, "days until churn"). Ask: *would this exist at prediction time in production?*
3. Test-time **distribution must match** train-time features — feature stores exist to guarantee this ([21](21-mlops-and-production-ml.md)).
4. Feature quality > feature quantity. Correlated junk multiplies overfitting surface.

## 2. Numeric features

- **Scaling:** StandardScaler (zero-mean/unit-var), MinMaxScaler (bounds), RobustScaler (outlier-resistant). Needed for distance/gradient models; irrelevant for trees.
- **Skew:** `log1p`, Box-Cox/Yeo-Johnson for heavy tails (prices, counts).
- **Binning/discretization:** turn continuous into ordinal buckets — captures nonlinearity, robust to outliers.
- **Ratios & interactions:** `income/age`, `price per sqm` — domain ratios beat raw columns.
- **Polynomial features / crosses:** hand-pick meaningful two-way interactions instead of blind degree-3 explosions.
- **Outliers:** cap at quantiles (winsorize) or flag as separate binary feature.
- **Missing values:** often informative! Add `was_missing` indicator; impute with median/mean/model-based (KNN impute, iterative).

## 3. Categorical features

| Encoding | Idea | Watch out |
|---|---|---|
| One-hot | binary column per category | cardinality explosion; use for low-card & linear models |
| Ordinal/label | integer ranks | only for true orderings |
| **Target/mean encoding** | replace with `mean(y)` per category | **must be smoothed + computed inside CV folds** or leaks |
| Count/frequency | category popularity | usually safe |
| Hashing | fixed buckets for huge cardinality | collisions; used in NLP/recommenders |
| Embedding lookup | learned dense vectors | deep models; categories → vectors ([23](23-advanced-and-specialized-topics.md)) |

- **High cardinality** (IDs, zip codes): target encoding with smoothing, or drop ID-like columns — they memorize.
- **Unknown at inference:** every scheme needs an "unseen category" path (one-hot drop, embedding index 0).

## 4. Dates & time

- Parts: hour, day-of-week, month, is-weekend, holiday flags, season.
- **Cyclical encoding:** `sin/cos(2π·hour/24)` — hour 23 and 0 are adjacent, not far apart.
- **Lags & rolling windows** for time series: `value_lag_1..n`, 7/30-day rolling mean, expanding statistics — all computed **in the past only** ([23](23-advanced-and-specialized-topics.md)).

## 5. Text features (classical)

- **Bag-of-words / CountVectorizer**, **TF-IDF** (`tf · log(N/df)`) — down-weights ubiquitous words.
- n-grams (bigrams capture "new york"), char-ngrams (robust to typos, languages).
- Length features, caps, punctuation counts, language ID.
- Stopword handling & stemming/lemmatization — sometimes hurt; validate.
- The modern replacement: **dense embeddings** ([11](11-transformers-and-foundation-models.md), [17](17-retrieval-augmented-generation.md)).

## 6. Images (classical CV)

Histogram equalization, resizing/normalization to [0,1] or model-specific means/stds; augmentation covered in [08](08-training-and-regularizing-networks.md)/[09](09-convolutional-networks-and-vision.md).

## 7. Feature selection

- **Filter:** correlation/mutual-information thresholds, chi-square for text — fast, model-agnostic.
- **Wrapper:** recursive feature elimination (RFE) — model in the loop, costly.
- **Embedded:** L1 (selects), tree **permutation importance**, SHAP values.
- Remove: constant/near-constant, highly redundant (collinear pair), leaky features.

## 8. The modern caveat

For **tabular** data, hand features still win. For **images, text, audio**: representation learning won — feed raw inputs to a network and let it build features; your job shifts to data quality and curation ([07](07-deep-learning-fundamentals.md), [14](14-large-language-models.md)).

## Mastery Checklist

- [ ] Explains leakage vs legal feature with a concrete example
- [ ] Chooses an encoding for low/medium/high-cardinality categoricals and justifies it
- [ ] Computes target encoding safely inside CV folds with smoothing
- [ ] Engineers time features with cyclical encoding and past-only lags
- [ ] Builds TF-IDF features and knows how n-grams change them
- [ ] Runs filter, wrapper, and embedded feature selection
