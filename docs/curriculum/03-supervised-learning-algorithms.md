# 03 — Supervised Learning Algorithms (Exhaustive)

> Back to [index](../../README.md) · Prev: [02 ML Foundations](02-machine-learning-foundations.md) · Next: [04 Unsupervised Learning](04-unsupervised-learning.md)

Each algorithm: **intuition → strengths → weaknesses → when to use.**

## A. Linear models

**Linear regression (regression):** fit `ŷ = w·x + b` minimizing MSE; closed form via normal equations. MSE = MLE under Gaussian noise. *Pros:* fast, interpretable, strong baseline. *Cons:* linear relationships only; outlier-sensitive (⇒ **Ridge** L2, **Lasso** L1 for sparsity, **ElasticNet** both); no interactions without manual features.

**Logistic regression (classification, despite the name):** `p = σ(w·x)`, minimize cross-entropy (MLE for Bernoulli); boundary at p=0.5. *Pros:* probabilistic, fast, interpretable, hard to beat on high-dim sparse text (with TF-IDF). *Cons:* linear boundary only. *Extensions:* softmax (multiclass), one-vs-rest.

## B. Distance & instance-based

**KNN:** prediction = majority/mean of K closest points. *Pros:* no training phase, nonlinear, intuitive. *Cons:* O(N) inference, curse of dimensionality, needs scaling, sensitive to K and irrelevant features.

**Naive Bayes:** Bayes' rule + conditional independence: `P(y|x) ∝ P(y)·∏P(xᵢ|y)`. Variants: Gaussian, Multinomial (text), Bernoulli. *Pros:* works with tiny data, extremely fast, superb text baseline, online learning. *Cons:* independence rarely holds → overconfident probabilities (ranking usually still fine).

## C. Tree-based (workhorses of tabular ML)

**Decision Trees (CART):** recursive splits where impurity drops most — Gini `1−Σpᵢ²` or entropy `−Σpᵢ log pᵢ`; regression uses MSE reduction. *Pros:* interpretable, no scaling, mixed types & nonlinearities. *Cons:* high variance → overfits; unstable to small data changes.

**Random Forest:** bag many decorrelated trees — bootstrap samples + random feature subsets; average predictions. *Pros:* big variance reduction, robust defaults, feature importances, minimal tuning. *Cons:* large/slow inference, poor extrapolation, less interpretable than one tree.

**Gradient Boosting — XGBoost / LightGBM / CatBoost:** add trees sequentially, each fitting the previous ensemble's residuals (negative gradients). *Pros:* **dominant algorithm on tabular data**; missing values, native categoricals (CatBoost), class weights; Kaggle standard. *Cons:* hyperparameter-sensitive (depth, lr, n_estimators), overfits noisy data, black box. LightGBM = leaf-wise + histogram binning (fast); XGBoost = depth-wise, regularized; CatBoost = best categorical defaults.

**Trees vs linear?** Boosted trees for structured/tabular data; linear models when you need interpretability, extreme speed, or sparse text.

## D. Maximum-margin

**SVM:** hyperplane maximizing the margin to nearest points (support vectors); slack C trades margin vs errors. **Kernel trick** (RBF, poly) implicitly maps to higher dimensions. *Pros:* effective in high dims, memory-efficient, principled generalization. *Cons:* slow on large N (O(N²–N³)), clumsy probability calibration, largely superseded by boosting/deep nets.

## E. Other important methods

- **LDA (Linear Discriminant Analysis):** generative classifier with covariance structure; also dimensionality reduction.
- **Calibration:** trees/SVM probabilities are poorly calibrated → Platt scaling / isotonic regression (needed for risk thresholds).
- **Stacking/blending:** meta-model on out-of-fold predictions of several models — consistent leaderboard winner.

## F. Choosing an algorithm

```
Tabular data?
  ├─ strong default        → Gradient boosting (LightGBM/XGBoost/CatBoost)
  ├─ need interpretability → Single tree / linear model
  └─ small data            → Ridge / RF / CatBoost
Text (sparse high-dim)     → Logistic reg / linear SVM, then transformers ([11](11-transformers-and-foundation-models.md))
Images / audio / sequences → Deep learning ([07](07-deep-learning-fundamentals.md))
```

## G. Details that matter for all of them

- **Scale features** for distance/gradient methods (KNN, SVM, linear/NN) — not for trees.
- **Learning curves:** both scores low → underfitting (more capacity); train high & val low → overfitting (data, regularization, simpler model).
- **Imbalance:** class weights, resampling, threshold tuning ([05](05-model-evaluation-and-tuning.md)).
- **Importance:** permutation importance (preferred, model-agnostic), SHAP ([23](23-advanced-and-specialized-topics.md)); impurity-based is biased toward high-cardinality features.

## Mastery Checklist

- [ ] Explains linear models' bias vs trees' variance, and how ensembles fix each
- [ ] States why gradient boosting wins on tabular data
- [ ] Knows what a kernel computes implicitly for SVM
- [ ] Picks a first-model baseline for any table/text/image dataset with a rationale
- [ ] Distinguishes bagging (parallel, variance↓) from boosting (sequential, bias↓)
- [ ] Knows L1 vs L2 effects on weights
