# 00 — Mathematical Foundations for AI

> Back to [index](README.md) · Next: [01 Python & Data Tooling](01-python-and-data-tooling.md)

You do **not** need a math degree. You need working intuition for six areas. This file is the pragmatic syllabus: what to learn, why it matters in AI, and the minimum you must be able to *do*.

## 1. Linear Algebra — the language of data

**Why AI runs on it:** every dataset is a matrix; every model is matrix operations; GPUs exist to do fast matrix multiplication.

| Concept | AI relevance |
|---|---|
| Vectors, norm (‖x‖) | Data points; embedding similarity (cosine = dot ÷ norms) |
| Matrix multiply | Layers: `Y = WX + b` — one line runs a whole mini-batch |
| Dot product | Similarity, attention scores |
| Eigenvalues/vectors | PCA, covariance structure, Google's original PageRank |
| Positive-definite matrices | Loss surfaces, covariance, Gaussians |
| SVD / low-rank | Compression, embeddings geometry, LoRA in 15 |

**You should be able to:** multiply matrices by hand once, know shapes must align, understand broadcasting, explain what a linear layer computes.

## 2. Calculus & Gradients — how models learn

- **Derivative = sensitivity.** ∂L/∂w tells you how the loss changes when weight `w` nudges.
- **Gradient** ∇L = the vector of all partial derivatives → the direction of steepest *increase*; learning subtracts it.
- **Chain rule = backpropagation.** If `L = f(g(x))`, then `dL/dx = f'(g(x))·g'(x)`. Neural nets are just the chain rule applied thousands of times, systematically.
- **Partial derivatives & Jacobians:** outputs w.r.t. vector inputs.
- **Hessian (second order):** curvature — explains why optimization is hard and what "saddle points" are.

**Minimum bar:** compute `d/dx` of polynomials and `exp`, apply the chain rule to a 2-layer toy network by hand.

## 3. Probability — reasoning under uncertainty

- **Axioms, conditional probability, Bayes' rule:**
  `P(H|D) = P(D|H)·P(H) / P(D)` — the backbone of Bayesian learning, Naive Bayes classifiers, and Bayesian optimization for tuning.
- **Distributions you must know:** Bernoulli, Categorical/Multinomial (classification labels), Gaussian (noise, VAEs, Kalman filters), Poisson (counts), Uniform (initialization, sampling).
- **Expectation & variance:** E[X], Var(X) = E[X²]−(E[X])²; linearity of expectation.
- **Joint / marginal / independence:** P(X,Y), P(X)=ΣY P(X,Y).
- **Likelihood vs probability:** likelihood P(data | model) is what we maximize (MLE) in nearly every training loss.
- **Sampling:** Monte Carlo, importance sampling (used in diffusion/RL estimators).

## 4. Statistics — from sample to claim

- **Estimators, bias–variance of estimators** (distinct from the model bias–variance tradeoff in 05 — related but not identical).
- **Confidence intervals & hypothesis tests** (p-values) — for A/B testing model changes; know their common misuse.
- **Correlation ≠ causation** — the single most expensive confusion in applied ML.
- **Maximum likelihood estimation (MLE)** and **MAP** — connects probability to training objectives.
- **Covariance & correlation matrices** — input to PCA, portfolio models.

## 5. Optimization — finding good parameters

- **Objective functions** and why non-convexity (loss landscapes of deep nets) means "good enough," not "the best."
- **Gradient descent family:** batch / stochastic / mini-batch; learning rate is the most important hyperparameter.
- **Convergence conditions**, step sizes, convexity basics (convex ⇒ global optimum; deep learning abandons this).
- **Lagrange multipliers / constrained optimization** — at least conceptually (SVMs use them; RLHF optimizes under KL constraints).
- **Search strategies:** grid vs random vs **Bayesian optimization** (05).

## 6. Information Theory — measuring information

- **Entropy** `H(p) = −Σ p·log p` — surprise/uncertainty; the floor of lossless compression.
- **Cross-entropy** `H(p,q) = −Σ p·log q` — *the* classification loss; why `CrossEntropyLoss = softmax + log + NLL`.
- **KL divergence** — how one distribution differs from another; basis of distillation and RLHF's KL penalty.
- **Mutual information** — feature selection, information bottleneck.

## Suggested exercises

1. By hand: `(2×3 matrix) · (3×2 matrix)`; verify shapes.
2. Backprop by hand on a 2-input, 1-hidden-unit network.
3. Apply Bayes' rule to a medical-test paradox problem (1% prevalence, 90% accurate test).
4. Compute cross-entropy of a confident-wrong vs confident-right prediction — feel the loss explode.
5. Implement gradient descent on `f(x)=x²` in 10 lines of NumPy.

## Mastery Checklist

- [ ] Can state what a matrix multiplication computes in a neural layer and why shapes matter
- [ ] Can apply the chain rule to a 2-layer network manually
- [ ] Can use Bayes' rule and name 5 common distributions with their use cases
- [ ] Explains MLE and why cross-entropy loss is MLE for categorical data
- [ ] Knows that learning = descending the gradient of a loss, with mini-batch noise
- [ ] Knows what entropy, cross-entropy, and KL divergence each measure
