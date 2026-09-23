# 00 — Mathematical Foundations for AI

> Back to [index](../../README.md) · Next: [01 Python & Data Tooling](01-python-and-data-tooling.md)

You do **not** need a math degree. You need working intuition for six areas. This file is the pragmatic syllabus: what to learn, why it matters in AI, and the minimum you must be able to *do*.

### 📊 Visual guide — 31 figures, one per topic

| § | Topics | Figures |
|---|---|---|
| 1 · Linear algebra | vectors/dot/cosine · matmul shapes · eigen→PCA · SVD/low-rank | [01](../figures/00-math/01_vectors_dot.png) · [02](../figures/00-math/02_matmul_shapes.png) · [03](../figures/00-math/03_eigen_pca.png) · [04](../figures/00-math/04_svd_lowrank.png) |
| 2 · Calculus | derivative · gradient field · chain rule/backprop · partials · Hessian/saddle | [05](../figures/00-math/05_derivative_tangent.png) · [06](../figures/00-math/06_gradient_field.png) · [07](../figures/00-math/07_chain_graph.png) · [08](../figures/00-math/08_partials.png) · [09](../figures/00-math/09_hessian_curvature.png) · [31](../figures/00-math/31_learning_loop.png) |
| 3 · Probability | Bayes medical test · 5 distributions · E/Var · joint/marginal · likelihood/MLE · Monte Carlo | [10](../figures/00-math/10_bayes_test.png) · [11](../figures/00-math/11_distributions.png) · [12](../figures/00-math/12_expectation_variance.png) · [13](../figures/00-math/13_joint_marginal.png) · [14](../figures/00-math/14_likelihood.png) · [15](../figures/00-math/15_monte_carlo.png) · [30](../figures/00-math/30_probability_tree.png) |
| 4 · Statistics | estimator bias–variance · CI/p-value · correlation≠causation · MLE vs MAP · covariance | [16](../figures/00-math/16_estimator_biasvar.png) · [17](../figures/00-math/17_ci_pvalue.png) · [18](../figures/00-math/18_correlation_causation.png) · [19](../figures/00-math/19_mle_map.png) · [20](../figures/00-math/20_covariance_correlation.png) |
| 5 · Optimization | convexity · learning rate · batch vs SGD · Lagrange · grid/random/Bayesian search | [21](../figures/00-math/21_convex_nonconvex.png) · [22](../figures/00-math/22_learning_rate.png) · [23](../figures/00-math/23_batch_vs_sgd.png) · [24](../figures/00-math/24_lagrange.png) · [25](../figures/00-math/25_grid_random_bayes.png) |
| 6 · Information theory | entropy · cross-entropy loss · KL divergence · mutual information | [26](../figures/00-math/26_entropy.png) · [27](../figures/00-math/27_cross_entropy.png) · [28](../figures/00-math/28_kl_divergence.png) · [29](../figures/00-math/29_mutual_information.png) |

Regenerate all figures: `python3 tools/make_math_figures.py` · Verify every worked number: `python3 tools/calc_00_math.py` · Executable versions: `make run M=00` · **Jupyter notebook (all of the above, live): [`notebooks/00-mathematical-foundations.ipynb`](../../notebooks/00-mathematical-foundations.ipynb)** — `make notebook` to open · rebuild+execute+verify: `make notebook-run`

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

**🧮 Formula sheet**

| What | Formula | Plain English |
|---|---|---|
| Dot product | `a·b = a₁b₁ + a₂b₂ + …` | multiply matching entries, add them up |
| Length (norm) | `‖a‖ = √(a₁² + a₂² + …)` | distance from the origin |
| Cosine similarity | `cos θ = a·b / (‖a‖·‖b‖)` | 1 = same direction, 0 = unrelated |
| Matrix multiply | `C[i,j] = Σₖ A[i,k]·B[k,j]` | row *i* of A dotted with column *j* of B |
| Shape rule | `(m×n) · (n×p) → (m×p)` | the middle numbers must be equal |
| Linear layer | `Y = W·X + b` | `W` = learned weights, `X` = one column per sample |

**🔢 Worked calculation — `[[1,2,3],[4,5,6]] · [[7,8],[9,10],[11,12]]`**

| cell | row · column | expand | result |
|---|---|---|---|
| C[0,0] | `1·7 + 2·9 + 3·11` | 7 + 18 + 33 | **58** |
| C[0,1] | `1·8 + 2·10 + 3·12` | 8 + 20 + 36 | **64** |
| C[1,0] | `4·7 + 5·9 + 6·11` | 28 + 45 + 66 | **139** |
| C[1,1] | `4·8 + 5·10 + 6·12` | 32 + 50 + 72 | **154** |

→ `[[58,64],[139,154]]` — each cell is just four multiplications and three additions.

**🔢 Worked calculation — cosine of `a=(3,1)`, `b=(1,3)`**

| step | formula | numbers | result |
|---|---|---|---|
| 1 · dot | `Σ aᵢ·bᵢ` | `3·1 + 1·3` | 6 |
| 2 · norms | `√(a₁²+a₂²)` | `√10` each | 3.162 |
| 3 · cosine | `a·b / (‖a‖·‖b‖)` | `6 / (3.162×3.162)` | **0.60** |
| 4 · angle | `arccos(0.60)` | — | **53.13°** |

> ✅ every number on this page is machine-checked by `python3 tools/calc_00_math.py`

**🖼️ Figures — dot product & cosine similarity · matmul shape flow · eigenvectors = PCA · SVD truncation:**

![Dot product and cosine similarity between two vectors](../figures/00-math/01_vectors_dot.png)

![Matrix multiplication shapes: inner dimensions must agree](../figures/00-math/02_matmul_shapes.png)

![Eigenvectors of the covariance matrix are the PCA directions](../figures/00-math/03_eigen_pca.png)

![SVD low-rank approximation at ranks 10, 5, 2, 1](../figures/00-math/04_svd_lowrank.png)

**💡 Simple examples:**
- **Shapes:** `(2×3) · (3×2) → (2×2)` — inner `3 = 3` must match, else error.
- **Multiply by hand:** `[[1,2,3],[4,5,6]] · [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]` — row·column dot products.
- **Cosine similarity:** `a=(3,1), b=(1,3)` → `a·b = 6`, `‖a‖‖b‖ = 10` → `cos θ = 0.6` → θ = 53.1°. Embeddings use this instead of raw distance.
- **Linear layer:** `Y = WX + b` with `W:(out×in)`, `X:(in×batch)` — each column of `Y` is one sample's scores.

## 2. Calculus & Gradients — how models learn

- **Derivative = sensitivity.** ∂L/∂w tells you how the loss changes when weight `w` nudges.
- **Gradient** ∇L = the vector of all partial derivatives → the direction of steepest *increase*; learning subtracts it.
- **Chain rule = backpropagation.** If `L = f(g(x))`, then `dL/dx = f'(g(x))·g'(x)`. Neural nets are just the chain rule applied thousands of times, systematically.
- **Partial derivatives & Jacobians:** outputs w.r.t. vector inputs.
- **Hessian (second order):** curvature — explains why optimization is hard and what "saddle points" are.

**Minimum bar:** compute `d/dx` of polynomials and `exp`, apply the chain rule to a 2-layer toy network by hand.

**🧮 Formula sheet**

| Rule | Formula | Says |
|---|---|---|
| Power rule | `d/dx xⁿ = n·xⁿ⁻¹` | bring the exponent down, drop one |
| Exponential | `d/dx eˣ = eˣ` | its slope equals its value |
| Chain rule | `dL/dx = (dL/dg)·(dg/dx)` | multiply the local slopes along the way |
| Partial | `∂f/∂xᵢ` | slope along ONE axis, others frozen |
| Gradient | `∇f = (∂f/∂x₁, …, ∂f/∂xₙ)` | full vector of slopes = steepest climb |
| Learning step | `w ← w − η·∇L` | subtract gradient × learning rate |

**🔢 Worked — chain rule for `L = (2x+1)³` at `x = 1`**

| step | what | calculation | value |
|---|---|---|---|
| 1 | inner value | `g = 2·1 + 1` | 3 |
| 2 | inner slope | `dg/dx` | 2 |
| 3 | outer slope | `dL/dg = 3g² = 3·3²` | 27 |
| 4 | multiply | `27 × 2` | **54** |

Figure 7 draws exactly this graph; `calc_00_math.py` double-checks it against finite differences.

**🔢 Worked — a derivative and one learning step**
- `f(x)=x³−2x` → `f'(x)=3x²−2` → `f'(2) = 3·4 − 2 = **10**` (slope at x=2).
- `L(w)=w²`, `w=3`, `η=0.1` → `∇L = 2w = 6` → `w ← 3 − 0.1·6 = **2.4**`, loss `9 → 5.76` (**−36% in one step**).

![The five-step training loop: forward, loss, backward, update, repeat](../figures/00-math/31_learning_loop.png)

**🖼️ Figures — derivative as tangent slope · gradient field (red = uphill, green = learning) · chain-rule computation graph · partial derivatives · Hessian curvature · the training loop:**

![Derivative as the slope of the tangent line](../figures/00-math/05_derivative_tangent.png)

![Gradient field showing uphill gradient and downhill learning direction](../figures/00-math/06_gradient_field.png)

![Chain rule as a computation graph: dL/dx = 27 x 2 = 54](../figures/00-math/07_chain_graph.png)

![Partial derivatives of a 2-D function and the resulting gradient](../figures/00-math/08_partials.png)

![Hessian eigenvalues classify minimum, saddle, and maximum](../figures/00-math/09_hessian_curvature.png)

**💡 Simple examples:**
- **Chain rule:** `L = (2x+1)³` at `x=1` → `dL/dx = 3(2x+1)² · 2 = 3·9·2 = 54` (the figure, computed).
- **Gradient:** for `L = x₁² + 2x₂²` at `(1,1)` → `∇L = (2, 4)`; step `w ← w − η·∇L`.
- **Saddle point:** at `(0,0)` of `f = x² − y²` the Hessian eigenvalues are `+1, −1` — mixed signs = neither min nor max; gradients vanish there, which stalls naive optimizers.

## 3. Probability — reasoning under uncertainty

- **Axioms, conditional probability, Bayes' rule:**
  `P(H|D) = P(D|H)·P(H) / P(D)` — the backbone of Bayesian learning, Naive Bayes classifiers, and Bayesian optimization for tuning.
- **Distributions you must know:** Bernoulli, Categorical/Multinomial (classification labels), Gaussian (noise, VAEs, Kalman filters), Poisson (counts), Uniform (initialization, sampling).
- **Expectation & variance:** E[X], Var(X) = E[X²]−(E[X])²; linearity of expectation.
- **Joint / marginal / independence:** P(X,Y), P(X)=ΣY P(X,Y).
- **Likelihood vs probability:** likelihood P(data | model) is what we maximize (MLE) in nearly every training loss.
- **Sampling:** Monte Carlo, importance sampling (used in diffusion/RL estimators).

**🧮 Formula sheet**

| Rule | Formula | Plain English |
|---|---|---|
| Conditional | `P(A\|B) = P(A∩B) / P(B)` | A's share once you restrict to B |
| Bayes | `P(H\|D) = P(D\|H)·P(H) / P(D)` | update belief after seeing evidence |
| Total probability | `P(D) = Σₕ P(D\|Hₕ)·P(Hₕ)` | every way the evidence can happen |
| Independence | `P(A∩B) = P(A)·P(B)` | only if knowing A says nothing about B |
| Expectation | `E[X] = Σ x·p(x)` | probability-weighted average |
| Variance | `Var = E[X²] − (E[X])²` | average squared spread around the mean |

**🔢 Worked — the medical test, per 1000 people**

| group | size | P(+ \| group) | expected "+" results |
|---|---|---|---|
| sick | 10 (1%) | 0.90 | `10 × 0.90 =` **9** |
| healthy | 990 (99%) | 0.05 | `990 × 0.05 =` **49.5** |
| all positives | — | — | **58.5** |

`P(sick|+) = 9 / 58.5 = 0.1538 →` **15.4%**

⚠️ mind the swap: `P(+|sick) = 90%` but `P(sick|+) = 15.4%` — the condition changed sides. The tree below shows exactly why (9 vs 49.5 positives).

![Probability tree: branch from 1000 people and normalize the positive leaves](../figures/00-math/30_probability_tree.png)

**📋 Reference table — the five distributions you must know**

| Distribution | Parameters | E[X] | Var | Used for |
|---|---|---|---|---|
| Bernoulli | p | p | p(1−p) | yes/no labels |
| Categorical | p₁…p_K | — | — | K-class labels |
| Gaussian | μ, σ² | μ | σ² | noise, VAE priors |
| Poisson | λ | λ | λ | counting events |
| Uniform | a, b | (a+b)/2 | (b−a)²/12 | init, sampling |

**🔢 Worked — a fair die:** `E = (1+2+3+4+5+6)/6 = 3.5` · `Var = (6.25+2.25+0.25+0.25+2.25+6.25)/6 = 17.5/6 = 2.917` (= 35/12)

**🖼️ Figures — Bayes medical test · the five distributions · expectation & variance · joint/marginal tables · likelihood & MLE · Monte Carlo π:**

![Bayes rule on a medical test: prior 1 percent lifts to posterior 15.4 percent](../figures/00-math/10_bayes_test.png)

![The five must-know distributions: Bernoulli, Categorical, Gaussian, Poisson, Uniform](../figures/00-math/11_distributions.png)

![Expectation and variance read off a Gaussian sample](../figures/00-math/12_expectation_variance.png)

![Joint probability table with marginal sums](../figures/00-math/13_joint_marginal.png)

![Likelihood of data for candidate means and the log-likelihood curve maximized by MLE](../figures/00-math/14_likelihood.png)

![Monte Carlo estimation of pi by sampling points in a square](../figures/00-math/15_monte_carlo.png)

**💡 Simple examples:**
- **Medical test (exercise #3):** prevalence 1%, sensitivity 90%, false-positive 5% → of 1000 people: 9 sick+positive vs 50 healthy+positive → `P(sick|+) = 9/59 = 15.4%`, not 90%.
- **Bayes on a die:** roll once, posterior given “even” → `P(2|even)=1/3` each, `P(odd|even)=0`.
- **Expectation:** fair die `E[X] = 3.5`, `Var(X) = 35/12 ≈ 2.92`; `E[2X+1] = 2E[X]+1 = 8` (linearity).
- **MLE:** fit a Gaussian to data → the maximizing μ̂ is just the sample mean (figure 14's argmax).

## 4. Statistics — from sample to claim

- **Estimators, bias–variance of estimators** (distinct from the model bias–variance tradeoff in 05 — related but not identical).
- **Confidence intervals & hypothesis tests** (p-values) — for A/B testing model changes; know their common misuse.
- **Correlation ≠ causation** — the single most expensive confusion in applied ML.
- **Maximum likelihood estimation (MLE)** and **MAP** — connects probability to training objectives.
- **Covariance & correlation matrices** — input to PCA, portfolio models.

**🧮 Formula sheet**

| Quantity | Formula | Meaning |
|---|---|---|
| Mean | `x̄ = Σx / n` | center of the sample |
| Variance | `s² = Σ(x−x̄)² / n` | average squared distance from center |
| Std deviation | `s = √s²` | spread in the original units |
| Standard error | `SE = s / √n` | how wobbly the *mean* is |
| 95% confidence interval | `x̄ ± 1.96·SE` | range that should capture the truth |
| Pearson correlation | `r = Sxy / √(Sxx·Syy)` | linear strength: +1 / 0 / −1 |
| Maximum likelihood | `θ̂ = argmax P(D\|θ)` | best fit using **data only** |
| Maximum a posteriori | `θ̂ = argmax P(D\|θ)·P(θ)` | best fit = **data × prior** |

**🔢 Worked — dataset {2, 4, 4, 4, 5, 5, 7, 9}**

| step | calculation | result |
|---|---|---|
| 1 · count | `n` | 8 |
| 2 · mean | `(2+4+4+4+5+5+7+9)/8 = 40/8` | **5** |
| 3 · deviations | `x − 5` | −3, −1, −1, −1, 0, 0, 2, 4 |
| 4 · squared devs | `9+1+1+1+0+0+4+16` | **32** |
| 5 · variance | `32 / 8` | **4** |
| 6 · std dev | `√4` | **2** |

**🔢 Worked — correlation of `x={1,2,3}`, `y={2,4,5}`**

`x̄ = 2`, `ȳ = 11/3 ≈ 3.667` → `Sxy = 3`, `Sxx = 2`, `Syy = 14/3 ≈ 4.667`
`r = 3 / √(2 × 4.667) = 3 / 3.055 =` **0.982** → strong positive… and *still* not causation (figure 18).

**🖼️ Figures — estimator bias vs variance · confidence intervals & p-values · correlation ≠ causation · MLE vs MAP · covariance/correlation:**

![Bias and variance of estimators: unbiased but spread vs biased but tight](../figures/00-math/16_estimator_biasvar.png)

![Confidence intervals across A/B repeats and the null distribution p-value tails](../figures/00-math/17_ci_pvalue.png)

![Three datasets with similar correlations but different causal truths](../figures/00-math/18_correlation_causation.png)

![MLE versus MAP: likelihood, prior, and their posterior product](../figures/00-math/19_mle_map.png)

![Scatter clouds at different correlations plus a correlation matrix heatmap](../figures/00-math/20_covariance_correlation.png)

**💡 Simple examples:**
- **Estimator vs model:** the *sample mean* is unbiased (`E[x̄]=μ`) but noisy; averaging n draws shrinks `Var` by `1/n` — and no amount of data fixes a biased estimator (figure 16).
- **95% CI (exercise-adjacent):** `x̄ ± 1.96·s/√n`; if the CI for a model lift excludes 0, ship the change — but a p-value is *not* `P(H₀ true)` (figure 17).
- **Confounder:** summer heat raises *both* ice-cream sales and drownings — correlating the two finds a strong link with zero causation (figure 18).
- **MLE vs MAP:** 8 datapoints averaging 2.0 say μ̂=2.0 (MLE); a prior centered at 1.2 pulls the posterior peak between them (figure 19).

## 5. Optimization — finding good parameters

- **Objective functions** and why non-convexity (loss landscapes of deep nets) means "good enough," not "the best."
- **Gradient descent family:** batch / stochastic / mini-batch; learning rate is the most important hyperparameter.
- **Convergence conditions**, step sizes, convexity basics (convex ⇒ global optimum; deep learning abandons this).
- **Lagrange multipliers / constrained optimization** — at least conceptually (SVMs use them; RLHF optimizes under KL constraints).
- **Search strategies:** grid vs random vs **Bayesian optimization** (05).

**🧮 Formula sheet**

| Method | Update rule | Note |
|---|---|---|
| Batch gradient descent | `w ← w − η·∇L(all data)` | stable, expensive per step |
| SGD / mini-batch | `w ← w − η·∇L(batch)` | noisy — the practical default |
| Weight decay | `w ← w − η·(∇L + λw)` | regularization: steadily shrink weights |
| Adam | `m,v` = running averages of ∇ and ∇²; `w ← w − η·m̂/(√v̂+ε)` | per-parameter step sizes; the most-used default |
| 1-D stability | `abs(1 − η·L″) < 1` | curvature × step must not over-correct |

**🔢 Worked — gradient descent on `L(w)=w²`, start `w = 2.400`, `η = 0.1`** (here `∇L = 2w`)

| step | w | ∇L = 2w | new w = w − 0.1·∇L | loss = w² |
|---|---|---|---|---|
| 0 | 2.400 | 4.800 | **1.920** | 5.760 |
| 1 | 1.920 | 3.840 | **1.536** | 3.686 |
| 2 | 1.536 | 3.072 | **1.229** | 2.359 |
| 3 | 1.229 | 2.458 | **0.983** | 1.510 |
| 4 | 0.983 | 1.966 | **0.786** | 0.966 |

Loss falls `5.76 → 0.97` in five tiny steps — training a neural net is exactly this, millions of times, with `∇L` computed by backprop.

**⚠️ Stability check** — on this bowl the rule simplifies to `w ← (1−2η)·w`:

| η | `abs(1−2η)` | outcome |
|---|---|---|
| 0.10 | 0.80 | ✅ stable but slow |
| 0.85 | 0.70 | ✅ fast (zig-zags near the bottom) |
| 1.12 | 1.24 | ❌ **diverges** — each step multiplies the error by 1.24 |

**🖼️ Figures — convex vs non-convex landscapes · learning-rate regimes · batch vs mini-batch vs SGD · Lagrange tangency · grid/random/Bayesian search:**

![Convex function has one optimum; non-convex loss has many valleys](../figures/00-math/21_convex_nonconvex.png)

![Gradient descent trajectories for learning rates 0.1, 0.85, and 1.12 where the last diverges](../figures/00-math/22_learning_rate.png)

![Batch, mini-batch, and SGD loss curves with different noise levels](../figures/00-math/23_batch_vs_sgd.png)

![Constrained optimum where the gradient of f is parallel to the gradient of the constraint](../figures/00-math/24_lagrange.png)

![Grid search versus random search versus Bayesian optimization on the same function](../figures/00-math/25_grid_random_bayes.png)

**💡 Simple examples:**
- **Learning rate (exercise #5):** on `L=x²`, update `x ← x − η·2x` is stable iff `|1−2η| < 1` → **η must be < 1**; `η=1.12` explodes (figure 22).
- **Batch math:** full-batch epoch = `n` examples per update; pure SGD = 1 → 100× more updates, noisy path, often better generalization (figure 23).
- **Lagrange:** minimize `(x−4)²+(y−3)²` subject to `x+y=4` → `∇f = λ∇g` requires `x−4 = y−3`, plus the constraint → optimum at **(2.5, 1.5)** with `∇f = (−3,−3) = −3·∇g` (λ = −3) — `λ` = price of the constraint (figure 24 — SVMs & RLHF's KL budget use exactly this).
- **Search:** 9 evals — grid misses the peak between ticks, random sometimes hits it, Bayesian clusters its budget around the incumbent (figure 25).

## 6. Information Theory — measuring information

- **Entropy** `H(p) = −Σ p·log p` — surprise/uncertainty; the floor of lossless compression.
- **Cross-entropy** `H(p,q) = −Σ p·log q` — *the* classification loss; why `CrossEntropyLoss = softmax + log + NLL`.
- **KL divergence** — how one distribution differs from another; basis of distillation and RLHF's KL penalty.
- **Mutual information** — feature selection, information bottleneck.

**🧮 Formula sheet**  (`log₂` = bits · `ln` = nats · 1 nat = 1.44 bits)

| Quantity | Formula | Meaning |
|---|---|---|
| Entropy | `H(p) = −Σ p·log₂ p` | expected surprise; max at 50/50 |
| Cross-entropy | `H(p,q) = −Σ p·log₂ q` | *the* classification loss |
| KL divergence | `KL(P‖Q) = Σ p·log₂(p/q)` | extra bits to code P using Q (asymmetric!) |
| Mutual information | `I(X;Y) = H(X) − H(X\|Y)` | how many bits X and Y share |

**🔢 Worked — entropy of coins**

| coin | calculation | H |
|---|---|---|
| fair | `−0.5·log₂0.5 − 0.5·log₂0.5 = 0.5 + 0.5` | **1.000 bit** |
| biased (p=0.9) | `−0.9·log₂0.9 − 0.1·log₂0.1 = 0.137 + 0.332` | **0.469 bits** |
| always heads | `−1·log₂1` | **0 bits** (no news) |

**🔢 Worked — the cross-entropy sting** (true class has model prob `q`)

| model says q | loss `−log₂ q` | compared to baseline |
|---|---|---|
| 0.9 — confident & right | **0.152 bits** | baseline |
| 0.5 — pure guess | 1.000 bit | 6.6× worse |
| 0.001 — confident & WRONG | **9.966 bits** | **65.6× worse** |

**🔢 Worked — KL asymmetry** with `P=(0.5, 0.5)`, `Q=(0.25, 0.75)`

- `KL(P‖Q) = 0.5·log₂(2) + 0.5·log₂(2/3) = 0.500 − 0.292 =` **0.208 bits**
- `KL(Q‖P) = 0.25·log₂(0.5) + 0.75·log₂(1.5) = −0.250 + 0.439 =` **0.189 bits**
- same two distributions, reversed direction → **different number**. That's why RLHF/distillation must state *which* KL they use (figure 28).

**🖼️ Figures — entropy curve & surprise · cross-entropy loss explosion · asymmetric KL · mutual-information heatmaps:**

![Entropy of a coin flip peaks at 1 bit and surprise bars for rarer events](../figures/00-math/26_entropy.png)

![Cross-entropy loss of confident-right versus confident-wrong predictions](../figures/00-math/27_cross_entropy.png)

![KL divergence between two distributions is asymmetric](../figures/00-math/28_kl_divergence.png)

![Joint heatmaps showing high versus zero mutual information](../figures/00-math/29_mutual_information.png)

**💡 Simple examples:**
- **Entropy:** fair coin `H = 1 bit`; always-heads coin `H = 0` — zero surprise carries zero information (figure 26).
- **Cross-entropy (exercise #4):** confident-right `q=.99 → −log₂ = 0.015 bits`; confident-wrong `q=.001 → 9.97 bits` — the loss explodes ~660× (figure 27). Why `CrossEntropyLoss = softmax → log → NLL`.
- **KL asymmetry:** `P=(.45,.35,.15,.05)`, `Q=(.20,.30,.30,.20)` → `KL(P‖Q) ≠ KL(Q‖P)` though both ≥ 0 — direction matters, which is why distillation and RLHF pick one deliberately (figure 28).
- **Mutual information:** `I(X;Y) = 0` iff independent — two perfectly separable class clusters have high `I`; white-noise heatmap ≈ 0 bits (figure 29).

Visualize every topic of module 00 with 31 generated figures, and machine-check
every worked number in the tables above:

```bash
python3 tools/make_math_figures.py   # writes docs/figures/00-math/*.png (~1.3MB)
python3 tools/calc_00_math.py        # recomputes + asserts every worked calculation
```

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

**Checklist → figure map (self-test by explaining each figure aloud):**

| Checklist item | Figures that prove it |
|---|---|
| Matrix multiplication in a layer / shapes | [01](../figures/00-math/01_vectors_dot.png), [02](../figures/00-math/02_matmul_shapes.png) |
| Chain rule on a 2-layer network | [07](../figures/00-math/07_chain_graph.png), [08](../figures/00-math/08_partials.png) |
| Bayes' rule + 5 distributions | [10](../figures/00-math/10_bayes_test.png), [11](../figures/00-math/11_distributions.png) |
| MLE & cross-entropy as MLE | [14](../figures/00-math/14_likelihood.png), [19](../figures/00-math/19_mle_map.png), [27](../figures/00-math/27_cross_entropy.png) |
| Learning = descending the gradient | [06](../figures/00-math/06_gradient_field.png), [22](../figures/00-math/22_learning_rate.png), [23](../figures/00-math/23_batch_vs_sgd.png) |
| Entropy / cross-entropy / KL | [26](../figures/00-math/26_entropy.png), [27](../figures/00-math/27_cross_entropy.png), [28](../figures/00-math/28_kl_divergence.png) |

Figures are generated by [`tools/make_math_figures.py`](../../tools/make_math_figures.py) — rerun it after editing to refresh `docs/figures/00-math/`.
