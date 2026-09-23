"""Generate didactic figures for docs/curriculum/00-mathematical-foundations.md.

One figure per topic across all six sections (linear algebra, calculus,
probability, statistics, optimization, information theory). Output:
    docs/figures/00-math/NN_topic.png
Run:  python3 tools/make_math_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "00-math"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 100, "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "figure.facecolor": "white",
    "axes.titlesize": 10, "axes.titleweight": "bold",
})
C = {"b": "#2563eb", "r": "#dc2626", "g": "#16a34a", "o": "#f59e0b",
     "p": "#7c3aed", "k": "#111827"}


def save(fig, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name:<34} {path.stat().st_size // 1024:>3} KB")
    return path


# =========================================================================
# §1 Linear algebra
# =========================================================================
def fig_vectors_dot():
    """Dot product = cosine similarity: geometry of two embeddings."""
    fig, ax = plt.subplots(figsize=(5.2, 4))
    a, b = np.array([3.0, 1.0]), np.array([1.0, 3.0])
    for v, c, lab in ((a, C["b"], "a = (3,1)"), (b, C["r"], "b = (1,3)")):
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
        ax.text(*v * 1.05, lab, color=c, fontsize=10, weight="bold")
    ax.plot(*np.vstack([a * 0, a * np.dot(a, b) / np.dot(a, a)]),  # projection
            ls="--", color=C["b"], lw=1)
    th = np.linspace(0, np.arctan2(3, 1) - np.arctan2(1, 3), 30)
    ang = np.arctan2(1, 3)
    ax.plot(0.8 * np.cos(ang + th), 0.8 * np.sin(ang + th), color=C["k"], lw=1)
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    ax.text(1.0, 0.45, f"θ = {np.degrees(np.arccos(cos)):.1f}°")
    ax.set_title(f"a·b = {np.dot(a, b)}   |a|={np.linalg.norm(a):.2f} "
                 f"|b|={np.linalg.norm(b):.2f}   cos θ = {cos:.2f}")
    ax.set_xlim(-0.4, 4.2); ax.set_ylim(-0.4, 4.2)
    ax.set_aspect("equal"); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    ax.annotate("", xy=(0, 0), xytext=b,
                arrowprops=dict(arrowstyle="<|-", color="gray", lw=0.8,
                                ls=":"))
    return save(fig, "01_vectors_dot.png")


def fig_matmul_shapes():
    """Matrix multiply as block shapes: inner dims must agree."""
    rng = np.random.default_rng(0)
    W, X = rng.normal(size=(2, 3)), rng.normal(size=(3, 2))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    mats = [("W", W, (2, 3), 0.0), ("X", X, (3, 2), 2.6),
            ("Y = WX", W @ X, (2, 2), 5.2)]
    for name, M, shape, x in mats:
        ax.imshow(M, cmap="coolwarm", extent=(x, x + shape[1] * 0.42,
                                               0, shape[0] * 0.42))
        ax.text(x + shape[1] * 0.21, -0.35, f"{name} {shape}", ha="center",
                fontsize=9, weight="bold")
    ax.annotate("", xy=(2.5, 1.4), xytext=(2.0, 1.4),
                arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.annotate("", xy=(5.05, 1.4), xytext=(4.55, 1.4),
                arrowprops=dict(arrowstyle="-|>", lw=1.5))
    ax.text(2.45, 1.62, "inner 3 = 3 ✓", color=C["g"], weight="bold",
            fontsize=8)
    ax.text(0.0, 2.05, "(2×3)·(3×2) → (2×2): each output row = "
            "one sample's features ⊗ weights", fontsize=8)
    ax.set_title("Y = W·X  — one line scores a whole mini-batch")
    ax.axis("off")
    return save(fig, "02_matmul_shapes.png")


def fig_eigen_pca():
    """Eigenvectors of covariance = principal axes (PCA)."""
    rng = np.random.default_rng(1)
    th = rng.uniform(0, 2 * np.pi, 300)
    r = rng.normal(0, 1, 300)
    P = np.c_[2.2 * r * np.cos(th) + 0.4 * rng.normal(size=300),
              0.7 * r * np.sin(th) + 0.4 * rng.normal(size=300)]
    P = P @ np.array([[1, 0.5], [0, 1]])           # correlate the axes
    mu = P.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((P - mu).T))
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    ax.scatter(P[:, 0], P[:, 1], s=6, alpha=0.35, color=C["b"])
    for i, (val, vec) in enumerate(zip(evals, evecs.T)):
        ax.annotate("", xy=mu + 2 * np.sqrt(val) * vec, xytext=mu,
                    arrowprops=dict(arrowstyle="-|>", color=C["r"] if i == 0
                                    else C["g"], lw=2.5))
        ax.text(*(mu + 2.15 * np.sqrt(val) * vec),
                f"v{i+1} (λ={val:.2f})", color=C["r"] if i == 0 else C["g"],
                weight="bold", fontsize=9)
    ax.set_title("Eigendecomposition of covariance = PCA directions")
    ax.set_aspect("equal")
    return save(fig, "03_eigen_pca.png")


def fig_svd_lowrank():
    """SVD / low-rank: how few numbers approximate a matrix (LoRA's idea)."""
    rng = np.random.default_rng(2)
    A = rng.normal(size=(10, 10)) @ rng.normal(size=(10, 10))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.6))
    for ax, k in zip(axes, (10, 5, 2, 1)):
        Ak = (U[:, :k] * s[:k]) @ Vt[:k]
        err = np.linalg.norm(A - Ak) / np.linalg.norm(A)
        ax.imshow(Ak, cmap="viridis")
        ax.set_title(f"rank {k} · err {err:.0%}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("SVD truncation: store k·(m+n) instead of m·n numbers",
                 fontsize=10, weight="bold")
    return save(fig, "04_svd_lowrank.png")


# =========================================================================
# §2 Calculus & gradients
# =========================================================================
def fig_derivative():
    """Derivative = slope of the tangent (secant → limit)."""
    x = np.linspace(-2.6, 2.2, 400)
    f = lambda t: t ** 3 - t
    df = lambda t: 3 * t ** 2 - 1
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, x0, ttl in ((axes[0], -1.5, "secant (average rate)"),
                        (axes[1], 1.3, "tangent = derivative at x₀")):
        h = 0.9 if x0 < 0 else 0.35
        sec = (f(x0 + h) - f(x0)) / h
        ax.plot(x, f(x), color=C["b"], lw=2, label="f(x) = x³ − x")
        ax.plot([x0, x0 + h], [f(x0), f(x0 + h)], color=C["o"], lw=2,
                label="secant")
        if ttl.startswith("tangent"):
            ax.plot(x, f(x0) + df(x0) * (x - x0), color=C["r"], lw=2,
                    label=f"slope f′(x₀) = {df(x0):.2f}")
        ax.plot(x0, f(x0), "ko", ms=6)
        ax.set_title(ttl); ax.legend(fontsize=8, loc="upper left")
        ax.set_ylim(-4, 4)
    fig.suptitle("f′(x) = limₕ→₀ [f(x+h) − f(x)] / h   — sensitivity of output "
                 "to input", fontsize=10, weight="bold")
    return save(fig, "05_derivative_tangent.png")


def fig_gradient_field():
    """∇L points uphill; learning moves along −∇L."""
    x, y = np.meshgrid(np.linspace(-2.2, 2.2, 15), np.linspace(-2.2, 2.2, 15))
    f = lambda X, Y: X ** 2 + 2.5 * Y ** 2
    gx, gy = 2 * x, 5 * y
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.contour(x, y, f(x, y), levels=np.linspace(0.2, 9, 10),
               cmap="Blues_r", linewidths=0.8)
    ax.quiver(x, y, gx, gy, color=C["r"], alpha=0.8, scale=30,
              width=0.004, label="∇L (uphill)")
    ax.quiver(x, y, -gx, -gy, color=C["g"], alpha=0.8, scale=30,
              width=0.004, label="−∇L (learning direction)")
    ax.plot(0, 0, "k*", ms=14, label="minimum")
    ax.set_title("Gradient field of L(x,y) = x² + 2.5y²")
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="lower right")
    return save(fig, "06_gradient_field.png")


def fig_chain_graph():
    """Chain rule = backprop: multiply local derivatives along the graph."""
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    boxes = [("x = 1", 0.6, C["k"]), ("g = 2x+1 = 3", 2.6, C["b"]),
             ("L = g³ = 27", 4.9, C["r"])]
    for txt, xp, c in boxes:
        ax.add_patch(plt.Rectangle((xp - 0.75, 0.65), 1.5, 0.7,
                                   fc="white", ec=c, lw=2))
        ax.text(xp, 1.0, txt, ha="center", va="center", fontsize=9,
                weight="bold", color=c)
    for x0, x1, lab in ((1.38, 1.82, "∂g/∂x = 2"), (3.38, 4.12, "∂L/∂g = 27")):
        ax.annotate("", xy=(x1, 1.0), xytext=(x0, 1.0),
                    arrowprops=dict(arrowstyle="-|>", color=C["p"], lw=2))
        ax.text((x0 + x1) / 2, 1.52, lab, ha="center", color=C["p"],
                fontsize=9, weight="bold")
    ax.text(2.75, 0.15, "dL/dx = (∂L/∂g)·(∂g/∂x) = 27 × 2 = 54",
            ha="center", fontsize=11, weight="bold", color=C["k"],
            bbox=dict(fc="#fef9c3", ec=C["o"]))
    ax.set_xlim(0, 5.8); ax.set_ylim(-0.4, 2.3); ax.axis("off")
    ax.set_title("Computation graph: backprop = chain rule, node by node")
    return save(fig, "07_chain_graph.png")


def fig_partials():
    """Partial derivatives: one axis at a time; gradient = all of them."""
    x1, x2 = np.meshgrid(np.linspace(-2, 2, 60), np.linspace(-2, 2, 60))
    f = lambda X, Y: X ** 2 + 2 * Y ** 2
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    axes[0].contour(x1, x2, f(x1, x2), levels=12, cmap="viridis")
    p = (1.0, 1.0)
    axes[0].plot(*p, "r*", ms=14)
    axes[0].text(p[0] + 0.15, p[1], "(1,1)", fontsize=9)
    axes[0].set_title("L(x₁,x₂) = x₁² + 2x₂²")
    t = np.linspace(-2, 2, 100)
    axes[1].plot(t, t ** 2 + 2, label="∂L/∂x₁ = 2x₁ (at x₂=1)")
    axes[1].plot(t, 1 + 2 * t ** 2, label="∂L/∂x₂ = 4x₂ (at x₁=1)")
    axes[1].plot(1, 2, "b*", ms=13); axes[1].plot(1, 3, "r*", ms=13)
    axes[1].set_title("each partial = slope along one axis")
    axes[1].legend(fontsize=8)
    axes[2].bar(["∂L/∂x₁", "∂L/∂x₂"], [2, 4], color=[C["b"], C["r"]])
    axes[2].set_title("∇L = (2, 4) — the full direction")
    axes[2].set_ylim(0, 5)
    fig.tight_layout()
    return save(fig, "08_partials.png")


def fig_hessian():
    """Second derivatives → curvature: min, max, saddle."""
    x, y = np.meshgrid(np.linspace(-2, 2, 60), np.linspace(-2, 2, 60))
    cases = [("minimum  λ>0: +1,+2", x ** 2 + 2 * y ** 2, C["g"]),
             ("saddle  λ mixed: +1,−1", x ** 2 - y ** 2, C["o"]),
             ("maximum  λ<0: −1,−1", -x ** 2 - 0.7 * y ** 2, C["r"])]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    for ax, (ttl, F, c) in zip(axes, cases):
        ax.contour(x, y, F, levels=10, cmap="coolwarm", linewidths=0.9)
        ax.plot(0, 0, "*", color=c, ms=15)
        ax.set_title(ttl, fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Hessian eigenvalues classify critical points — "
                 "saddles stall naive optimizers", fontsize=10, weight="bold")
    return save(fig, "09_hessian_curvature.png")


# =========================================================================
# §3 Probability
# =========================================================================
def fig_bayes_test():
    """Medical-test paradox: 1% prevalence, 90% sensitive, 5% false-positive."""
    prev, sens, fpr = 0.01, 0.90, 0.05
    n = 1000
    tp, fn = n * prev * sens, n * prev * (1 - sens)
    fp = n * (1 - prev) * fpr
    post = tp / (tp + fp)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6),
                             gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    ax.bar(["TP\n(sick & +)", "FP\n(healthy & +)"], [tp, fp],
           color=[C["r"], C["o"]])
    for i, v in enumerate([tp, fp]):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", weight="bold")
    ax.set_title("Of 1000 tested, who actually tests positive?")
    ax.set_ylabel("people")
    ax = axes[1]
    ax.bar(["prior\nP(sick)", "posterior\nP(sick|+)"], [prev * 100, post * 100],
           color=["gray", C["r"]])
    ax.set_ylabel("%")
    for i, v in enumerate([prev * 100, post * 100]):
        ax.text(i, v + 0.6, f"{v:.1f}%", ha="center", weight="bold")
    ax.set_title(f"Bayes lifts 1% → {post:.1%}")
    axes[0].set_ylim(0, 58)                     # headroom for value labels
    axes[1].set_ylim(0, 18)
    fig.suptitle("P(H|D) = P(D|H)·P(H) / P(D)  — base rates dominate",
                 fontsize=10, weight="bold")
    fig.tight_layout()
    return save(fig, "10_bayes_test.png")


def fig_distributions():
    """The named distributions the syllabus demands."""
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5))
    ax = axes[0, 0]
    ax.bar([0, 1], [0.7, 0.3], color=[C["b"], C["r"]]); ax.set_xticks([0, 1])
    ax.set_title("Bernoulli(p=0.3)\n— binary labels", fontsize=9)
    ax = axes[0, 1]
    ax.bar([0, 1, 2], [0.2, 0.5, 0.3], color=C["b"]); ax.set_xticks([0, 1, 2])
    ax.set_title("Categorical([.2,.5,.3])\n— K-class labels", fontsize=9)
    ax = axes[0, 2]
    xs = np.linspace(-4, 7, 300)
    ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color=C["b"],
            label="N(0,1)")
    ax.plot(xs, np.exp(-(xs - 3) ** 2 / 8) / np.sqrt(8 * np.pi), color=C["r"],
            label="N(3,2)")
    ax.legend(fontsize=8)
    ax.set_title("Gaussian\n— noise, VAE priors", fontsize=9)
    ax = axes[1, 0]
    ks = np.arange(0, 8)
    lam = 3.0
    from math import factorial
    pmf = np.exp(-lam) * lam ** ks / [factorial(int(k)) for k in ks]
    ax.bar(ks, pmf, color=C["b"])
    ax.set_title("Poisson(3)\n— count data", fontsize=9)
    ax = axes[1, 1]
    ax.bar([0, 1], [1, 1], color=C["g"], width=1.0, edgecolor="k",
           align="edge")
    ax.set_xticks([0.0, 0.5, 1.0]); ax.set_ylim(0, 1.4)
    ax.set_title("Uniform[0,1]\n— init, sampling", fontsize=9)
    ax = axes[1, 2]
    ax.axis("off")
    ax.text(0, 0.95, "Pick by shape:", fontsize=10, weight="bold")
    ax.text(0, 0.8, "• 2 outcomes → Bernoulli\n• K outcomes → Categorical\n"
            "• counts → Poisson\n• real-valued noise → Gaussian\n"
            "• no info → Uniform", fontsize=9, va="top")
    fig.tight_layout()
    return save(fig, "11_distributions.png")


def fig_expectation_variance():
    """E[X] and Var(X) read straight off the sample."""
    rng = np.random.default_rng(0)
    x = rng.normal(5, 2, 4000)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.hist(x, bins=45, density=True, color=C["b"], alpha=0.65)
    xs = np.linspace(-3, 13, 300)
    ax.plot(xs, np.exp(-(xs - 5) ** 2 / 8) / np.sqrt(8 * np.pi), color="k",
            lw=1.5)
    ax.axvline(5, color=C["r"], lw=2, label=f"E[X] ≈ {x.mean():.2f} (μ=5)")
    ax.axvline(3, color=C["o"], ls="--", label="μ−σ = 3")
    ax.axvline(7, color=C["o"], ls="--", label="μ+σ = 7")
    ax.set_title(f"Var(X) = E[X²]−(E[X])² ≈ {x.var():.2f}  (σ²=4)")
    ax.legend(fontsize=8)
    return save(fig, "12_expectation_variance.png")


def fig_joint_marginal():
    """Joint P(X,Y) and marginals P(X)=Σ_Y P(X,Y)."""
    px = np.array([0.15, 0.35, 0.5])
    py_given = np.array([[0.6, 0.4], [0.3, 0.7], [0.2, 0.8]])
    J = px[:, None] * py_given
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5),
                             gridspec_kw={"width_ratios": [3, 1],
                                          "height_ratios": [1, 3],
                                          "hspace": 0.05, "wspace": 0.05})
    axes[0, 0].imshow(px[None, :], cmap="Blues", aspect="auto", vmin=0)
    axes[0, 0].set_title("P(X) = Σ_Y P(X,Y)", fontsize=9)
    axes[0, 0].set_xticks([]); axes[0, 0].set_yticks([])
    axes[1, 0].imshow(J, cmap="Blues", vmin=0)
    axes[1, 0].set_xlabel("Y"); axes[1, 0].set_ylabel("X")
    axes[1, 0].set_xticks([0, 1]); axes[1, 0].set_yticks([0, 1, 2])
    for i in range(3):
        for j in range(2):
            axes[1, 0].text(j, i, f"{J[i, j]:.2f}", ha="center", color="w",
                            fontsize=9)
    axes[1, 0].set_title("joint P(X,Y)", fontsize=9)
    axes[1, 1].imshow(J.sum(0)[None, :], cmap="Blues", aspect="auto")
    axes[1, 1].set_title("P(Y)", fontsize=9)
    axes[1, 1].set_xticks([]); axes[1, 1].set_yticks([])
    axes[0, 1].axis("off")
    fig.suptitle("marginals are projections: sum the other axis",
                 fontsize=10, weight="bold")
    return save(fig, "13_joint_marginal.png")


def fig_likelihood():
    """Likelihood P(data|μ): maximize it → MLE."""
    rng = np.random.default_rng(3)
    data = rng.normal(1.5, 0.7, 12)
    mus = [0.2, 1.5, 2.8]
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.2), sharex=True)
    xs = np.linspace(-2, 5, 300)
    axes[0].scatter(data, np.full_like(data, -0.06), color=C["k"], s=28,
                    zorder=3, label="observed data")
    for mu, c in zip(mus, [C["o"], C["g"], C["r"]]):
        axes[0].plot(xs, np.exp(-(xs - mu) ** 2 / (2 * 0.7 ** 2)),
                     color=c, label=f"assumed μ={mu}")
    axes[0].set_title("P(data | μ) for three candidate models")
    axes[0].legend(fontsize=8, loc="upper right"); axes[0].set_yticks([])
    grid = np.linspace(-1, 4, 200)
    ll = np.array([np.log(np.exp(-(data - m) ** 2 / (2 * 0.7 ** 2))
                          + 1e-300).sum() for m in grid])
    axes[1].plot(grid, ll - ll.max(), color=C["b"])
    axes[1].axvline(grid[np.argmax(ll)], color=C["g"], ls="--",
                    label=f"MLE μ̂ = {grid[np.argmax(ll)]:.2f} (true 1.5)")
    axes[1].set_title("log-likelihood vs μ — the argmax is the fit")
    axes[1].set_xlabel("μ"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    return save(fig, "14_likelihood.png")


def fig_monte_carlo():
    """Monte Carlo: area by sampling; error shrinks like 1/√n."""
    rng = np.random.default_rng(4)
    pts = rng.uniform(-1, 1, (2000, 2))
    inside = (pts ** 2).sum(1) <= 1
    est = np.cumsum(inside) / np.arange(1, len(inside) + 1) * 4
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    axes[0].scatter(pts[~inside, 0], pts[~inside, 1], s=2, color="gray")
    axes[0].scatter(pts[inside, 0], pts[inside, 1], s=2, color=C["b"])
    th = np.linspace(0, np.pi / 2, 100)
    axes[0].plot(np.cos(th), np.sin(th), color=C["r"], lw=2)
    axes[0].set_aspect("equal")
    axes[0].set_title(f"π ≈ 4·(inside/total) = {est[-1]:.3f}   (true 3.14159)")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].plot(est, color=C["b"], lw=1.5)
    axes[1].axhline(np.pi, color=C["r"], ls="--", label="π")
    axes[1].set_xlabel("samples n"); axes[1].set_ylabel("estimate")
    axes[1].set_title("error ~ 1/√n — averaging kills noise")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return save(fig, "15_monte_carlo.png")


# =========================================================================
# §4 Statistics
# =========================================================================
def fig_estimator_biasvar():
    """Bias–variance OF ESTIMATORS: accuracy vs precision of an estimate."""
    rng = np.random.default_rng(5)
    truth = 0.5
    unbiased = truth + rng.normal(0, 0.18, 300)      # centered, spread out
    biased = 0.36 + rng.normal(0, 0.06, 300)         # tight but off-target
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    for ax, s, ttl, c in ((axes[0], unbiased, "unbiased: mean ≈ truth", C["g"]),
                          (axes[1], biased, "biased: precise but wrong", C["r"])):
        ax.hist(s, bins=28, color=c, alpha=0.7)
        ax.axvline(truth, color="k", lw=2, ls="--", label="truth 0.5")
        ax.axvline(s.mean(), color=c, lw=2, label=f"avg est {s.mean():.2f}")
        ax.set_title(f"{ttl}\nspread(Var)={s.var():.3f}", fontsize=9)
        ax.legend(fontsize=7); ax.set_xlim(0.1, 0.9)
    fig.suptitle("Estimator quality = low bias AND low variance "
                 "(averaging reduces Var, not bias)", fontsize=10,
                 weight="bold")
    return save(fig, "16_estimator_biasvar.png")


def fig_ci_pvalue():
    """Confidence intervals & p-values — the A/B-test toolkit."""
    rng = np.random.default_rng(6)
    effects = rng.normal(0.35, 0.5, 10)               # true lift 0.35
    lo, hi = effects - 1.96 * 0.25, effects + 1.96 * 0.25
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    ax = axes[0]
    for i, (l, h) in enumerate(zip(lo, hi)):
        sig = not (l <= 0 <= h)
        ax.plot([l, h], [i, i], lw=3, color=C["g"] if sig else "gray")
    ax.axvline(0, color=C["r"], lw=1.5, ls="--", label="null: no effect")
    ax.set_title("95% CIs of 10 A/B repeats\n"
                 "green = CI excludes 0 → significant", fontsize=9)
    ax.set_xlabel("measured lift"); ax.legend(fontsize=8)
    ax = axes[1]
    null = rng.normal(0, 1, 20000)
    ax.hist(null, bins=80, density=True, color="gray", alpha=0.7)
    tail = null >= 1.96
    ax.hist(null[tail], bins=80, density=True, color=C["r"], alpha=0.9)
    ax.axvline(1.96, color=C["r"], ls="--")
    ax.axvline(-1.96, color=C["r"], ls="--")
    ax.set_title("null distribution: p = red tails ≈ 5%\n"
                 "p = P(data this extreme | H₀ true)", fontsize=9)
    fig.tight_layout()
    return save(fig, "17_ci_pvalue.png")


def fig_correlation_causation():
    """Same-looking correlation, three different truths."""
    rng = np.random.default_rng(7)
    x = rng.uniform(1, 10, 40)
    y1 = 2 * x + rng.normal(0, 2, 40)
    y2 = 30 / (x + 1) + rng.normal(0, 1.2, 40)
    y3 = 1.1 * x + rng.normal(0, 2, 40); y3[0] = -18   # plant outlier
    panels = [
        (f"x causes y (r={np.corrcoef(x, y1)[0, 1]:.2f})", y1, C["b"]),
        (f"curved, not linear (r={np.corrcoef(x, y2)[0, 1]:.2f})", y2, C["o"]),
        (f"one outlier fakes r={np.corrcoef(x, y3)[0, 1]:.2f}", y3, C["g"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2), sharey=True)
    for ax, (ttl, y, c) in zip(axes, panels):
        ax.scatter(x, y, color=c, s=24)
        ax.set_title(ttl, fontsize=9)
    fig.suptitle("Correlation ≠ causation: always ask about confounders "
                 "(ice-cream sales 'cause' drownings? no — heat does)",
                 fontsize=10, weight="bold")
    return save(fig, "18_correlation_causation.png")


def fig_mle_map():
    """MLE vs MAP: likelihood alone vs likelihood × prior."""
    rng = np.random.default_rng(8)
    data = rng.normal(2.0, 0.9, 8)
    grid = np.linspace(-1, 5, 400)
    log_lik = np.array([np.log(np.exp(-(data - m) ** 2 / (2 * 0.81))
                               + 1e-300).sum() for m in grid])
    log_lik -= log_lik.max()
    log_prior = -((grid - 1.2) ** 2) / (2 * 0.6 ** 2)   # prior says ~1.2
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(grid, np.exp(log_lik), color=C["b"], lw=2,
            label="likelihood P(D|μ)")
    ax.plot(grid, np.exp(log_prior - log_prior.max()),
            color=C["o"], lw=2, ls="--", label="prior P(μ)")
    post = log_lik + log_prior
    post -= post.max()
    ax.plot(grid, np.exp(post), color=C["r"], lw=2.4,
            label="posterior ∝ lik × prior (MAP)")
    ax.axvline(grid[np.argmax(log_lik)], color=C["b"], ls=":", lw=1.2)
    ax.axvline(grid[np.argmax(post)], color=C["r"], ls=":", lw=1.2)
    ax.text(2.0, -0.12, "MLE\n(data wins)", ha="center", fontsize=8,
            color=C["b"])
    ax.text(grid[np.argmax(post)], -0.12, "MAP\n(compromise)", ha="center",
            fontsize=8, color=C["r"])
    ax.set_xlabel("parameter μ"); ax.set_ylim(-0.2, 1.15)
    ax.set_title("MLE = argmax likelihood · MAP = argmax likelihood × prior")
    ax.legend(fontsize=8, loc="upper left")
    return save(fig, "19_mle_map.png")


def fig_covariance_correlation():
    """Covariance vs correlation: scale-invariant association."""
    rng = np.random.default_rng(9)
    fig, axes = plt.subplots(2, 1, figsize=(5.6, 5.6),
                             gridspec_kw={"height_ratios": [3, 2]})
    ax0 = axes[0]
    for i, (rho, c) in enumerate([(0.9, C["b"]), (-0.7, C["r"])]):
        z1 = rng.normal(size=300)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.normal(size=300)
        ax0.scatter(z1 + (6 if i else 0), z2, s=8, alpha=0.5, color=c)
        r = np.corrcoef(z1, z2)[0, 1]
        cov = np.cov(z1, z2)[0, 1]
        ax0.text((6 if i else 0) + 0.5, 4.0,
                 f"ρ={r:.2f}  cov={cov:.2f}", color=c, fontsize=9,
                 weight="bold", ha="center")
    ax0.set_title("correlation = covariance normalized (scale-free)")
    axes[1].imshow([[1.0, 0.62], [0.62, 1.0]], cmap="RdBu_r",
                   vmin=-1, vmax=1)
    axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["f1", "f2"]); axes[1].set_yticklabels(["f1", "f2"])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f"{[[1.0, .62], [.62, 1.0]][i][j]:.2f}",
                         ha="center", va="center", color="w", fontsize=10)
    axes[1].set_title("correlation matrix → input to PCA", fontsize=9)
    axes[1].grid(False)
    fig.tight_layout()
    return save(fig, "20_covariance_correlation.png")


# =========================================================================
# §5 Optimization
# =========================================================================
def fig_convex_nonconvex():
    """Convex ⇒ global optimum; deep loss landscapes ⇒ 'good enough'."""
    x = np.linspace(-3.2, 4.2, 400)
    convex = 0.45 * (x - 0.5) ** 2 + 0.4
    nonconvex = (np.sin(1.7 * x) * np.exp(-0.16 * x ** 2) + 0.35 * x ** 2
                 - 0.2 * x)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    axes[0].plot(x, convex, color=C["b"], lw=2)
    axes[0].plot(0.5, 0.4, "*", color=C["g"], ms=16)
    axes[0].set_title("convex: ONE minimum ⇒ gradient descent finds THE optimum",
                      fontsize=9)
    axes[1].plot(x, nonconvex, color=C["r"], lw=2)
    for m in (-1.05, 0.62, 2.9):
        axes[1].plot(m, np.interp(m, x, nonconvex), "o", color="gray", ms=8)
    gbest = x[np.argmin(nonconvex)]
    axes[1].plot(gbest, nonconvex.min(), "*", color=C["g"], ms=16)
    axes[1].plot(0.62, np.interp(0.62, x, nonconvex), "*", color=C["o"],
                 ms=14)
    axes[1].set_title("non-convex: many valleys ⇒ stop at first GOOD one "
                      "(orange ≠ green)", fontsize=9)
    fig.suptitle("Why deep learning optimizes for 'good enough' — convexity "
                 "is abandoned on purpose", fontsize=10, weight="bold")
    return save(fig, "21_convex_nonconvex.png")


def fig_learning_rate():
    """Learning rate: too slow / just right / diverges (on f(x)=x²)."""
    f = lambda t: t ** 2
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.linspace(-3, 3, 400)
    ax.plot(x, f(x), color="gray", lw=1.5, label="L(x) = x²")
    for lr, c, lab in ((0.1, C["b"], "η=0.1 slow but steady"),
                       (0.85, C["g"], "η=0.85 fast zig-zag"),
                       (1.12, C["r"], "η=1.12 DIVERGES")):
        pts = [2.4]
        for _ in range(9):
            pts.append(pts[-1] - lr * 2 * pts[-1])     # x ← x − η·2x
        ax.plot(pts, f(np.array(pts)), "o-", color=c, ms=3.5, lw=1.2,
                label=lab)
    ax.set_ylim(-0.5, 26)
    ax.set_title("gradient descent on L=x²:  x ← x − η·2x   "
                 "(|1−2η|<1 ⇒ stable)", fontsize=9)
    ax.legend(fontsize=8, loc="upper center")
    return save(fig, "22_learning_rate.png")


def fig_batch_vs_sgd():
    """Batch / mini-batch / SGD — noise vs speed per epoch."""
    rng = np.random.default_rng(11)
    steps = 60
    t = np.arange(steps)
    base = 1.2 * np.exp(-0.09 * t)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.plot(t, base, lw=2.4, color=C["b"],
            label="batch GD  (smooth, costly per step)")
    ax.plot(t, base + rng.normal(0, 0.045, steps), lw=1.2, color=C["g"],
            label="mini-batch  (practical default)")
    ax.plot(t, base + rng.normal(0, 0.16, steps), lw=0.9, color=C["r"],
            label="pure SGD  (noisy — escapes sharp minima)")
    ax.set_yscale("log"); ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log)")
    ax.set_title("same destination, different noise: batch=full data, "
                 "SGD=1 sample", fontsize=9)
    ax.legend(fontsize=8)
    return save(fig, "23_batch_vs_sgd.png")


def fig_lagrange():
    """Constrained optimization: optimum where contours touch the constraint."""
    x, y = np.meshgrid(np.linspace(-1, 5, 120), np.linspace(-1, 5, 120))
    f = lambda X, Y: (X - 4) ** 2 + (Y - 3) ** 2
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.contour(x, y, f(x, y), levels=12, cmap="viridis", linewidths=0.9)
    xs = np.linspace(-0.5, 4.5, 100)
    ax.plot(xs, 4 - xs, color=C["r"], lw=2.4,
            label="constraint g(x,y)=x+y−4=0")
    opt = (2.0, 2.0)                          # point on line closest to (4,3)
    ax.plot(*opt, "*", color=C["o"], ms=18,
            label="constrained optimum")
    ax.annotate("", xy=(3.4, 3.4), xytext=opt,
                arrowprops=dict(arrowstyle="-|>", color=C["b"], lw=2.4))
    ax.text(3.5, 3.1, "∇f", color=C["b"], fontsize=11, weight="bold")
    ax.annotate("", xy=(1.4, 2.6), xytext=opt,
                arrowprops=dict(arrowstyle="-|>", color=C["g"], lw=2.4))
    ax.text(1.1, 2.7, "∇g", color=C["g"], fontsize=11, weight="bold")
    ax.set_title("at the optimum ∇f = λ·∇g  — the Lagrange multiplier λ\n"
                 "(SVMs & RLHF's KL-constraint live here)", fontsize=9)
    ax.set_aspect("equal"); ax.legend(fontsize=8, loc="lower left")
    return save(fig, "24_lagrange.png")


def fig_search_strategies():
    """Grid vs random vs Bayesian optimization of an expensive function."""
    rng = np.random.default_rng(12)
    xs = np.linspace(0, 10, 500)
    f = lambda t: (0.7 * t + 2.2 * np.sin(1.35 * t + 0.6)
                   + 1.4 * np.cos(2.4 * t)) + 4
    y = f(xs)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), sharey=True)
    # grid
    g = np.linspace(0.4, 9.6, 9)
    axes[0].plot(xs, y, color="gray", lw=1)
    axes[0].plot(g, f(g), "o", color=C["b"], ms=7)
    axes[0].set_title("grid: 9 evals, wastes budget in flat regions", fontsize=9)
    # random
    r = np.sort(rng.uniform(0.3, 9.7, 9))
    axes[1].plot(xs, y, color="gray", lw=1)
    axes[1].plot(r, f(r), "o", color=C["o"], ms=7)
    axes[1].set_title("random: covers space, no learning", fontsize=9)
    # 'bayesian': sequential — explore, then exploit around the incumbent
    picks = [1.2, 7.4, 4.6, 5.3, 4.9, 5.6, 5.0, 4.8, 5.15]
    axes[2].plot(xs, y, color="gray", lw=1)
    # crude surrogate: RBF through the picked points
    W = np.exp(-((xs[:, None] - np.array(picks)[None, :]) ** 2) / 1.4)
    surr = (W @ f(np.array(picks))) / np.maximum(W.sum(1), 1e-9)
    axes[2].plot(xs, surr, color=C["p"], lw=1.4, ls="--", label="surrogate")
    cols = [plt.cm.plasma(i / len(picks)) for i in range(len(picks))]
    for i, p in enumerate(picks):
        axes[2].plot(p, f(p), "o", color=cols[i], ms=7)
    axes[2].plot(picks[np.argmax(f(np.array(picks)))],
                 max(f(np.array(picks))), "*", color=C["g"], ms=15)
    axes[2].set_title("Bayesian: model the objective, spend evals where "
                      "promise is", fontsize=9)
    axes[2].legend(fontsize=7)
    fig.suptitle("Hyperparameter search: grid → random → Bayesian (05)",
                 fontsize=10, weight="bold")
    fig.tight_layout()
    return save(fig, "25_grid_random_bayes.png")


# =========================================================================
# §6 Information theory
# =========================================================================
def fig_entropy():
    """Entropy = uncertainty (max 1 bit at p=0.5); surprise = −log p."""
    p = np.linspace(0.001, 0.999, 400)
    H = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    axes[0].plot(p, H, color=C["b"], lw=2.4)
    axes[0].plot(0.5, 1.0, "o", color=C["r"], ms=7)
    axes[0].annotate("fair coin: 1 bit", (0.5, 1.0), (0.62, 0.72),
                     fontsize=8, color=C["r"],
                     arrowprops=dict(arrowstyle="->", color=C["r"]))
    axes[0].plot(0.9, np.interp(0.9, p, H), "o", color=C["o"], ms=6)
    axes[0].annotate(f"biased: {np.interp(0.9, p, H):.2f} bits", (0.9, np.interp(0.9, p, H)),
                     (0.55, 0.28), fontsize=8, color=C["o"],
                     arrowprops=dict(arrowstyle="->", color=C["o"]))
    axes[0].set_xlabel("P(heads) = p"); axes[0].set_ylabel("H(p)  [bits]")
    axes[0].set_title("entropy of a coin: H(p) = −Σ p·log p")
    ev = [0.5, 0.1, 0.01]
    surprise = [-np.log2(e) for e in ev]
    axes[1].barh([f"P={e}" for e in ev], surprise,
                 color=[C["b"], C["o"], C["r"]])
    for i, s in enumerate(surprise):
        axes[1].text(s + 0.1, i, f"{s:.2f} bits", va="center", fontsize=9)
    axes[1].set_xlabel("surprise = −log₂ P  [bits]")
    axes[1].set_title("rarer event ⇒ more information")
    fig.tight_layout()
    return save(fig, "26_entropy.png")


def fig_cross_entropy():
    """Cross-entropy is THE loss: confident-right ≈ 0, confident-wrong → ∞."""
    q = np.linspace(0.0005, 1.0, 400)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    axes[0].plot(q, -np.log2(q), color=C["b"], lw=2.4)
    axes[0].axvline(0.99, color=C["g"], ls="--")
    axes[0].axvline(0.01, color=C["r"], ls="--")
    axes[0].plot(0.99, -np.log2(0.99), "o", color=C["g"], ms=7)
    axes[0].plot(0.01, -np.log2(0.01), "o", color=C["r"], ms=7)
    axes[0].text(0.42, 0.9, "confident-right\n−log(0.99) ≈ 0.015",
                 color=C["g"], fontsize=8)
    axes[0].text(0.26, 6.0, "confident-WRONG\n−log(0.01) ≈ 6.64  ← punished",
                 color=C["r"], fontsize=8)
    axes[0].set_xlabel("q = model prob of the TRUE class")
    axes[0].set_ylabel("loss = −log₂ q  [bits]")
    axes[0].set_title("why cross-entropy punishes confident mistakes")
    bars = [("confident-right\nq=.99", -np.log2(0.99), C["g"]),
            ("uniform\nq=.5", -np.log2(0.5), C["b"]),
            ("confident-wrong\nq=.001", -np.log2(0.001), C["r"])]
    axes[1].bar([b[0] for b in bars], [b[1] for b in bars],
                color=[b[2] for b in bars])
    for i, b in enumerate(bars):
        axes[1].text(i, b[1] + 0.2, f"{b[1]:.2f}", ha="center",
                     weight="bold")
    axes[1].set_ylabel("cross-entropy [bits]")
    axes[1].set_title("exercise #4: feel the loss explode")
    fig.tight_layout()
    return save(fig, "27_cross_entropy.png")


def fig_kl_divergence():
    """KL(P‖Q): asymmetric 'extra bits' — distillation & RLHF penalty."""
    P = np.array([0.45, 0.35, 0.15, 0.05])
    Q = np.array([0.20, 0.30, 0.30, 0.20])
    kl_pq = float((P * np.log2(P / Q)).sum())
    kl_qp = float((Q * np.log2(Q / P)).sum())
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    xs = np.arange(4)
    axes[0].bar(xs - 0.19, P, width=0.38, color=C["b"], label="P (true)")
    axes[0].bar(xs + 0.19, Q, width=0.38, color=C["r"], label="Q (model)")
    axes[0].set_xticks(xs); axes[0].set_xticks(xs)
    axes[0].set_xticklabels(["c1", "c2", "c3", "c4"])
    axes[0].legend(fontsize=8)
    axes[0].set_title("two distributions over the same 4 outcomes")
    bars = axes[1].bar([f"KL(P‖Q)\n={kl_pq:.3f}", f"KL(Q‖P)\n={kl_qp:.3f}"],
                       [kl_pq, kl_qp], color=[C["b"], C["r"]])
    axes[1].axhline(0, color="k", lw=1)
    axes[1].set_ylabel("bits")
    axes[1].set_title("asymmetric: KL(P‖Q) ≠ KL(Q‖P)\n"
                      "both ≥ 0, = 0 only if P=Q", fontsize=9)
    fig.suptitle("KL = extra bits spent coding truth with the model — "
                 "RLHF adds β·KL(new‖old) to stay near the base model",
                 fontsize=10, weight="bold")
    return save(fig, "28_kl_divergence.png")


def fig_mutual_information():
    """Mutual information I(X;Y): how much knowing X tells you about Y."""
    rng = np.random.default_rng(13)

    def discretize(z1, z2, bins=8):
        h, _, _ = np.histogram2d(z1, z2, bins=bins)
        return h / h.sum()

    def mi(joint):
        px, py = joint.sum(1), joint.sum(0)
        m = 0.0
        for i in range(joint.shape[0]):
            for j in range(joint.shape[1]):
                if joint[i, j] > 0:
                    m += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
        return m

    z = rng.normal(size=800)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for ax, (t1, t2, c) in zip(axes, [
            ("Y ≈ X  →  high I(X;Y)", z + 0.3 * rng.normal(size=800), C["r"]),
            ("Y independent of X  →  I≈0", rng.normal(size=800), C["b"])]):
        J = discretize(z, t2)
        ax.imshow(J.T, origin="lower", cmap="Reds", vmin=0)
        ax.grid(False)
        ax.set_xlabel("X (binned)"); ax.set_ylabel("Y (binned)")
        ax.set_title(f"{t1}\nI = {mi(J):.3f} bits", fontsize=9)
    fig.suptitle("I(X;Y) = KL(P(X,Y) ‖ P(X)P(Y)) — feature selection keeps "
                 "high-MI inputs", fontsize=10, weight="bold")
    return save(fig, "29_mutual_information.png")


def fig_probability_tree():
    """Bayes as a probability tree: branch, count, normalize the leaves."""
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    nodes = {
        "root": (0.0, 0.0, "1000\npeople", "white"),
        "sick": (2.2, 1.3, "SICK  10\n(1%)", "#fee2e2"),
        "well": (2.2, -1.3, "HEALTHY  990\n(99%)", "#dcfce7"),
        "sp": (4.6, 2.1, "+ test\n9  (90%)", "#fecaca"),
        "sn": (4.6, 0.5, "− test\n1", "white"),
        "wp": (4.6, -0.5, "+ test\n49.5  (5%)", "#ffedd5"),
        "wn": (4.6, -2.1, "− test\n940.5", "white"),
    }
    edges = [("root", "sick", "1%"), ("root", "well", "99%"),
             ("sick", "sp", "90%"), ("sick", "sn", "10%"),
             ("well", "wp", "5%"), ("well", "wn", "95%")]
    for a, b, lab in edges:
        xa, ya = nodes[a][0], nodes[a][1]
        xb, yb = nodes[b][0], nodes[b][1]
        ax.plot([xa + 0.75, xb - 0.75], [ya, yb], color="gray", lw=1.5,
                zorder=1)
        ax.text((xa + xb) / 2, (ya + yb) / 2 + 0.13, lab, fontsize=8,
                color="gray", ha="center")
    for x, y, txt, fc in nodes.values():
        ax.add_patch(plt.Rectangle((x - 0.72, y - 0.34), 1.44, 0.68, fc=fc,
                                   ec="black", lw=1.2, zorder=2))
        ax.text(x, y, txt, ha="center", va="center", fontsize=8.5, zorder=3)
    ax.text(7.2, -0.9,
            "posterior\nP(sick|+) = 9 / (9 + 49.5)\n= 9 / 58.5 = 15.4%",
            fontsize=10, weight="bold", ha="center",
            bbox=dict(fc="#fef9c3", ec="#f59e0b"))
    ax.set_xlim(-1, 9.6); ax.set_ylim(-3.1, 3.2); ax.axis("off")
    ax.set_title("Probability tree: multiply along branches, "
                 "normalize the leaves you care about")
    return save(fig, "30_probability_tree.png")


def fig_learning_loop():
    """The training loop: forward → loss → backward → update, repeat."""
    fig, ax = plt.subplots(figsize=(8.8, 3.0))
    steps = [
        ("1. forward\nŷ = f_W(x)", "#dbeafe", "#2563eb"),
        ("2. loss\nL(ŷ, y)", "#fee2e2", "#dc2626"),
        ("3. backward\n∇W = ∂L/∂W", "#fef3c7", "#f59e0b"),
        ("4. update\nW ← W − η·∇W", "#dcfce7", "#16a34a"),
        ("5. repeat\nnext batch", "#f3e8ff", "#7c3aed"),
    ]
    w = 1.55
    for i, (txt, fc, ec) in enumerate(steps):
        x = 0.4 + i * (w + 0.55)
        ax.add_patch(plt.Rectangle((x, 1.0), w, 0.95, fc=fc, ec=ec, lw=2))
        ax.text(x + w / 2, 1.475, txt, ha="center", va="center",
                fontsize=9, weight="bold")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + 0.5, 1.475), xytext=(x + w + 0.03,
                        1.475), arrowprops=dict(arrowstyle="-|>",
                        color="black", lw=1.6))
    x0 = 0.4 + 4 * (w + 0.55) + w / 2
    x1 = 0.4 + w / 2
    ax.annotate("", xy=(x1, 0.85), xytext=(x0, 0.85),
                arrowprops=dict(arrowstyle="-|>", color="#7c3aed", lw=2,
                                connectionstyle="arc3,rad=0.25"))
    ax.text((x0 + x1) / 2, 0.42,
            "next epoch — calculus (step 3) meets optimization (step 4)",
            ha="center", fontsize=9, color="#7c3aed", style="italic")
    ax.set_xlim(0, x0 + 1.6); ax.set_ylim(0.2, 2.4); ax.axis("off")
    ax.set_title("One training step = the whole syllabus in a loop")
    return save(fig, "31_learning_loop.png")


# =========================================================================
FIGURES = [
    fig_vectors_dot, fig_matmul_shapes, fig_eigen_pca, fig_svd_lowrank,
    fig_derivative, fig_gradient_field, fig_chain_graph, fig_partials,
    fig_hessian,
    fig_bayes_test, fig_distributions, fig_expectation_variance,
    fig_joint_marginal, fig_likelihood, fig_monte_carlo,
    fig_estimator_biasvar, fig_ci_pvalue, fig_correlation_causation,
    fig_mle_map, fig_covariance_correlation,
    fig_convex_nonconvex, fig_learning_rate, fig_batch_vs_sgd,
    fig_lagrange, fig_search_strategies,
    fig_entropy, fig_cross_entropy, fig_kl_divergence,
    fig_mutual_information,
    fig_probability_tree, fig_learning_loop,
]


def main() -> int:
    print(f"writing figures → {OUT}")
    paths = [fn() for fn in FIGURES]
    missing = [p.name for p in paths if not p.exists() or p.stat().st_size < 1000]
    assert not missing, f"missing/empty figures: {missing}"
    print(f"\nOK — {len(paths)} figures generated, "
          f"{sum(p.stat().st_size for p in paths) // 1024} KB total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())









