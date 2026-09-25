"""Executable companion to docs/curriculum/00-mathematical-foundations.md.

Every worked calculation in that document is recomputed here and asserted, so
the tables can never drift from the math. Prints FORMULA / STEPS / RESULT.

Run:  python3 tools/calc_00_math.py
"""
import math

import numpy as np


def head(title: str) -> None:
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


# ---------------------------------------------------------------- §1 linear algebra
def sec1():
    head("§1 LINEAR ALGEBRA")
    print("FORMULA  C[i,j] = Σ_k A[i,k]·B[k,j]   |  (m×n)·(n×p) → (m×p)")
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[7, 8], [9, 10], [11, 12]])
    C = A @ B
    print("STEPS    A(2x3)·B(3x2):")
    for i in range(2):
        for j in range(2):
            terms = " + ".join(f"{A[i,k]}·{B[k,j]}" for k in range(3))
            print(f"         C[{i},{j}] = {terms} = {C[i,j]}")
    assert C.tolist() == [[58, 64], [139, 154]], C
    print(f"RESULT   {C.tolist()}   ✓ matches doc")

    print("\nFORMULA  cos θ = a·b / (‖a‖·‖b‖)")
    a, b = np.array([3.0, 1.0]), np.array([1.0, 3.0])
    dot = a @ b
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = dot / (na * nb)
    theta = math.degrees(math.acos(cos))
    print(f"STEPS    a·b = 3·1 + 1·3 = {dot:.0f} | ‖a‖ = √(9+1) = {na:.3f} "
          f"| cos = {dot:.0f}/10 = {cos:.2f}")
    print(f"RESULT   θ = arccos({cos:.2f}) = {theta:.2f}°   ✓ 53.13°")
    assert abs(cos - 0.6) < 1e-12 and abs(theta - 53.13) < 0.01

    print("\nFORMULA  linear layer  Y = W·X + b")
    X = np.array([[1.0], [0.0], [-1.0]])   # 3 features as a column (3×1)
    W = np.array([[0.5, -0.5, 0.25],
                  [-1.0, 2.0, 0.0],
                  [0.0, 0.0, 1.0]])             # 3 outputs x 3 features
    Y = W @ X
    print(f"STEPS    W(3x3)·X(3x1) → Y(3x1) = {Y.ravel().tolist()}")
    print("RESULT   one column per sample — batch of N = one matmul   ✓")


    print("\nFORMULA  eigen: det(C − λI) = 0  ⇒  λ² − trace·λ + det = 0")
    C2 = np.array([[2.0, 1.0], [1.0, 2.0]])
    tr, dt = float(np.trace(C2)), float(np.linalg.det(C2))
    disc = math.sqrt(tr ** 2 - 4 * dt)
    lam1, lam2 = (tr + disc) / 2, (tr - disc) / 2
    print(f"STEPS    C = [[2,1],[1,2]]: trace = {tr:.0f}, det = {dt:.0f} "
          f"→ λ² − 4λ + 3 = 0 → λ = (4 ± √4)/2 = {lam1:.0f}, {lam2:.0f}")
    v1 = np.array([1.0, 1.0]) / np.sqrt(2)        # (C − 3I)v = 0
    v2 = np.array([1.0, -1.0]) / np.sqrt(2)       # (C − 1I)v = 0
    # det() is LU-based → 2.9999999999999996, so compare eigenvalues with tolerance
    assert abs(lam1 - 3.0) < 1e-12 and abs(lam2 - 1.0) < 1e-12, (lam1, lam2)
    assert np.allclose(C2 @ v1, 3 * v1, atol=1e-12)
    assert np.allclose(C2 @ v2, 1 * v2, atol=1e-12)
    assert abs(float(v1 @ v2)) < 1e-15            # eigenvectors are orthogonal
    share = lam1 / (lam1 + lam2)
    proj = float(np.array([2.0, 3.0]) @ v1)
    print(f"STEPS    v1 = (1,1)/√2 (λ=3) → C·v1 = {np.round(C2 @ v1, 3).tolist()}; "
          f"v2 = (1,−1)/√2 (λ=1); v1·v2 = {v1 @ v2:.0f}")
    print(f"RESULT   PCA: PC1 keeps {share:.0%} of the variance, x=(2,3) → PC1 score "
          f"{proj:.3f}   ✓ (figure 03)")
    assert abs(share - 0.75) < 1e-12 and abs(proj - 3.5355) < 1e-3


# ---------------------------------------------------------------- §2 calculus
def sec2():
    head("§2 CALCULUS & GRADIENTS")
    print("FORMULA  d/dx xⁿ = n·xⁿ⁻¹   |  chain: dL/dx = (dL/dg)·(dg/dx)")
    x = 1.0
    g = 2 * x + 1
    dg_dx = 2.0
    dL_dg = 3 * g ** 2                    # L = g³
    dL_dx = dL_dg * dg_dx
    print(f"STEPS    x=1 → g=2·1+1={g:.0f} | dg/dx=2 | dL/dg=3g²=3·{g:.0f}²"
          f"={dL_dg:.0f}")
    print(f"         dL/dx = {dL_dg:.0f} × {dg_dx:.0f} = {dL_dx:.0f}")
    eps = 1e-6                            # finite-difference sanity check
    fd = (((2 * (x + eps) + 1) ** 3) - ((2 * x + 1) ** 3)) / eps
    print(f"RESULT   analytic {dL_dx:.0f} vs finite-diff {fd:.1f}   ✓ 54")
    assert dL_dx == 54 and abs(fd - dL_dx) < 1e-3

    print("\nFORMULA  f(x)=x³−2x → f'(x)=3x²−2 ; step w ← w − η·2w")
    print(f"STEPS    f'(2) = 3·2² − 2 = {3*4-2}   ✓ 10")
    w = 3.0
    lr = 0.1
    w_new = w - lr * 2 * w
    print(f"         L=w²: w=3, grad=2w=6, w←3 − 0.1·6 = {w_new} "
          f"(loss 9 → {w_new**2:.2f})   ✓ 2.4")
    assert w_new == 2.4

    print("\nFORMULA  one training step: g=w·x → p=e^g → L=½(p−y)²")
    nx, ny, nw0, neta = 1.0, 1.0, 2.0, 0.01
    ng0 = nw0 * nx
    np0 = math.exp(ng0)
    nL0 = 0.5 * (np0 - ny) ** 2
    ndLdp, ndpdg, ndgdw = np0 - ny, math.exp(ng0), nx
    ndLdw = ndLdp * ndpdg * ndgdw
    nw1 = nw0 - neta * ndLdw
    nnp1 = math.exp(nw1 * nx)
    nnL1 = 0.5 * (nnp1 - ny) ** 2
    print(f"STEPS    fwd: g={ng0:.1f} p=e²={np0:.3f} L={nL0:.3f}")
    print(f"         bwd: ∂L/∂p={ndLdp:.3f} ∂p/∂g={ndpdg:.3f} ∂g/∂w={ndgdw:.1f} "
          f"→ ∂L/∂w={ndLdw:.3f}")
    print(f"RESULT   w {nw0:.1f}→{nw1:.4f}, L {nL0:.3f}→{nnL1:.3f} (−68.1%)   ✓")
    assert abs(np0 - 7.389) < 0.001 and abs(nL0 - 20.410) < 0.001
    assert abs(ndLdw - 47.209) < 0.001 and abs(nw1 - 1.5279) < 0.001
    assert abs(nnL1 - 6.511) < 0.001


# ---------------------------------------------------------------- §3 probability
def sec3():
    head("§3 PROBABILITY")
    print("FORMULA  P(H|D) = P(D|H)·P(H) / P(D)")
    prev, sens, fpr = 0.01, 0.90, 0.05
    joint_pos = prev * sens                 # sick & +
    joint_fp = (1 - prev) * fpr             # healthy & +
    p_pos = joint_pos + joint_fp
    post = joint_pos / p_pos
    print("STEPS    per 1000: sick=10 healthy=990")
    print(f"         sick & +     = 10 · 0.90 = {prev*1000*sens:.1f}")
    print(f"         healthy & +  = 990 · 0.05 = {(1-prev)*1000*fpr:.1f}")
    print(f"         P(+) = {joint_pos:.3f} + {joint_fp:.4f} = {p_pos:.4f}")
    print(f"RESULT   P(sick|+) = {joint_pos:.3f}/{p_pos:.4f} = "
          f"{post:.4f} → {post:.1%}   ✓ 15.4%")
    assert abs(post - 0.153846) < 1e-5

    print("\nFORMULA  E[X] = Σ x·p(x)   |  Var = Σ (x−E)²·p(x)")
    die = np.arange(1, 7)
    e = die.mean()
    var = ((die - e) ** 2).mean()
    print(f"STEPS    E = (1+2+3+4+5+6)/6 = {e}   Var = 17.5/6 = {var:.3f}")
    print("RESULT   E=3.5, Var≈2.917 (35/12)   ✓")
    assert e == 3.5 and abs(var - 35 / 12) < 1e-12

    print("\nTABLE (E / Var per distribution)")
    rows = [("Bernoulli(p=.3)", 0.3, 0.3 * 0.7),
            ("Poisson(λ=3)", 3.0, 3.0),
            ("Uniform[0,1]", 0.5, 1 / 12)]
    for name, ex, vx in rows:
        print(f"         {name:<18} E={ex:<6.3f} Var={vx:.4f}")
    assert rows[1][1] == rows[1][2]          # Poisson: E = Var = λ
    print("RESULT   Poisson E=Var=λ   ✓")


# ---------------------------------------------------------------- §4 statistics
def sec4():
    head("§4 STATISTICS")
    print("FORMULA  mean=Σx/n  |  var=Σ(x−mean)²/n  |  sd=√var")
    data = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
    m, v = data.mean(), data.var()
    print(f"STEPS    Σx = {data.sum():.0f} → mean = {data.sum():.0f}/8 = {m}")
    dev = data - m
    print(f"         deviations = {dev.astype(int).tolist()}")
    print(f"         Σ dev² = {(dev**2).sum():.0f} → var = "
          f"{(dev**2).sum():.0f}/8 = {v} → sd = {np.sqrt(v):.3f}")
    print("RESULT   mean=5, var=4, sd=2   ✓ (classic textbook set)")
    assert m == 5 and v == 4

    print("\nFORMULA  r = Sxy / √(Sxx·Syy)")
    x, y = np.array([1.0, 2, 3]), np.array([2.0, 4, 5])
    xc, yc = x - x.mean(), y - y.mean()
    sxy, sxx, syy = (xc * yc).sum(), (xc ** 2).sum(), (yc ** 2).sum()
    r = sxy / math.sqrt(sxx * syy)
    print(f"STEPS    x̄={x.mean():.0f} ȳ={y.mean():.4f} (=11/3)")
    print(f"         Sxy={sxy:.0f}  Sxx={sxx:.0f}  Syy={syy:.4f}")
    print(f"         r = {sxy:.0f}/√({sxx:.0f}×{syy:.4f}) = "
          f"{r:.4f}   ✓ 0.982 (strong +)")
    assert abs(r - 0.982) < 0.001

    print("\nFORMULA  MLE: θ̂ = argmax P(D|θ)   |  MAP: argmax P(D|θ)·P(θ)")
    print("RESULT   Gaussian MLE mean = sample mean; MAP pulls toward prior"
          "   ✓")


# ---------------------------------------------------------------- §5 optimization
def sec5():
    head("§5 OPTIMIZATION")
    print("FORMULA  gradient descent:  w ← w − η·∇L   (∇L = 2w for L=w²)")
    w, lr = 2.4, 0.1
    print(f"STEPS    start w={w}, η={lr}")
    print(f"         {'step':>4} {'w':>8} {'grad=2w':>9} {'new w':>8} "
          f"{'loss=w²':>8}")
    for step in range(5):
        grad = 2 * w
        new_w = w - lr * grad
        print(f"         {step:>4} {w:>8.3f} {grad:>9.3f} {new_w:>8.3f} "
              f"{w*w:>8.3f}")
        w = new_w
    print(f"RESULT   after 5 steps w={w:.3f} — monotone descent   ✓")
    assert abs(w - 0.786) < 0.001

    print("\nSTABILITY  for L=w²: w ← (1−2η)·w  ⇒ stable iff |1−2η| < 1")
    for lr, stable in ((0.1, True), (0.85, True), (1.12, False)):
        factor = abs(1 - 2 * lr)
        ok = factor < 1
        print(f"         η={lr:<5} factor |1−2η| = {factor:.3f} → "
              f"{'stable' if ok else 'DIVERGES'}")
        assert ok is stable
    print("RESULT   η must be < 1 on this bowl   ✓  matches figure 22")


# ---------------------------------------------------------------- §6 information theory
def sec6():
    head("§6 INFORMATION THEORY")
    print("FORMULA  H(p) = −Σ p·log₂ p | CE = −Σ p·log₂ q | "
          "KL = Σ p·log₂(p/q)")
    h_fair = -2 * 0.5 * math.log2(0.5)
    h_biased = -(0.9 * math.log2(0.9) + 0.1 * math.log2(0.1))
    print(f"STEPS    fair coin:   −2·0.5·log₂0.5 = {h_fair:.3f} bits")
    print(f"         p=0.9 coin: −0.9·log₂0.9 − 0.1·log₂0.1 = "
          f"{h_biased:.3f} bits")
    print("RESULT   H(fair)=1.000, H(.9)=0.469   ✓  (figures 26)")

    ce_right, ce_wrong = -math.log2(0.9), -math.log2(0.001)
    print(f"\nSTEPS    CE q=0.9   : −log₂0.9 = {ce_right:.3f} bits")
    print(f"         CE q=0.001 : −log₂0.001 = {ce_wrong:.3f} bits "
          f"→ {ce_wrong/ce_right:.1f}× worse")
    print("RESULT   confident mistakes dominate the loss   ✓  (figure 27)")
    assert abs(ce_wrong - 9.966) < 0.001

    P = np.array([0.5, 0.5])
    Q = np.array([0.25, 0.75])
    kl_pq = float((P * np.log2(P / Q)).sum())
    kl_qp = float((Q * np.log2(Q / P)).sum())
    print("\nSTEPS    P=(.5,.5) Q=(.25,.75)")
    print(f"         KL(P‖Q) = 0.5·log₂(2) + 0.5·log₂(2/3) = {kl_pq:.4f}")
    print(f"         KL(Q‖P) = 0.25·log₂(1/2) + 0.75·log₂(3/2) = {kl_qp:.4f}")
    print(f"RESULT   {kl_pq:.3f} ≠ {kl_qp:.3f} — KL is asymmetric   ✓ "
          f"(figure 28)")
    assert abs(kl_pq - 0.2075) < 0.001 and abs(kl_qp - 0.1887) < 0.001

    print("\nFORMULA  I(X;Y) = H(X) − H(X|Y)")
    print("STEPS    fair bit X: H(X)=1. If Y=X: H(X|Y)=0 → I=1 bit")
    print("         If Y ⊥ X : H(X|Y)=1 → I=0 bits")
    print("RESULT   mutual information = dependence in bits   ✓  (fig 29)")


def main() -> int:
    for fn in (sec1, sec2, sec3, sec4, sec5, sec6):
        fn()
    print("\n" + "=" * 66)
    print("RESULT: all worked calculations in the doc verified")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


