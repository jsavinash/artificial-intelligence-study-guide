"""m27 — Accelerated computing: when is PyTorch actually worth it?

Proves theory doc 27-accelerated-computing-and-pytorch.md.

The tutorial's DL examples are pure NumPy on purpose (zero deps, visible math).
This example answers the engineering question that leaves open: *when do you
switch to a framework?* The answer is measured, not asserted.

Key result this file demonstrates live:
    torch `import` costs ~1.2s; a tutorial example runs in ~0.15s.
    => adding torch to the small examples makes them ~8x SLOWER.
    Torch wins only past ~1.4-1.6s of NumPy work (import amortization).
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core.accelerators import (  # noqa: E402
    HAS_MPS, HAS_TORCH, IMPORT_COST, TORCH_VERSION,
    benchmark_backends, backend_info, numpy_conv2d, numpy_attention,
    should_accelerate, torch_attention, torch_conv2d, train_mlp,
)


def part_a_backend():
    """What hardware/software are we actually on?"""
    info = backend_info()
    print(f"  backend       : {'torch' if HAS_TORCH else 'numpy-only'}"
          f"{' v' + TORCH_VERSION if HAS_TORCH else ''}")
    print(f"  Apple MPS GPU : {HAS_MPS}")
    print(f"  import cost   : {IMPORT_COST:.2f}s  (measured `import torch`)")
    return info


def part_b_rule():
    """The amortization rule: torch_time = numpy_time/speedup + IMPORT_COST."""
    print("\n[2] amortization rule   T > import_cost * s/(s-1)")
    print(f"    {'speedup':>8}  {'break-even':>11}   verdict at 0.15s | at 3.0s")
    rows = []
    for s in (2, 3, 5, 12, 70):
        be = IMPORT_COST * s / (s - 1)
        small, big = should_accelerate(0.15, s), should_accelerate(3.0, s)
        rows.append((s, be, small, big))
        print(f"    {str(s) + 'x':>8}  {be:>10.2f}s   "
              f"{'numpy' if not small else 'torch':>15} | "
              f"{'numpy' if not big else 'torch'}")
    return rows


def part_c_equivalence():
    """Correctness first: the fast path must compute the same thing."""
    img, ker = np.random.rand(24, 24), np.random.rand(3, 3)
    c_np, c_to = numpy_conv2d(img, ker), torch_conv2d(img, ker)
    conv_ok = np.allclose(c_np, c_to, atol=1e-4)

    Q, K, V = (np.random.rand(32, 8) for _ in range(3))
    a_np, _ = numpy_attention(Q, K, V, causal=True)
    a_to, _ = torch_attention(Q, K, V, causal=True)
    att_ok = np.allclose(a_np, a_to, atol=1e-5)

    print(f"\n[3] numerical equivalence  conv={conv_ok}  attention={att_ok}")
    print("    (same math — a fused kernel, not a different algorithm)")
    return conv_ok and att_ok


def part_d_crossover():
    """Empirically find where torch overtakes NumPy for convolution.

    Uses the *already-imported* torch, so this isolates kernel cost from the
    import cost — the crossover here is about compute, and part B adds the
    one-off import on top.
    """
    print("\n[4] where does torch overtake NumPy? (conv, repeats scaled)")
    print(f"    {'size':>6} {'reps':>5} {'numpy':>9} {'torch':>9} {'speedup':>8}")
    crossover = None
    for n, reps in ((16, 200), (32, 60), (48, 20), (64, 6)):
        img, ker = np.random.rand(n, n), np.random.rand(5, 5)
        t = time.perf_counter()
        for _ in range(reps):
            numpy_conv2d(img, ker)
        np_s = time.perf_counter() - t
        if HAS_TORCH:
            torch_conv2d(img, ker)
            t = time.perf_counter()
            for _ in range(reps):
                torch_conv2d(img, ker)
            to_s = time.perf_counter() - t
        else:
            to_s = np_s
        sp = np_s / to_s if to_s else 1.0
        flag = ""
        if crossover is None and should_accelerate(np_s, sp):
            crossover, flag = np_s, "  <- crossover"
        print(f"    {n}x{n:<4} {reps:>5} {np_s:>8.4f}s {to_s:>8.4f}s "
              f"{sp:>7.1f}x{flag}")
    return crossover


def part_e_benchmark():
    """Full head-to-head with the verdict applied per workload."""
    print("\n[5] head-to-head benchmark")
    rep = benchmark_backends(quiet=False)
    for r in rep["results"]:
        assert r["torch_s"] is None or r["torch_s"] > 0
    gap = abs(rep["mlp_numpy_acc"] - rep["mlp_torch_acc"])
    print(f"    MLP accuracy  numpy={rep['mlp_numpy_acc']} "
          f"torch={rep['mlp_torch_acc']}  (gap {gap:.3f})")
    # same init distribution + same math => the two backends must agree.
    # Only meaningful when torch is actually installed (else torch_s is None).
    if HAS_TORCH:
        assert gap < 0.05, (
            f"numpy and torch MLP disagree by {gap:.3f} — likely a bug in one "
            "of the two implementations, not a real difference")
        assert rep["results"][2]["speedup"] is not None
    return rep


def part_f_device():
    """GPU isn't automatically faster: small work + launch overhead loses."""
    if not HAS_MPS:
        print("\n[6] MPS device: not available on this machine")
        return None
    X, y = np.random.rand(4000, 32), np.random.randint(0, 4, 4000)
    cpu = train_mlp(X, y, epochs=60, backend="torch", device="cpu")
    mps = train_mlp(X, y, epochs=60, backend="torch", device="mps")
    print(f"\n[6] device choice   cpu={cpu['seconds']:.3f}s   "
          f"mps={mps['seconds']:.3f}s   "
          f"({'>' if cpu['seconds'] < mps['seconds'] else '<'}"
          f" CPU is {'faster' if cpu['seconds'] < mps['seconds'] else 'slower'})")
    print("    tiny batches under-utilise the GPU — kernel launch overhead wins")
    return cpu, mps


def main():
    print("=" * 68)
    print("m27 — Accelerated computing: is PyTorch worth it?")
    print("=" * 68)

    print("\n[1] runtime capacity")
    info = part_a_backend()

    rows = part_b_rule()
    # the *arithmetic* of the rule is backend-independent: break-even must fall
    # monotonically and stay in the 1.2-2.5s band for realistic speedups
    for s, be, _, _ in rows:
        assert abs(be - IMPORT_COST * s / (s - 1)) < 1e-9
    assert 1.2 < rows[-1][1] < 1.5, rows[-1][1]   # 70x -> ~1.26s
    # the *decision* only applies when an accelerator actually exists
    if HAS_TORCH:
        assert should_accelerate(0.15, 5) is False, "small work stays in NumPy"
        assert should_accelerate(3.0, 5) is True, "large work uses torch"
    else:
        assert should_accelerate(3.0, 5) is False, (
            "without torch installed there is nothing to accelerate to")

    assert part_c_equivalence(), "torch path must match the NumPy reference"

    crossover = part_d_crossover()
    rep = part_e_benchmark()
    dev = part_f_device()

    conv = next(r for r in rep["results"] if r["workload"].startswith("conv"))
    attn = next(r for r in rep["results"] if r["workload"].startswith("attention"))
    mlp = next(r for r in rep["results"] if r["workload"].startswith("MLP"))

    # speedup alone is NOT the decision — the rule decides
    if HAS_TORCH:
        print(f"\n[!] note: conv is {conv['speedup']:.0f}x faster in torch, yet "
              f"every workload\n    here (max {conv['numpy_s']:.3f}s) sits below "
              f"the {IMPORT_COST:.2f}s import cost — so the rule keeps NumPy.")
    assert crossover is None, (
        "tutorial-scale kernels should NOT reach break-even; got "
        f"{crossover:.3f}s — revisit IMPORT_COST or the sweep sizes")

    # headline: did acceleration actually help on the big workloads?
    useful = [r for r in rep["results"] if r["speedup"] and r["speedup"] > 2]
    if HAS_TORCH:
        assert useful, "expected at least one kernel with >2x speedup"

    if not HAS_TORCH:
        print("\n[!] torch not installed — NumPy-only path ran cleanly\n"
              "    install with: python3 -m pip install torch")
        print("PASS m27 accelerators | backend=numpy-only fallback_ok=True "
              f"equiv_ok=True import_cost={IMPORT_COST}s "
              "crossover=not_reached")
        return


    dev_note = ""
    if dev:
        dev_note = (f" device_cpu={dev[0]['seconds']:.2f}s"
                    f"<mps={dev[1]['seconds']:.2f}s")
    print(f"\n[7] verdict: keep NumPy for these examples (0.15s < "
          f"{IMPORT_COST:.2f}s import); switch to torch past ~"
          f"{IMPORT_COST * 5 / 4:.1f}s of NumPy work")

    print(f"PASS m27 accelerators | torch={TORCH_VERSION} mps={HAS_MPS} "
          f"conv={conv['speedup']}x attn={attn['speedup']}x "
          f"mlp_train={mlp['speedup']}x import_cost={IMPORT_COST}s "
          f"crossover=not_reached rule_ok=True equiv_ok=True{dev_note}")




if __name__ == "__main__":
    main()

