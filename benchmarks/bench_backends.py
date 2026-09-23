#!/usr/bin/env python3
"""Benchmark NumPy vs PyTorch across the tutorial's workloads.

Usage:
    python3 benchmarks/bench_backends.py            # CPU (default)
    python3 benchmarks/bench_backends.py --device mps
    python3 benchmarks/bench_backends.py --json     # machine-readable report

Answers one question with numbers: *should this workload run in torch?*
The decision is the amortization rule from ai_core.accelerators —
torch_time = numpy_time / speedup + IMPORT_COST, so a framework only pays
off once the workload exceeds roughly 1.4-1.6s of NumPy compute.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from ai_core.accelerators import (  # noqa: E402
    HAS_TORCH, IMPORT_COST, backend_info, benchmark_backends,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    if not HAS_TORCH:
        print("torch is not installed — install it to run this benchmark:\n"
              "  python3 -m pip install torch")
        return 1

    if not args.json:
        info = backend_info(args.device)
        print("=" * 72)
        print("Backend benchmark — NumPy vs PyTorch")
        print("=" * 72)
        print(f"  torch {info['torch_version']} | device={args.device} | "
              f"mps={info['mps_available']} | import_cost={IMPORT_COST:.2f}s")
        print()

    report = benchmark_backends(device=args.device, quiet=args.json)

    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print("\n" + "-" * 72)
    print("Interpretation")
    print("-" * 72)
    for r in report["results"]:
        if r["speedup"]:
            verdict = ("worth switching to torch" if r["worth_it"]
                       else "keep NumPy (workload too small)")
            print(f"  {r['workload']:<32} {r['speedup']:>6.1f}x  -> {verdict}")
    print(f"\n  Break-even is ~{IMPORT_COST * 5 / 4:.2f}s of NumPy work at 5x "
          f"speedup (import amortization).")
    print("  Tutorial-scale kernels sit far below it, so the examples stay in "
          "NumPy by design.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
