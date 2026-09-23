"""Verify the executed notebook: all cells ran, zero errors, key numbers present.

Run:  python3 tools/verify_math_notebook.py        (exit 0 = notebook is healthy)
"""
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "00-mathematical-foundations.ipynb"
nb = nbformat.read(NB, as_version=4)
code = [c for c in nb.cells if c.cell_type == "code"]
md = [c for c in nb.cells if c.cell_type == "markdown"]

errors = [(i, o.get("ename")) for i, c in enumerate(nb.cells)
          if c.cell_type == "code"
          for o in c.get("outputs", []) if o.output_type == "error"]
unrun = [i for i, c in enumerate(code) if c.execution_count is None]
images = sum(1 for c in code for o in c.get("outputs", [])
             if o.output_type == "display_data" and "image/png" in o.get("data", {}))
all_text = "\n".join(o.get("text", "") for c in code
                     for o in c.get("outputs", []) if o.output_type == "stream")

checks = [
    "[[58, 64], [139, 154]]",   # §1 matrix multiply
    "53.13",                    # §1 cosine angle
    "54",                       # §2 chain rule
    "15.4%",                    # §3 Bayes posterior
    "mean=5, var=4",            # §4 dataset
    "0.982",                    # §4 correlation
    "0.786",                    # §5 GD table
    "DIVERGES",                 # §5 learning rate
    "PARALLEL",                 # §5 Lagrange condition
    "9.966",                    # §6 cross-entropy
    "ASYMMETRIC",               # §6 KL
    "backprop in 10 lines",     # exercise 2
    "Exercise 5 ✓",
]
missing = [s for s in checks if s not in all_text]

print(f"cells   : {len(nb.cells)} total = {len(md)} markdown + {len(code)} code")
print(f"executed: {len(code) - len(unrun)}/{len(code)}   errors: {len(errors)}   "
      f"png outputs: {images}")
print(f"key numbers: {'ALL PRESENT' if not missing else 'MISSING ' + str(missing)}")

if errors or unrun or missing or images < 15:
    print("VERIFICATION FAILED", errors, unrun, missing, file=sys.stderr)
    sys.exit(1)
print("NOTEBOOK VERIFIED — every assert passed, all figures rendered ✓")
sys.exit(0)
