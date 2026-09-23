"""Convert the percent-format notebook source into a real .ipynb.

Source : notebooks/00_math_foundations.py   (jupytext-style `# %%` cells)
Output : notebooks/00-mathematical-foundations.ipynb

Run:  python3 tools/build_math_notebook.py
Then: jupyter execute --inplace notebooks/00-mathematical-foundations.ipynb
      (or open it in JupyterLab / VS Code and Run All)
"""
from __future__ import annotations

import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "00_math_foundations.py"
OUT = ROOT / "notebooks" / "00-mathematical-foundations.ipynb"


def parse_cells(text: str) -> list[tuple[str, str]]:
    """Split percent-format source into (cell_type, source) pairs."""
    cells: list[tuple[str, str]] = []
    cur_type: str | None = None
    buf: list[str] = []

    def flush():
        nonlocal buf, cur_type
        if cur_type is not None:
            src = "\n".join(buf).strip("\n")
            if src.strip():
                cells.append((cur_type, src + "\n"))
        buf = []

    for line in text.splitlines():
        if line.startswith("# %%"):
            flush()
            cur_type = "markdown" if "[markdown]" in line else "code"
        else:
            if cur_type is None:                 # preamble before first marker
                cur_type = "code"
            if cur_type == "markdown":
                # strip exactly one leading '# ' or '#'
                if line.startswith("# "):
                    line = line[2:]
                elif line.startswith("#"):
                    line = line[1:]
            buf.append(line)
    flush()
    return cells


def main() -> int:
    parsed = parse_cells(SRC.read_text())
    cells = []
    for i, (ctype, src) in enumerate(parsed):
        cell = nbformat.v4.new_markdown_cell(src) if ctype == "markdown" \
            else nbformat.v4.new_code_cell(src)
        cell["id"] = f"cell-{i:03d}"
        cells.append(cell)

    nb = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.14"},
            "title": "00 — Mathematical Foundations for AI",
        },
    )
    nbformat.validate(nb)
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")

    md = sum(1 for c in cells if c.cell_type == "markdown")
    code = len(cells) - md
    print(f"built {OUT.relative_to(ROOT)} — {len(cells)} cells "
          f"({md} markdown, {code} code), nbformat validated ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
