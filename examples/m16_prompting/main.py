"""m16 — Prompt engineering: prompt pattern library + golden-set eval harness.

Runs fully offline against MockLLM (auto-upgrades with an API key).
Proves theory doc 16-prompt-engineering.md.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core import get_llm  # noqa: E402

SYSTEM_BASE = "You are a precise assistant. Follow the user's format exactly."

PATTERNS = {
    "summarize": lambda doc: (
        f"Summarize the following in one sentence:\n\"\"\"\n{doc}\n\"\"\""),
    "classify": lambda text, labels: (
        f"Classify the text into [{', '.join(labels)}]: {text}"),
    "extract_json": lambda text: (
        "Extract fields as JSON with keys summary, emails, amount. "
        "Return ONLY JSON. Text: " + text),
    "few_shot": lambda text: (
        "Examples:\nText: 'love it' -> positive\nText: 'hate it' -> negative\n"
        f"Classify the text into [positive, negative]: {text}"),
    "chain_of_thought": lambda q: (
        "Think step by step. List givens, plan, compute, then give the final answer. "
        f"Question: {q}"),
}

# Golden set: (pattern, input, checker) — the regression suite for prompt changes
GOLDEN = [
    ("summarize", "Quantum computing uses qubits that exist in superposition to "
     "solve certain problems faster than classical computers.",
     lambda out: isinstance(out, str) and len(out.split()) <= 60 and out.strip()),
    ("classify", ("The server crashed and data was lost", ["bug", "feature_request", "question"]),
     lambda out: out.strip() == "bug"),
    ("extract_json", "Contact ada@corp.com for the $1,250 invoice. Overall it failed.",
     lambda out: _is_valid_json_with(out, emails=True, amount=1250.0)),
    ("few_shot", "absolutely wonderful", lambda out: out.strip() == "positive"),
    ("chain_of_thought", "A train travels 120 km in 2 hours. What is its speed?",
     lambda out: "60" in out),
]


def _is_valid_json_with(out, emails=False, amount=None):
    try:
        d = json.loads(out)
    except Exception:
        return False
    if emails and not d.get("emails"):
        return False
    if amount is not None and d.get("amount") != amount:
        return False
    return True


def run_pattern(llm, name, arg, labels=None):
    build = PATTERNS[name]
    user = build(arg, labels) if name == "classify" else build(arg)
    return llm.chat([{"role": "system", "content": SYSTEM_BASE},
                     {"role": "user", "content": user}],
                    json_mode=(name == "extract_json"))


def main():
    llm = get_llm()
    results = []
    for pattern, arg, check in GOLDEN:
        labels = arg[1] if pattern == "classify" else None
        text_arg = arg[0] if pattern == "classify" else arg
        out = run_pattern(llm, pattern, text_arg, labels)
        ok = bool(check(out))
        results.append((pattern, ok))
        assert ok, f"golden failed: {pattern} -> {out!r}"

    # versioning discipline: identical prompt => deterministic result (temp=0)
    a = run_pattern(llm, "classify", "crash bug here", ["bug", "feature_request"])
    b = run_pattern(llm, "classify", "crash bug here", ["bug", "feature_request"])
    assert a == b

    # structured output is parseable (the production contract)
    js = run_pattern(llm, "extract_json", "Pay bob@x.io the $10 fee. It failed.")
    assert isinstance(json.loads(js), dict)

    passed = sum(ok for _, ok in results)
    detail = " ".join(f"{n}={'OK' if ok else 'FAIL'}" for n, ok in results)
    print(f"PASS m16 prompt-engineering | golden {passed}/{len(results)} ({detail}) "
          f"deterministic=True provider={llm.name}")


if __name__ == "__main__":
    main()
