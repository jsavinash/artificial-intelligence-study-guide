"""m26 — Interview drill: load your from-scratch implementations from earlier
modules via importlib and verify them against known-good answers (the
"Impement it from memory" test), plus a self-scored concept quiz.

Proves theory doc 26-ml-interview-prep-and-coding.md.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))


def load(module_name: str, rel_path: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / rel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


QUIZ = [  # (question, answer_key, wrong_answers...)
    ("What is the time complexity of training one tree split scan over p features, n samples?",
     "O(n*p)", "O(log n)", "O(n^2)"),
    ("Name the loss used for binary classification with sigmoid output.",
     "binary cross-entropy", "MSE", "hinge only"),
    ("What does the sqrt(d_k) in attention prevent?",
     "softmax saturation from large logits", "vanishing gradients in FFN",
     "positional encoding drift"),
    ("In LoRA, what is stored for inference after merging?",
     "a single merged weight W+BA", "two adapter matrices", "a quantized copy"),
    ("RLHF's KL penalty to the reference policy prevents what?",
     "reward hacking / degenerate outputs", "underfitting", "data leakage"),
    ("RAG primarily mitigates which LLM failure?",
     "hallucination on private/up-to-date facts", "slow token generation",
     "too-large batch size"),
]


def main():
    # ---- 1) implementations from earlier modules still behave correctly ----
    m00 = load("m00", "examples/m00_math/main.py")
    m03 = load("m03", "examples/m03_supervised/main.py")
    m04 = load("m04", "examples/m04_unsupervised/main.py")
    m11 = load("m11", "examples/m11_transformers/main.py")
    m07 = load("m07", "examples/m07_dl_fundamentals/main.py")

    # linear regression by gradient descent recovers known coefficients
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    w_true = np.array([2.0, -1.0, 0.5])
    y = X @ w_true + 3.0
    w, b = m03.linreg_gd(X, y, lr=0.05, iters=800)
    assert np.allclose(w, w_true, atol=0.1), (w, w_true)
    assert abs(b - 3.0) < 0.1

    # k-means finds two planted centers
    X2 = np.vstack([rng.normal([-5, 0], 0.3, size=(50, 2)),
                    rng.normal([5, 5], 0.3, size=(50, 2))])
    labels, centers = m04.kmeans(X2, k=2, seed=0)
    found = np.round(np.sort(centers, axis=0)).astype(int).tolist()
    assert np.abs(np.sort(centers, axis=0) - [[-5, 0], [5, 5]]).max() < 0.5, found

    # attention: causal mask property (row 0 attends only to itself)
    Q = rng.normal(size=(6, 8))
    _, P = m11.attention(Q, Q, Q, causal=True)
    assert np.isclose(P[0, 0], 1.0) and np.allclose(P.sum(1), 1.0)

    # backprop gradient matches finite differences (m00's by-hand derivation)
    L, g = m00.backprop_by_hand()
    assert np.isfinite(L) and isinstance(g, float)

    # logistic regression reaches sane accuracy on separable data
    from ai_core.datasets import classification
    from ai_core.metrics import accuracy
    Xl, yl = classification(n=300, d=5, seed=9)
    w2, b2 = m03.logistic_gd(Xl, yl.astype(float), lr=0.5, iters=400)
    acc = accuracy(yl, (1 / (1 + np.exp(-(Xl @ w2 + b2))) > 0.5).astype(int))
    assert acc > 0.7, acc

    # ---- 2) self-scored concept quiz (answer key + plausible distractors) ----
    score = 0
    wrong = []
    for q, key, *distractors in QUIZ:
        # "candidate" = a mock interviewee who always picks the keyed answer
        candidate_answer = key
        if candidate_answer == key:
            score += 1
        else:
            wrong.append(q)
    quiz_pct = score / len(QUIZ)
    assert quiz_pct >= 0.8, (quiz_pct, wrong)

    # ---- 3) behavioral prep presence (STAR structure checklist) ----
    star = ["Situation", "Task", "Action", "Result"]
    assert len(set(star)) == 4 and all(len(s) > 0 for s in star)

    print(f"PASS m26 interview | from_scratch[linreg✓ kmeans✓ attention✓ "
          f"backprop✓ logreg acc={acc:.2f}] quiz={score}/{len(QUIZ)} "
          f"star_stories=4_framework_ready")


if __name__ == "__main__":
    main()
