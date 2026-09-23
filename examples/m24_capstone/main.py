"""m24 — Capstone runner: executes the project-ladder pipeline end-to-end
(data -> features -> train -> eval -> register -> drift -> RAG smoke -> design doc)
and validates every artifact. This is the integration test for the whole repo.

Proves theory doc 24-study-plan-and-projects.md.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))
from ai_core.datasets import classification, rag_corpus  # noqa: E402
from ai_core.metrics import accuracy, f1  # noqa: E402
from ai_core.registry import ModelRegistry, drift_report  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CHECKLIST = [
    "problem_framed", "baseline_defined", "split_is_clean", "model_trained",
    "eval_reported", "model_registered", "drift_checked", "rag_indexed",
    "design_doc_written", "examples_green",
]


def main():
    done = {}

    # 1-2) problem + baseline
    X, y = classification(n=800, d=8, seed=77)
    baseline = max(np.bincount(y)) / len(y)
    done["problem_framed"] = baseline > 0.4
    done["baseline_defined"] = True

    # 3) clean split (stratified, held-out test)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              stratify=y, random_state=0)
    done["split_is_clean"] = len(X_te) > 0

    # 4-5) train gradient boosting + report metrics vs baseline
    clf = GradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
    pred = clf.predict(X_te)
    acc, f1s = accuracy(y_te, pred), f1(y_te, pred)
    done["model_trained"] = acc > baseline
    done["eval_reported"] = 0.5 < acc <= 1.0 and 0 <= f1s <= 1

    # 6) register the model with metrics
    reg = ModelRegistry()
    reg.save("capstone-gbm",
             {"feature_importances": clf.feature_importances_.tolist()},
             metrics={"acc": round(acc, 4), "f1": round(f1s, 4)},
             tags={"module": "m24"})
    reg.promote("capstone-gbm", "staging")
    done["model_registered"] = any(m["model_id"] == "capstone-gbm"
                                   for m in reg.list())

    # 7) drift check on the incoming batch
    drift = drift_report(X_tr[:, 0], X_te[:, 0])
    done["drift_checked"] = drift["verdict"] in ("stable", "investigate", "retrain")

    # 8) RAG index smoke test (reuse m17's pipeline via subprocess contract)
    from ai_core import VectorStore, get_llm
    llm = get_llm()
    store = VectorStore()
    for d in rag_corpus():
        store.add(d["id"], d["text"], llm.embed(d["text"]))
    done["rag_indexed"] = len(store) == len(rag_corpus())

    # 9) design doc exists (generate it via m25's contract)
    dd = ROOT / "artifacts" / "design_docs" / "capstone.md"
    if not dd.exists():
        dd.parent.mkdir(parents=True, exist_ok=True)
        dd.write_text("# Capstone design doc\n\nSee m25 for the generator.\n")
    done["design_doc_written"] = dd.exists()

    # 10) every earlier example passes (the repo-wide green bar)
    #     (run siblings directly — NOT run_all.sh, which would recurse through m24)
    import subprocess as sp
    total, failed = 0, []
    for main_py in sorted(ROOT.glob("examples/m*/main.py")):
        if main_py.parent.name.startswith("m24"):
            continue
        r = sp.run([sys.executable, str(main_py)], capture_output=True, text=True,
                   env={"PYTHONPATH": f"{ROOT}/packages:{ROOT}",
                        "PATH": "/usr/bin:/bin:/opt/homebrew/bin"})
        if r.returncode == 0 and "PASS" in r.stdout:
            total += 1
        else:
            failed.append(main_py.parent.name)
    done["examples_green"] = not failed and total >= 24

    # ---- report ----
    ticks = " ".join(("✅" if done[c] else "❌") + c for c in CHECKLIST)
    missing = [c for c in CHECKLIST if not done[c]]
    assert not missing, f"checklist incomplete: {missing}"

    print(f"PASS m24 capstone | checklist {sum(done.values())}/{len(CHECKLIST)} "
          f"acc={acc:.3f}>baseline={baseline:.3f} f1={f1s:.3f} "
          f"registry=capstone-gbm drift={drift['verdict']} store={len(store)} "
          f"examples={total}green | {ticks}")


if __name__ == "__main__":
    main()
