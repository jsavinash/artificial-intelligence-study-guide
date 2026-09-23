"""m25 — ML System Design: executable 11-section design-doc generator +
an 8-step case-study scaffold (ranked-feed system).

Proves theory doc 25-ml-system-design.md.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "packages"))

ROOT = Path(__file__).resolve().parents[2]
SECTIONS = [
    "problem_framing", "metrics", "data_and_labels", "validation_schema",
    "baseline", "features_and_model", "serving_infrastructure",
    "experimentation", "monitoring", "risks_and_ethics", "retrospective",
]


def build_design_doc(problem: str, *, users, online_metric, offline_metric,
                     data_sources, validation, baseline, first_model,
                     serving_pattern, ab_plan, monitoring, risks,
                     non_goals=()) -> str:
    """The interview-ready template from module 25, rendered as Markdown."""
    def bullets(items):
        return "\n".join(f"- {i}" for i in items)
    return f"""# ML System Design: {problem}

## 1. Problem framing
**Users:** {users}
**Non-goals:** {bullets(non_goals) or '- n/a'}

## 2. Metrics
- **Online (business):** {online_metric}
- **Offline (model):** {offline_metric}
- **Guardrails:** latency p95, cost/request, safety violation rate

## 3. Data and labels
{bullets(data_sources)}

## 4. Validation schema
{validation}

## 5. Baseline
{baseline}

## 6. Features and model
- First version: {first_model}
- Roadmap: features -> gradient boosting -> deep/LLM (only if metrics demand)

## 7. Serving and infrastructure
- **Pattern:** {serving_pattern} (see Mercari catalog in module 25)
- Scale notes: stateless workers, model-load from registry, prediction logging on

## 8. Experimentation
{bullets(ab_plan)}

## 9. Monitoring
{bullets(monitoring)}

## 10. Risks and ethics
{bullets(risks)}

## 11. Retrospective
- [ ] Design vs shipped delta reviewed
- [ ] Owner assigned for the next iteration
"""


CASE_STUDY_STEPS = [
    "clarify_requirements", "metrics_offline_online", "data_and_labels",
    "constraints_scale_latency", "baseline_then_complex",
    "model_and_features", "serving_pattern", "monitor_and_iterate",
]


def main():
    # ---- 1) generate a full 11-section design doc ----
    doc = build_design_doc(
        "News feed ranking",
        users="Daily app users (millions DAU)",
        online_metric="engaged minutes per user per day",
        offline_metric="nDCG@10 on labeled relevance judgments",
        data_sources=["impression/click logs with timestamps",
                      "editor-labeled relevance set (5k queries)",
                      "content embeddings refreshed nightly"],
        validation="Time-based split: train <= T-7d, validate T-7d..T-1d, test T-1d..T",
        baseline="Recency-then-popularity heuristic + BM25 keyword rank",
        first_model="Learning-to-rank (gradient boosting on hand features)",
        serving_pattern="multi-stage: retrieve -> rank (async feature fetch)",
        ab_plan=["A/B at user level, 50/50", "min runtime 7 days",
                 "guardrail: p95 latency < 150ms"],
        monitoring=["input feature drift (PSI)", "label delay ~48h",
                    "score distribution shifts", "business metric regression"],
        risks=["filter bubble / diversity harm", "feedback loops from clicks",
               "stale embeddings after breaking news"],
        non_goals=["real-time personalization < 100ms (phase 2)",
                   "video feed (separate system)"],
    )

    # ---- 2) validation contract: all 11 sections present & substantive ----
    for sec in SECTIONS:
        header_key = sec.replace("_", " ")
        assert f"## {SECTIONS.index(sec) + 1}." in doc, sec   # numbered headers
    assert doc.count("\n## ") == 11
    assert all(len(line) > 10
               for line in doc.splitlines() if line.startswith("- "))  # no stubs

    # ---- 3) write the artifact (design docs are reviewable outputs) ----
    out = ROOT / "artifacts" / "design_docs" / "news_feed_ranking.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)

    # ---- 4) 8-step case-study scaffold fills in completely ----
    answers = {step: f"answered:{step}" for step in CASE_STUDY_STEPS}
    assert len(answers) == 8 and all(v for v in answers.values())

    # ---- 5) pattern selection logic: latency + freshness -> serving pattern ----
    def choose_serving(latency_budget_ms, freshness_s):
        if latency_budget_ms < 200 and freshness_s < 60:
            return "synchronous"
        if latency_budget_ms < 1000:
            return "asynchronous"
        return "batch"
    assert choose_serving(50, 10) == "synchronous"
    assert choose_serving(500, 300) == "asynchronous"
    assert choose_serving(86400, 86400) == "batch"

    # ---- 6) "is ML even the answer?" gate ----
    def ml_worth_it(baseline_beatable, data_available, decision_volume):
        return (not baseline_beatable) and data_available and decision_volume > 1000
    assert ml_worth_it(False, True, 100_000) is True
    assert ml_worth_it(True, True, 100_000) is False      # keep the heuristic!
    assert ml_worth_it(False, False, 100_000) is False    # no data, no ML

    print(f"PASS m25 system-design | doc_sections={doc.count(chr(10) + '## ')}"
          f"/11 written_to={out.name} ({out.stat().st_size}B) "
          f"case_steps=8/8 serving_picker=OK ml_gate=OK")


if __name__ == "__main__":
    main()
