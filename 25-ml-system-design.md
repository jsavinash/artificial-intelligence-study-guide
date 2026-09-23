# 25 — ML System Design & Case Studies

> Back to [index](README.md) · Read after [02](02-machine-learning-foundations.md), [05](05-model-evaluation-and-tuning.md), [21](21-mlops-and-production-ml.md) · Next: [26 Interviews](26-ml-interview-prep-and-coding.md)

The discipline of designing *production* ML systems — the gap between "model works in a notebook" and "system delivers value reliably." Synthesis of Chip Huyen's MLSD/DMLS, Mercari's pattern catalog, and the Manning MLSD book.

## 1. Before any model: the problem space

- **Problem space vs solution space:** articulate the need *before* proposing ML. Ask: **is ML even the answer?** Compare against heuristics, database lookups, rules engines, and human review — a simple baseline beats a complex model solving the wrong problem.
- ML as feature vs ML as product; cost of mistakes (FP vs FN in dollars); constraints: latency, budget, team, regulations.
- **The design doc is the central artifact:** communication tool + decision record + review target.

## 2. The design-document template (interview-ready skeleton)

1. **Problem framing** — users, scope, non-goals
2. **Metrics** — offline (model) vs online (business) + guardrail metrics
3. **Data** — sources, labeling plan, quantity/quality, privacy
4. **Validation schema** — splits per data type ([02](02-machine-learning-foundations.md), [05](05-model-evaluation-and-tuning.md))
5. **Baseline** — heuristics + simple model; expected gain vs complexity
6. **Features/model** — first version simple, roadmap to complex ([03](03-supervised-learning-algorithms.md))
7. **Serving/infrastructure** — pattern choice (below), scale, latency budget ([21](21-mlops-and-production-ml.md))
8. **Experimentation** — A/B plan, sample size, rollback
9. **Monitoring** — data/model/business metrics, drift, retrain triggers
10. **Risks & ethics** — failure modes, bias, misuse ([22](22-ai-safety-security-ethics.md))
11. **Retrospective** — what changed vs design; ownership

## 3. Pattern catalog (Mercari taxonomy)

**Serving patterns:** synchronous (low latency, online features) · asynchronous (queue, decoupled) · **batch** (offline scores) · prep-pred (separate prep & predict services) · microservice vertical/horizontal · **prediction/data cache** · **circuit-breaker** (fall back to heuristic when the model service fails) · multi-stage prediction (retrieve→rank, [17](17-retrieval-augmented-generation.md)).
*Antipatterns:* online bigsize model, all-in-one monolith.

**QA patterns:** **shadow testing** (score live traffic, compare) · online A/B ([21](21-mlops-and-production-ml.md)) · load testing. *Antipattern:* offline-only evaluation.

**Training patterns:** batch training · pipeline training · **parameter & architecture search (HPO**, [05](05-model-evaluation-and-tuning.md)).
*Antipatterns:* only-me (training lives on one laptop), training code inside the serving service, too-many-pipes.

**Operation patterns:** model-in-image vs **model-load** (registry-served weights) · **data & model versioning** · **prediction logging** (fuels future training + audits) · prediction monitoring · condition-based serving.
*Antipatterns:* no-logging, nobody-knows (no owner).

**Lifecycle patterns:** train-then-serve vs unified training-to-serving (CI/CT, [21](21-mlops-and-production-ml.md)).

## 4. Case-study method

Run any study (search ranking, recsys, ads CTR, fraud, moderation, dynamic pricing) in **8 steps:** clarify → metrics → data/labels → constraints → baseline → features+model → serving pattern → monitor & iterate. Practice aloud with the case-study repos; structure your answers with [26](26-ml-interview-prep-and-coding.md).

## 5. ML technical debt & data cascades (Sculley et al.)

- **Data dependencies** (fragile), **entanglement** (features inseparable), **shadow features**, **feedback loops** (model output becomes training input), **pipeline jungles**, **glue code**, configuration proliferation.
- **Silent failure modes:** exploding data problems, untested pipelines, monitoring that watches servers but not predictions.
- **Data cascades** (Sambasivan et al.): small upstream data-quality issues compound into downstream project failure — data quality *is* project quality.
- Mitigations: feature stores & contracts ([21](21-mlops-and-production-ml.md)), prediction logging, ownership, start-simple principles.

## Mastery Checklist

- [ ] Writes a full 11-section ML design doc in 45 minutes
- [ ] Chooses sync/async/batch/cache serving pattern from latency & scale constraints
- [ ] Names 4 antipatterns and the fix for each
- [ ] Runs the 8-step case-study method on a ranking or fraud system
- [ ] Explains feedback loops, entanglement, and prediction logging's dual role
- [ ] Argues when *not* to use ML for a proposed feature
