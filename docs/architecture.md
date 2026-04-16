# AutoQRA Architecture

This document describes how the modules in `autoqra/` map onto the three
phases of the paper.

```
┌─────────────────────────────────────────────────────────────────────┐
│                            autoqra.cli                          │
│                  python -m autoqra.cli ...                      │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
                      ┌─────────────────────┐
                      │  autoqra.AutoQRA│   (orchestrator)
                      └──────────┬───────────┘
                                 │
   ┌─────────────────────────────┼──────────────────────────┐
   │                             │                          │
   ▼                             ▼                          ▼
Phase I               Phase II                       Phase III
(implicit)            PhaseIIEvolution                PhaseIIIBO
warm_start_from_       NSGA-II + repair               GP (Matern-5/2)
importance             + multi-fidelity surrogate     + Expected
                                                       Improvement
```

## Phase mapping

| Paper phase | Module | Key class / function |
|-------------|--------|----------------------|
| **I. Importance-Guided Warm Start**   | `core.importance` + `search.operators` | `Importance`, `warm_start_from_importance` |
| **II. Multi-Fidelity Evolutionary Search**  | `search.phase2_evolution`              | `PhaseIIEvolution`                          |
| **III. Trust-Region Bayesian Optimization**   | `search.phase3_bo`                     | `PhaseIIIBO`                                |
| Memory accounting (Eqs. 1–3)            | `core.memory`                          | `MemoryModel`                               |
| Repair (Eqs. 4–5)                       | `search.operators`                     | `repair_to_budget`                          |
| Multi-fidelity surrogate (Eq. 7)        | `surrogate.mlp`                        | `SurrogateMLPPromotion`                     |
| Pareto / NSGA-II                        | `core.pareto`                          | `non_dominated_sort_constrained`            |
| Atomic neighborhoods (Eq. 11)           | `search.neighbors`                     | `generate_k_nearest_atomic_neighbors`       |

## Evaluation backends

`PhaseIIEvolution` accepts `lf_eval_mode in {"proxy", "real_task"}`:

* `proxy` — uses `ProxyEvaluator` (importance-weighted resource coverage).
  No GPU or model weights required. Suitable for algorithm experimentation.
* `real_task` — dispatches the SFT → post-quant → lm-eval pipeline via
  `RealTaskEvaluator`, calling `autoqra.training.sft`,
  `autoqra.training.post_quant`, and `autoqra.experiments.eval_tasks`
  as subprocesses.

## Outputs

```
outdir/
├── phase2_pareto.json     # {pareto: [...], hv_hist: [...], config: {...}}
├── phase3_selected.json   # {q*, r*, phigh*, mem*, utility*, alpha}
└── phase3_history.json    # per-iteration BO trajectory
```
