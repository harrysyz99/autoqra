"""
Top-level orchestrator that wires Phases I → II → III together.

Phase I is implicit (warm-start inside ``PhaseIIEvolution`` — driven by
the ``Importance`` object loaded from the sensitivity JSON).

Phase II runs the evolutionary search to approximate the Pareto front.

Phase III (optional) refines a single operating point on the front using
trust-region Bayesian optimization with a user-specified preference
weight ``α`` between performance and memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from autoqra.core.config import AutoQRAConfig, ConfigEncoding
from autoqra.core.importance import Importance
from autoqra.core.memory import MemoryModel
from autoqra.evaluation.proxy import ProxyEvaluator
from autoqra.search.phase2_evolution import PhaseIIEvolution
from autoqra.search.phase3_bo import PhaseIIIBO


class AutoQRA:
    """End-to-end AutoQRA runner."""

    def __init__(
        self,
        cfg: AutoQRAConfig,
        importance_json: Union[str, Path],
        real_eval_params: Optional[Dict] = None,
        lowbit_value: Optional[int] = None,
        max_lowbit_fraction: float = 1.0,
        target_avg_bits: Optional[float] = None,
    ) -> None:
        self.cfg = cfg
        self.imp = Importance.from_json(importance_json, num_layers=cfg.num_layers)
        self.enc = ConfigEncoding(cfg.Q, cfg.R)
        self.mem = MemoryModel(cfg)
        self.real_eval_params = real_eval_params
        self.lowbit_value = lowbit_value
        self.max_lowbit_fraction = max_lowbit_fraction
        self.target_avg_bits = target_avg_bits

    def run(
        self,
        outdir: Path,
        phase2_kwargs: Dict,
        phase3_alpha: Optional[float] = None,
    ) -> Dict:
        """Execute Phases II and III, persisting intermediate artifacts."""
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        # -------- Phase II -------------------------------------------
        evo = PhaseIIEvolution(
            self.cfg,
            self.imp,
            real_eval_params=self.real_eval_params,
            lowbit_value=self.lowbit_value,
            max_lowbit_fraction=self.max_lowbit_fraction,
            target_avg_bits=self.target_avg_bits,
        )
        result_p2 = evo.run(
            **phase2_kwargs, real_eval_params=self.real_eval_params
        )
        pareto = result_p2["pareto"]

        with open(outdir / "phase2_pareto.json", "w") as f:
            json.dump(
                {
                    "pareto": pareto,
                    "hv_hist": result_p2["hv_hist"],
                    "config": result_p2["config"],
                },
                f,
                indent=2,
            )

        # -------- Phase III -----------------------------------------
        best = None
        history: list = []
        if phase3_alpha is not None and len(pareto) >= 3:
            bo = PhaseIIIBO(
                enc=self.enc,
                importance=self.imp,
                mem_model=self.mem,
                budget_bytes=self.cfg.budget_bytes or float("inf"),
                I_q=self.imp.I_q,
                I_r=self.imp.I_r,
                k_neighbors=5,
                epsilon_ei=1e-4,
                max_iterations=3,
            )
            bo.fit(pareto, alpha=phase3_alpha)

            incumbent = max(
                pareto,
                key=lambda p: bo.scalarize(
                    p["phigh"], p["mem"], phase3_alpha,
                    bo.perf_norm, bo.mem_norm,
                ),
            )
            pareto_cfgs = [(p["q"], p["r"]) for p in pareto]

            for t in range(bo.max_iterations):
                q_new, r_new, max_ei, num_cand = bo.propose_multi_start(
                    pareto_cfgs, k_neighbors=bo.k_neighbors
                )
                if bo.has_converged(max_ei):
                    break

                proxy = ProxyEvaluator(self.cfg, self.imp, self.mem, self.enc)
                plow, M = proxy.evaluate(q_new, r_new, self.cfg.budget_bytes)
                phigh = float(plow + np.random.normal(scale=0.02))

                improved = bo.update(q_new, r_new, phigh, M)
                if improved:
                    incumbent = {
                        "q": list(q_new),
                        "r": list(r_new),
                        "phigh": phigh,
                        "mem": M,
                    }
                    pareto_cfgs.append((list(q_new), list(r_new)))
                history.append(
                    {
                        "iteration": t + 1,
                        "q": list(q_new),
                        "r": list(r_new),
                        "phigh": phigh,
                        "mem": M,
                        "max_ei": max_ei,
                        "num_candidates": num_cand,
                        "improved": improved,
                    }
                )

            best = {
                "q": incumbent["q"],
                "r": incumbent["r"],
                "phigh": incumbent["phigh"],
                "mem": incumbent["mem"],
                "utility": bo.best_y,
                "alpha": phase3_alpha,
            }
            with open(outdir / "phase3_selected.json", "w") as f:
                json.dump(best, f, indent=2)
            with open(outdir / "phase3_history.json", "w") as f:
                json.dump(history, f, indent=2)

        return {"phase2": result_p2, "phase3_best": best, "phase3_history": history}
