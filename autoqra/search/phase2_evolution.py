"""
Phase II: Multi-Fidelity Evolutionary Search.

Implements the constrained evolutionary strategy that approximates the
global Pareto front of (performance, memory). Key components:

  * Importance-warm-started population (Phase I).
  * k-nearest atomic neighbors as local-search offspring.
  * Deterministic ``repair_to_budget`` to enforce M(C) ≤ B_max.
  * Multi-fidelity surrogate promotion (Eq. 7) selects top-K offspring
    for expensive high-fidelity re-evaluation.
  * NSGA-II selection with constrained domination.
  * Hypervolume-based early stopping.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from autoqra.core.config import AutoQRAConfig, ConfigEncoding
from autoqra.core.importance import Importance
from autoqra.core.memory import MemoryModel
from autoqra.core.pareto import (
    crowding_distance,
    hypervolume_2d,
    non_dominated_sort_constrained,
)
from autoqra.evaluation.proxy import ProxyEvaluator
from autoqra.evaluation.real_task import RealTaskEvaluator
from autoqra.search.neighbors import generate_k_nearest_atomic_neighbors
from autoqra.search.operators import (
    jitter_configuration,
    repair_to_budget,
    warm_start_from_importance,
)
from autoqra.surrogate.mlp import SurrogateMLPPromotion
from autoqra.utils.numeric import set_seed


class PhaseIIEvolution:
    """Constrained evolutionary strategy for the (q, r) joint search space.

    Parameters
    ----------
    cfg : AutoQRAConfig
        Search-space configuration.
    importance : Importance
        Orthogonal sensitivity profiles.
    real_eval_params : dict | None
        Parameters for the real-task evaluator (used when
        ``lf_eval_mode='real_task'``).
    lowbit_value : int | None
        Explicit low-bit value whose fraction is capped.
    max_lowbit_fraction : float
        Maximum fraction of layers allowed at ``lowbit_value``.
    target_avg_bits : float | None
        Optional target average bit-width for the proxy evaluator.
    """

    def __init__(
        self,
        cfg: AutoQRAConfig,
        importance: Importance,
        real_eval_params: Optional[Dict] = None,
        lowbit_value: Optional[int] = None,
        max_lowbit_fraction: float = 1.0,
        target_avg_bits: Optional[float] = None,
    ) -> None:
        self.cfg = cfg
        self.importance = importance
        self.enc = ConfigEncoding(cfg.Q, cfg.R)
        self.mem = MemoryModel(cfg)
        self.proxy = ProxyEvaluator(
            cfg, importance, self.mem, self.enc, target_avg_bits=target_avg_bits
        )
        self.sur = SurrogateMLPPromotion(hidden_dims=(64, 32), patience=10)
        self.real_eval_params = real_eval_params
        self.real_eval: Optional[RealTaskEvaluator] = None
        self.lowbit_value = lowbit_value
        self.max_lowbit_fraction = max(0.0, min(1.0, max_lowbit_fraction))
        self.target_avg_bits = target_avg_bits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _enforce_lowbit_cap(self, q_vec: List[int]) -> List[int]:
        """Limit how many layers are allowed at the minimum bit-width."""
        if self.max_lowbit_fraction >= 1.0:
            return q_vec
        low_bit = (
            self.lowbit_value
            if self.lowbit_value is not None
            else min(self.enc.Q)
        )
        low_indices = [i for i, b in enumerate(q_vec) if b == low_bit]
        allowed = int(math.floor(self.max_lowbit_fraction * len(q_vec)))
        if len(low_indices) <= allowed:
            return q_vec
        higher = sorted(b for b in self.enc.Q if b > low_bit)
        if not higher:
            return q_vec
        promote_bit = higher[0]
        low_indices.sort(
            key=lambda idx: self.importance.score[idx], reverse=True
        )
        for idx in low_indices[: len(low_indices) - allowed]:
            q_vec[idx] = promote_bit
        return q_vec

    def _init_real_eval(self, real_eval_params: Dict) -> None:
        if self.real_eval is not None:
            return
        self.real_eval = RealTaskEvaluator(
            preset=real_eval_params["preset"],
            dataset=real_eval_params["dataset"],
            lf_sample_ratio=real_eval_params.get("lf_sample_ratio", 0.05),
            lf_epochs=real_eval_params.get("lf_epochs", 0.1),
            hf_sample_ratio=real_eval_params.get("hf_sample_ratio", 0.2),
            hf_epochs=real_eval_params.get("hf_epochs", 0.5),
            eval_task=real_eval_params["eval_task"],
            eval_shots=real_eval_params["eval_shots"],
            output_root=Path(real_eval_params["output_root"]),
            load_in_4bit=real_eval_params.get("load_in_4bit", True),
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(
        self,
        pop_size: int = 40,
        generations: int = 15,
        promote_k: int = 6,
        gamma: float = 1.5,
        hv_epsilon: float = 1e-3,
        hv_window: int = 3,
        ref_point: Optional[Tuple[float, float]] = None,
        use_warm_start: bool = True,
        use_importance_mutation: bool = True,
        use_coupled_mutation: bool = True,
        use_surrogate_promotion: bool = True,
        multi_fidelity: bool = True,
        lf_eval_mode: str = "proxy",
        progress_dir: Optional[Path] = None,
        real_eval_params: Optional[Dict] = None,
    ) -> Dict:
        """Run the evolutionary search and return the Pareto set."""
        real_eval_params = real_eval_params or self.real_eval_params
        set_seed(self.cfg.seed)

        nl = self.cfg.num_layers
        I = self.importance.score
        I_q = self.importance.I_q
        I_r = self.importance.I_r

        # ---- Warm start ------------------------------------------------
        if use_warm_start:
            q0, r0 = warm_start_from_importance(self.enc, I_q, I_r)
        else:
            q0 = [random.choice(self.enc.Q) for _ in range(nl)]
            r0 = [random.choice(self.enc.R) for _ in range(nl)]
        q0 = self._enforce_lowbit_cap(q0)

        population: List[Tuple[List[int], List[int]]] = [(q0, r0)]
        while len(population) < pop_size:
            q_j, r_j = jitter_configuration(
                q0, r0, self.enc, self.cfg.budget_bytes, self.mem
            )
            q_j = self._enforce_lowbit_cap(q_j)
            population.append((q_j, r_j))

        records: List[Dict] = []
        n_hf_total = 0
        n_lf_total = 0

        def _eval_and_record(
            q: List[int],
            r: List[int],
            high_fid: bool = False,
            generation: Optional[int] = None,
            cand_idx: Optional[int] = None,
            stage: str = "",
        ) -> Dict:
            q_loc = self._enforce_lowbit_cap(list(q))
            if lf_eval_mode == "real_task":
                self._init_real_eval(real_eval_params)
                perf, M = self.real_eval.evaluate(
                    q_loc, r, high_fidelity=high_fid,
                    generation=generation, cand_idx=cand_idx, stage=stage,
                )
                key = "phigh" if high_fid else "plow"
                return {"q": q_loc, "r": r, key: perf, "mem": M}

            plow, M = self.proxy.evaluate(q_loc, r, self.cfg.budget_bytes)
            rec = {"q": q_loc, "r": r, "plow": plow, "mem": M}
            if high_fid:
                phigh = float(plow + np.random.normal(scale=0.02))
                rec["phigh"] = phigh
                if use_surrogate_promotion:
                    self.sur.update(plow, phigh, M, q_loc, r, self.enc, I)
            return rec

        # ---- Initial population evaluation -----------------------------
        for i, (q, r) in enumerate(population):
            rec = _eval_and_record(
                q, r, high_fid=(i < promote_k),
                generation=-1, cand_idx=i, stage="init",
            )
            records.append(rec)
            if lf_eval_mode == "real_task":
                n_hf_total += 1

        hv_hist: List[float] = []
        gen_stats: List[Dict] = []

        # ---- Generations ----------------------------------------------
        for gen in range(generations):
            t0 = time.time()

            offsprings: List[Tuple[List[int], List[int]]] = []
            for pq, pr in population:
                for mq, mr in generate_k_nearest_atomic_neighbors(
                    pq, pr, self.enc, k=5
                ):
                    mq, mr = repair_to_budget(
                        mq, mr, self.enc, I_q, I_r, self.mem,
                        self.cfg.budget_bytes or float("inf"),
                    )
                    mq = self._enforce_lowbit_cap(mq)
                    offsprings.append((mq, mr))

            # LF evaluation for all offspring.
            off_recs = [
                _eval_and_record(
                    q, r, high_fid=False,
                    generation=gen, cand_idx=idx, stage="evo",
                )
                for idx, (q, r) in enumerate(offsprings)
            ]
            n_lf_total += len(off_recs)

            # HF promotion.
            promote_list: List[Dict] = []
            if not multi_fidelity:
                for rec in off_recs:
                    hf_rec = _eval_and_record(
                        rec["q"], rec["r"], high_fid=True,
                        generation=gen, stage="evo_hf",
                    )
                    rec["phigh"] = hf_rec["phigh"]
                n_hf_total += len(off_recs)
            else:
                if use_surrogate_promotion:
                    scored = [
                        (
                            self.sur.predict(
                                rec["plow"], rec["mem"], rec["q"], rec["r"],
                                self.enc, I,
                            ),
                            rec,
                        )
                        for rec in off_recs
                    ]
                    scored.sort(key=lambda x: x[0], reverse=True)
                    promote_list = [rec for _, rec in scored[:promote_k]]
                else:
                    promote_list = sorted(
                        off_recs, key=lambda r: r["plow"], reverse=True
                    )[:promote_k]

                for rec in promote_list:
                    hf_rec = _eval_and_record(
                        rec["q"], rec["r"], high_fid=True,
                        generation=gen, stage="evo_hf",
                    )
                    rec["phigh"] = hf_rec["phigh"]
                    if (
                        use_surrogate_promotion
                        and "plow" in rec
                        and "phigh" in rec
                    ):
                        self.sur.update(
                            rec["plow"], rec["phigh"], rec["mem"],
                            rec["q"], rec["r"], self.enc, I,
                        )
                n_hf_total += len(promote_list)

            records.extend(off_recs)

            # NSGA-II selection.
            evaluated = records
            objs = [
                (-rec.get("phigh", rec["plow"]), rec["mem"])
                for rec in evaluated
            ]
            feasible = [
                rec["mem"] <= (self.cfg.budget_bytes or float("inf"))
                for rec in evaluated
            ]
            fronts = non_dominated_sort_constrained(objs, feasible)

            new_pop: List[Tuple[List[int], List[int]]] = []
            for front in fronts:
                if len(new_pop) + len(front) <= pop_size:
                    new_pop.extend(
                        (evaluated[i]["q"], evaluated[i]["r"]) for i in front
                    )
                else:
                    cd = crowding_distance(objs, front)
                    front_sorted = sorted(
                        front, key=lambda i: cd[i], reverse=True
                    )
                    needed = pop_size - len(new_pop)
                    new_pop.extend(
                        (evaluated[i]["q"], evaluated[i]["r"])
                        for i in front_sorted[:needed]
                    )
                    break
            population = new_pop

            # Hypervolume tracking.
            pareto_pts = [objs[i] for i in fronts[0]] if fronts else objs
            if ref_point is None:
                max_cost = max(o[0] for o in objs)
                max_mem = max(o[1] for o in objs)
                ref = (
                    max_cost + abs(max_cost) * 0.1
                    if max_cost < 0
                    else max_cost * 1.1,
                    max_mem * 1.1,
                )
            else:
                ref = ref_point
            hv = hypervolume_2d(pareto_pts, ref)
            hv_hist.append(hv)

            gen_stats.append(
                {
                    "generation": gen,
                    "n_lf": len(off_recs),
                    "n_hf": len(promote_list) if multi_fidelity else len(off_recs),
                    "time_sec": time.time() - t0,
                    "hv": hv,
                }
            )

            # Hypervolume early stopping.
            if len(hv_hist) >= hv_window + 1:
                prev = hv_hist[-hv_window - 1]
                curr = hv_hist[-1]
                if prev > 0 and curr >= prev and (curr - prev) / prev < hv_epsilon:
                    break

        # ---- Finalize Pareto set --------------------------------------
        final = [rec for rec in records if "phigh" in rec]
        objs = [(-rec.get("phigh", rec["plow"]), rec["mem"]) for rec in final]
        feasible = [
            rec["mem"] <= (self.cfg.budget_bytes or float("inf"))
            for rec in final
        ]
        fronts = non_dominated_sort_constrained(objs, feasible)
        pareto = [final[i] for i in fronts[0]] if fronts else final

        return {
            "pareto": pareto,
            "all": final,
            "hv_hist": hv_hist,
            "config": {
                "pop_size": pop_size,
                "generations": generations,
                "promote_k": promote_k,
                "gamma": gamma,
                "multi_fidelity": multi_fidelity,
                "lf_eval_mode": lf_eval_mode,
            },
            "stats": {
                "gen_stats": gen_stats,
                "n_lf_total": n_lf_total,
                "n_hf_total": n_hf_total,
            },
        }
