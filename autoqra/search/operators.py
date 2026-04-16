"""
Sensitivity-guided search operators used by the Phase II evolutionary
strategy: warm-start initialization, deterministic repair, importance-
guided mutation, uniform crossover, and random jitter.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

from autoqra.core.config import ConfigEncoding
from autoqra.core.memory import MemoryModel


def repair_to_budget(
    q: List[int],
    r: List[int],
    enc: ConfigEncoding,
    I_q: np.ndarray,
    I_r: np.ndarray,
    mem: MemoryModel,
    budget_bytes: float,
    epsilon: float = 1e-8,
) -> Tuple[List[int], List[int]]:
    """Deterministic repair procedure (Eqs. 4–5).

    While M(C) > B_max, iteratively apply the single atomic downgrade
    (either q_ℓ or r_ℓ) that minimizes sensitivity-per-saved-memory:

        (ℓ*, t*) = argmin_{ℓ, t ∈ {q, r}}  (I_t(ℓ) + ε) / ΔM_t(ℓ)

    The loop terminates either when the configuration is feasible or when
    every variable is already at its minimum (in which case the search
    space cannot accommodate the budget).
    """
    q_out = list(q)
    r_out = list(r)

    def _lower(val: int, choices: List[int]) -> Optional[int]:
        try:
            idx = choices.index(val)
            return choices[idx - 1] if idx > 0 else None
        except ValueError:
            return None

    while mem.total_memory_bytes(q_out, r_out) > budget_bytes:
        best_cost = float("inf")
        best_layer: Optional[int] = None
        best_type: Optional[str] = None
        best_new_val: Optional[int] = None

        for l in range(len(q_out)):
            q_lower = _lower(q_out[l], enc.Q)
            if q_lower is not None:
                q_tmp = list(q_out)
                q_tmp[l] = q_lower
                delta = (
                    mem.total_memory_bytes(q_out, r_out)
                    - mem.total_memory_bytes(q_tmp, r_out)
                )
                if delta > 0:
                    cost = (I_q[l] + epsilon) / delta
                    if cost < best_cost:
                        best_cost, best_layer = cost, l
                        best_type, best_new_val = "q", q_lower

            r_lower = _lower(r_out[l], enc.R)
            if r_lower is not None:
                r_tmp = list(r_out)
                r_tmp[l] = r_lower
                delta = (
                    mem.total_memory_bytes(q_out, r_out)
                    - mem.total_memory_bytes(q_out, r_tmp)
                )
                if delta > 0:
                    cost = (I_r[l] + epsilon) / delta
                    if cost < best_cost:
                        best_cost, best_layer = cost, l
                        best_type, best_new_val = "r", r_lower

        if best_layer is None:
            break
        if best_type == "q":
            q_out[best_layer] = best_new_val
        else:
            r_out[best_layer] = best_new_val

    return q_out, r_out


def warm_start_from_importance(
    enc: ConfigEncoding,
    I_q: np.ndarray,
    I_r: np.ndarray,
    tau_q: str = "identity",
    tau_r: str = "identity",
) -> Tuple[List[int], List[int]]:
    """Phase I warm-start (Eq. 6).

        q_ℓ^(0) = round_Q( τ_q( Ĩ_q(ℓ) ) )
        r_ℓ^(0) = round_R( τ_r( Ĩ_r(ℓ) ) )

    where Ĩ denotes normalized importance and τ is a shaping function.
    """

    def _shape(x: float, kind: str) -> float:
        if kind == "sqrt":
            return math.sqrt(max(0.0, x))
        if kind == "square":
            return x * x
        return x  # identity

    q: List[int] = []
    r: List[int] = []
    for l in range(len(I_q)):
        q_cont = enc.q_min + (enc.q_max - enc.q_min) * _shape(float(I_q[l]), tau_q)
        r_cont = enc.r_min + (enc.r_max - enc.r_min) * _shape(float(I_r[l]), tau_r)
        q.append(enc.round_Q(q_cont))
        r.append(enc.round_R(r_cont))
    return q, r


def jitter_configuration(
    q: List[int],
    r: List[int],
    enc: ConfigEncoding,
    budget_bytes: Optional[float],
    mem: MemoryModel,
    max_jitter: int = 4,
) -> Tuple[List[int], List[int]]:
    """Random jitter to diversify a warm-started population."""
    q_j = list(q)
    r_j = list(r)
    for _ in range(max_jitter):
        l = random.randrange(len(q))
        if random.random() < 0.5:
            q_j[l] = random.choice(enc.Q)
        else:
            r_j[l] = random.choice(enc.R)
        if (
            budget_bytes is not None
            and mem.total_memory_bytes(q_j, r_j) > budget_bytes
            and random.random() < 0.5
        ):
            l2 = random.randrange(len(q))
            if random.random() < 0.5:
                q_j[l2] = enc.Q[0]
            else:
                r_j[l2] = enc.R[0]
    return q_j, r_j


def mutate_importance_guided(
    q: List[int],
    r: List[int],
    enc: ConfigEncoding,
    I_q: np.ndarray,
    I_r: np.ndarray,
    gamma: float,
    mem: MemoryModel,
    budget_bytes: Optional[float],
    use_coupled: bool = True,
) -> Tuple[List[int], List[int]]:
    """Sensitivity-guided mutation with budget-balanced coupling.

    Two mutation modes:

    1. **Single mutation** — select layer ℓ with Pr_t(ℓ) ∝ I_t(ℓ)^γ, then
       increment or decrement one ladder step (probability proportional
       to layer importance).

    2. **Coupled mutation** (probability 0.3 when a budget is set) — apply
       one memory-*increasing* edit, then call :func:`repair_to_budget`
       to compensate with memory-*decreasing* edits elsewhere.
    """
    q2 = list(q)
    r2 = list(r)
    nl = len(q)

    if use_coupled and budget_bytes is not None and random.random() < 0.3:
        probs_q = np.power(np.maximum(I_q, 1e-8), gamma)
        probs_q /= probs_q.sum()
        l_primary = int(np.random.choice(np.arange(nl), p=probs_q))

        if random.random() < 0.5:
            idx = enc.Q.index(q2[l_primary])
            if idx < len(enc.Q) - 1:
                q2[l_primary] = enc.Q[idx + 1]
        else:
            probs_r = np.power(np.maximum(I_r, 1e-8), gamma)
            probs_r /= probs_r.sum()
            l_primary = int(np.random.choice(np.arange(nl), p=probs_r))
            idx = enc.R.index(r2[l_primary])
            if idx < len(enc.R) - 1:
                r2[l_primary] = enc.R[idx + 1]

        if mem.total_memory_bytes(q2, r2) > budget_bytes:
            q2, r2 = repair_to_budget(q2, r2, enc, I_q, I_r, mem, budget_bytes)
    else:
        probs = np.power(np.maximum(I_q + I_r, 1e-8), gamma)
        probs /= probs.sum()
        l = int(np.random.choice(np.arange(nl), p=probs))

        if random.random() < 0.5:
            idx = enc.Q.index(q2[l])
            if random.random() < float(I_q[l]):
                if idx < len(enc.Q) - 1:
                    q2[l] = enc.Q[idx + 1]
            else:
                if idx > 0:
                    q2[l] = enc.Q[idx - 1]
        else:
            idx = enc.R.index(r2[l])
            if random.random() < float(I_r[l]):
                if idx < len(enc.R) - 1:
                    r2[l] = enc.R[idx + 1]
            else:
                if idx > 0:
                    r2[l] = enc.R[idx - 1]

    return q2, r2


def crossover_uniform(
    q1: Sequence[int],
    r1: Sequence[int],
    q2: Sequence[int],
    r2: Sequence[int],
) -> Tuple[List[int], List[int]]:
    """Per-layer uniform crossover."""
    nl = len(q1)
    return (
        [q1[i] if random.random() < 0.5 else q2[i] for i in range(nl)],
        [r1[i] if random.random() < 0.5 else r2[i] for i in range(nl)],
    )
