"""
Proxy (low-fidelity) evaluator.

Uses an importance-weighted resource-coverage score as a cheap surrogate
for downstream performance, along with a soft penalty for memory budget
violation. This is used inside Phase II to rapidly rank offspring before
selecting a subset for expensive high-fidelity evaluation.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from autoqra.core.config import AutoQRAConfig, ConfigEncoding
from autoqra.core.importance import Importance
from autoqra.core.memory import MemoryModel


class ProxyEvaluator:
    """Low-fidelity proxy P_low(C) = Σ_ℓ I(ℓ) · (s_q(q_ℓ) + s_r(r_ℓ)) / 2.

    Parameters
    ----------
    cfg : AutoQRAConfig
        Search-space configuration.
    importance : Importance
        Per-layer sensitivity profiles.
    mem : MemoryModel
        Memory accounting model.
    enc : ConfigEncoding
        Ordinal encoding of the discrete ladders.
    target_avg_bits : float | None
        Optional target average bit-width. When set, a Gaussian bonus of
        maximum 0.5 is added around the target to bias the search toward
        user-specified operating points.
    """

    def __init__(
        self,
        cfg: AutoQRAConfig,
        importance: Importance,
        mem: MemoryModel,
        enc: ConfigEncoding,
        target_avg_bits: Optional[float] = None,
    ) -> None:
        self.cfg = cfg
        self.I = importance.score.astype(np.float64)
        self.mem = mem
        self.enc = enc
        self.target_avg_bits = target_avg_bits

    def evaluate(
        self,
        q: Sequence[int],
        r: Sequence[int],
        budget_bytes: Optional[float],
    ) -> Tuple[float, float]:
        """Return (proxy_score, memory_bytes) for a given configuration."""
        nl = self.cfg.num_layers
        total = 0.0
        for l in range(nl):
            total += (
                float(self.I[l])
                * (self.enc.s_q(q[l]) + self.enc.s_r(r[l]))
                * 0.5
            )
        score = total / max(1, nl)

        if self.target_avg_bits is not None:
            avg_bits = sum(q) / len(q)
            bit_diff = abs(avg_bits - self.target_avg_bits)
            bit_bonus = 0.5 * np.exp(-0.5 * (bit_diff / 1.0) ** 2)
            score = score + bit_bonus

        mem_bytes = self.mem.total_memory_bytes(q, r)
        if budget_bytes is not None and mem_bytes > budget_bytes:
            over = (mem_bytes - budget_bytes) / max(budget_bytes, 1.0)
            score = score - 2.0 * (over ** 2)

        return score, mem_bytes
