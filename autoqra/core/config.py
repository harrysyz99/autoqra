"""
Configuration dataclasses and encodings for the AutoQRA search space.

Defines the search space S = (Q × R)^L, where each layer ℓ ∈ {1, ..., L}
selects a quantization bit-width q_ℓ ∈ Q and a LoRA adapter rank r_ℓ ∈ R.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AutoQRAConfig:
    """Top-level configuration for the AutoQRA search procedure.

    Attributes
    ----------
    num_layers : int
        Total number of transformer layers L.
    Q : list[int]
        Candidate quantization bit-widths (e.g., [2, 3, 4, 8]).
    R : list[int]
        Candidate LoRA adapter ranks (e.g., [4, 8, 16]).
    lora_precision_bits : int
        Precision p_r of LoRA adapter parameters (default: 16-bit).
    layer_param_bytes : list[int] | None
        Per-layer backbone parameter counts |W_ℓ|.
    lora_params_per_rank : list[int] | None
        Per-layer adapter parameters per unit rank (|A_ℓ| + |B_ℓ|)/r.
    seed : int
        Random seed for reproducibility.
    budget_bytes : float | None
        Memory budget B_max (in bytes). None disables the constraint.
    base_model_id : str | None
        HuggingFace model identifier for PTQ evaluation mode.
    task : str
        Downstream task type: 'generation' or 'classification'.
    """

    num_layers: int
    Q: List[int]
    R: List[int]
    lora_precision_bits: int = 16
    layer_param_bytes: Optional[List[int]] = None
    lora_params_per_rank: Optional[List[int]] = None
    seed: int = 42
    budget_bytes: Optional[float] = None
    base_model_id: Optional[str] = None
    task: str = "generation"


class ConfigEncoding:
    """Ordinal encoding for the discrete search space Q × R.

    Provides min-max normalized scores ``s_q`` and ``s_r`` for configuration
    embeddings used by the surrogate and the GP kernel, plus rounding
    operators ``round_Q`` and ``round_R`` that project continuous values
    back onto the discrete ladders.
    """

    def __init__(self, Q: List[int], R: List[int]) -> None:
        self.Q = sorted(Q)
        self.R = sorted(R)
        self.q_min, self.q_max = self.Q[0], self.Q[-1]
        self.r_min, self.r_max = self.R[0], self.R[-1]

    def s_q(self, q: int) -> float:
        """Min-max normalized bit-width score in [0, 1]."""
        if self.q_max <= self.q_min:
            return 0.0
        return (q - self.q_min) / (self.q_max - self.q_min)

    def s_r(self, r: int) -> float:
        """Min-max normalized rank score in [0, 1]."""
        if self.r_max <= self.r_min:
            return 0.0
        return (r - self.r_min) / (self.r_max - self.r_min)

    def round_Q(self, v: float) -> int:
        """Round a continuous value to the nearest discrete bit-width in Q."""
        return min(self.Q, key=lambda x: abs(x - v))

    def round_R(self, v: float) -> int:
        """Round a continuous value to the nearest discrete rank in R."""
        return min(self.R, key=lambda x: abs(x - v))

    def __repr__(self) -> str:
        return f"ConfigEncoding(Q={self.Q}, R={self.R})"
