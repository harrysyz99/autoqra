"""
Memory accounting model for the joint quantization + LoRA configuration space.

Implements Eqs. 1–3 in the paper:

    M(C) = Σ_ℓ [ m^W_ℓ(q_ℓ) + m^A_ℓ(r_ℓ) + m^meta_ℓ(q_ℓ) ]

where:
  m^W_ℓ(q_ℓ) : quantized backbone storage = |W_ℓ| · q_ℓ / 8
  m^A_ℓ(r_ℓ) : LoRA adapter storage      = |A_ℓ + B_ℓ|_r · r_ℓ · p_r / 8
  m^meta_ℓ   : quantization metadata (scales, zero-points per block)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from autoqra.core.config import AutoQRAConfig


class MemoryModel:
    """Per-paper Eq. 1 memory accounting.

    Parameters
    ----------
    cfg : AutoQRAConfig
        Search-space configuration, including per-layer parameter counts.
    block_size : int
        Group size used by block-wise quantization (default: 128, matching HQQ).
    meta_precision_bits : int
        Bit-width used to store per-block scales and zero-points (default: 16).
    """

    def __init__(
        self,
        cfg: AutoQRAConfig,
        block_size: int = 128,
        meta_precision_bits: int = 16,
    ) -> None:
        self.cfg = cfg
        self.block_size = block_size
        self.meta_precision_bits = meta_precision_bits

        nl = cfg.num_layers
        if cfg.layer_param_bytes is None:
            self.layer_params = np.ones((nl,), dtype=np.float64) * 1e6
        else:
            self.layer_params = np.array(cfg.layer_param_bytes, dtype=np.float64)

        if cfg.lora_params_per_rank is None:
            self.lora_params_per_rank = np.ones((nl,), dtype=np.float64) * 1e5
        else:
            self.lora_params_per_rank = np.array(
                cfg.lora_params_per_rank, dtype=np.float64
            )

    def layer_memory_bytes(self, l: int, q_l: int, r_l: int) -> float:
        """Memory footprint of a single layer (Eqs. 2–3)."""
        # Quantized backbone (Eq. 2)
        m_W = self.layer_params[l] * (q_l / 8.0)

        # LoRA adapter storage (Eq. 3)
        m_A = (self.lora_params_per_rank[l] * r_l) * (
            self.cfg.lora_precision_bits / 8.0
        )

        # Per-block quantization metadata (scale + zero-point)
        num_blocks = int(np.ceil(self.layer_params[l] / self.block_size))
        bytes_per_block = 2 * (self.meta_precision_bits / 8.0)
        m_meta = num_blocks * bytes_per_block

        return float(m_W + m_A + m_meta)

    def total_memory_bytes(
        self, q: Sequence[int], r: Sequence[int]
    ) -> float:
        """Total configuration memory M(C) per Eq. 1."""
        return float(
            sum(
                self.layer_memory_bytes(i, q[i], r[i])
                for i in range(self.cfg.num_layers)
            )
        )

    def __repr__(self) -> str:
        return (
            f"MemoryModel(num_layers={self.cfg.num_layers}, "
            f"block_size={self.block_size})"
        )
