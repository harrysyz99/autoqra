"""
Phase I: Importance-Guided Warm Start.

Defines orthogonal per-layer sensitivity signals:
  - I_q(ℓ) : quantization sensitivity (Fidelity demand)
  - I_r(ℓ) : task adaptation capacity (Plasticity demand)

These signals are *orthogonal* by construction: layers requiring high
precision do not necessarily require high adapter rank, and vice-versa.
This orthogonality motivates the joint optimization in Phases II and III.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np

from autoqra.utils.numeric import normalize_minmax


class Importance:
    """Orthogonal sensitivity profiles for quantization and adaptation.

    Per paper Section 3.1:
      * ``I_q(ℓ)`` — Fidelity sensitivity, measured as degradation of the
        backbone task loss when layer ℓ is heavily quantized.
      * ``I_r(ℓ)`` — Plasticity demand, measured as adaptation gain from
        allocating additional LoRA rank at layer ℓ.

    Both signals are min-max normalized to [0, 1] and, optionally, combined
    into a single scalar score ``I(ℓ) = w_q · I_q(ℓ) + w_r · I_r(ℓ)`` for
    legacy components that consume a unified signal.

    Parameters
    ----------
    bb : np.ndarray
        Raw backbone (quantization) sensitivity scores per layer.
    lr : np.ndarray
        Raw adapter (LoRA-rank) sensitivity scores per layer.
    w_bb : float
        Weight of the backbone signal in the combined score.
    w_lr : float
        Weight of the LoRA signal in the combined score.
    """

    def __init__(
        self,
        bb: np.ndarray,
        lr: np.ndarray,
        w_bb: float = 0.5,
        w_lr: float = 0.5,
    ) -> None:
        self.I_q: np.ndarray = normalize_minmax(bb)
        self.I_r: np.ndarray = normalize_minmax(lr)
        self.score: np.ndarray = w_bb * self.I_q + w_lr * self.I_r
        self._w_bb = w_bb
        self._w_lr = w_lr

    @classmethod
    def from_json(
        cls,
        path: Union[str, Path],
        num_layers: int,
        w_bb: float = 0.5,
        w_lr: float = 0.5,
    ) -> "Importance":
        """Load importance scores from a JSON file.

        The JSON is expected to contain two dictionaries keyed by layer index
        (stringified), one for the backbone metric and one for the LoRA metric.
        """
        with open(path, "r") as f:
            payload = json.load(f)

        bb = np.zeros((num_layers,), dtype=np.float32)
        lr = np.zeros((num_layers,), dtype=np.float32)

        bb_dict = payload.get("backbone_metric_per_layer", {})
        lr_dict = payload.get("lora_metric_per_layer", {})

        for i in range(num_layers):
            bb[i] = float(bb_dict.get(str(i), bb_dict.get(i, 0.0)))
            lr[i] = float(lr_dict.get(str(i), lr_dict.get(i, 0.0)))

        return cls(bb, lr, w_bb=w_bb, w_lr=w_lr)

    def __repr__(self) -> str:
        return (
            f"Importance(num_layers={len(self.I_q)}, "
            f"I_q_mean={self.I_q.mean():.3f}, "
            f"I_r_mean={self.I_r.mean():.3f})"
        )
