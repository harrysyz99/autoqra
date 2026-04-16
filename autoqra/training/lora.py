"""Per-layer / per-module LoRA rank pattern builder."""

from __future__ import annotations

from collections import Counter
from typing import List

from peft import LoraConfig

from autoqra.training.quantize import TARGET_MODULES


def _module_key(layer_idx: int, module: str) -> str:
    container = (
        "self_attn"
        if module in {"q_proj", "k_proj", "v_proj", "o_proj"}
        else "mlp"
    )
    return f"model.layers.{layer_idx}.{container}.{module}"


def build_lora_config(r_array: List[int]) -> LoraConfig:
    """Construct a :class:`peft.LoraConfig` with per-layer/per-module ranks.

    Convention on ``r_array`` length:
      * one entry per layer     → all modules in a layer share the rank.
      * one entry per module    → 7 entries per layer (Q, K, V, O, up,
        gate, down), giving a per-module rank.

    The base rank is the most frequent value in ``r_array``; deviations
    from the base are recorded in ``rank_pattern`` / ``alpha_pattern``
    (alpha is always 2 × rank).
    """
    if not r_array or len(set(r_array)) <= 1:
        avg_rank = int(sum(r_array) / len(r_array)) if r_array else 16
        return LoraConfig(
            r=avg_rank,
            lora_alpha=avg_rank * 2,
            target_modules=TARGET_MODULES,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    num_modules = len(TARGET_MODULES)
    base_rank = Counter(r_array).most_common(1)[0][0]
    rank_pattern: dict = {}
    alpha_pattern: dict = {}

    if len(r_array) % num_modules == 0 and len(r_array) > 100:
        num_layers = len(r_array) // num_modules
        for layer_idx in range(num_layers):
            for m_idx, module in enumerate(TARGET_MODULES):
                rank = r_array[layer_idx * num_modules + m_idx]
                if rank != base_rank:
                    key = _module_key(layer_idx, module)
                    rank_pattern[key] = rank
                    alpha_pattern[key] = rank * 2
    else:
        for layer_idx, layer_rank in enumerate(r_array):
            if layer_rank != base_rank:
                for module in TARGET_MODULES:
                    key = _module_key(layer_idx, module)
                    rank_pattern[key] = layer_rank
                    alpha_pattern[key] = layer_rank * 2

    if rank_pattern:
        return LoraConfig(
            r=base_rank,
            lora_alpha=base_rank * 2,
            target_modules=TARGET_MODULES,
            rank_pattern=rank_pattern,
            alpha_pattern=alpha_pattern,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    return LoraConfig(
        r=base_rank,
        lora_alpha=base_rank * 2,
        target_modules=TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
