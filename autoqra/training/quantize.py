"""HQQ (Hessian-aware Quantization) helpers with per-layer or per-module bits."""

from __future__ import annotations

from typing import List

import torch
from transformers import AutoModelForCausalLM

PRESETS = {
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "qwen2.5-14b": "Qwen/Qwen2.5-14B",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-8b": "Qwen/Qwen3-8B",
}

TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _module_parent(layer, module_name: str):
    """Return the parent container holding ``module_name`` inside a layer."""
    attention_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}
    return layer.self_attn if module_name in attention_modules else layer.mlp


def quantize_with_hqq(model_id: str, q_array: List[int]):
    """Quantize a HuggingFace model with HQQ at a per-layer or per-module granularity.

    The ``q_array`` argument is interpreted by length:

      * ``len(q_array) == num_layers``               → per-layer bits
      * ``len(q_array) == num_layers × num_modules`` → per-module bits

    Layers with bits ≥ 8 are left in FP16 (no quantization).
    """
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    layers = model.model.layers
    num_layers = len(layers)
    num_modules = len(TARGET_MODULES)

    if len(q_array) == num_layers:
        for layer_idx, bits in enumerate(q_array):
            if bits >= 8:
                continue
            layer = layers[layer_idx]
            qcfg = BaseQuantizeConfig(nbits=bits, group_size=128)
            for module_name in TARGET_MODULES:
                parent = _module_parent(layer, module_name)
                linear = getattr(parent, module_name)
                setattr(
                    parent,
                    module_name,
                    HQQLinear(linear, qcfg, compute_dtype=torch.float16),
                )

    elif len(q_array) == num_layers * num_modules:
        for layer_idx in range(num_layers):
            layer = layers[layer_idx]
            for m_idx, module_name in enumerate(TARGET_MODULES):
                bits = q_array[layer_idx * num_modules + m_idx]
                if bits >= 8:
                    continue
                parent = _module_parent(layer, module_name)
                linear = getattr(parent, module_name)
                qcfg = BaseQuantizeConfig(nbits=bits, group_size=128)
                setattr(
                    parent,
                    module_name,
                    HQQLinear(linear, qcfg, compute_dtype=torch.float16),
                )

    else:
        raise ValueError(
            f"q_array length {len(q_array)} incompatible with "
            f"num_layers={num_layers} and num_modules={num_modules}."
        )

    return model
