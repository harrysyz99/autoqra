"""
lm-eval harness entrypoint for a quantized + LoRA model.

Reconstructs the HQQ quantization layout from ``model_info.json`` (produced
by :mod:`autoqra.training.post_quant`), attaches the LoRA adapter, and
runs a single task through ``lm_eval.simple_evaluate``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from autoqra.training.quantize import PRESETS, TARGET_MODULES


def _reload_hqq_model(model_id: str, q_array):
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    layers = model.model.layers
    num_modules = len(TARGET_MODULES)

    if len(q_array) > 100:
        num_layers = len(q_array) // num_modules
        for layer_idx in range(num_layers):
            for m_idx, module in enumerate(TARGET_MODULES):
                bits = q_array[layer_idx * num_modules + m_idx]
                if bits >= 8:
                    continue
                parent = (
                    layers[layer_idx].self_attn
                    if module in {"q_proj", "k_proj", "v_proj", "o_proj"}
                    else layers[layer_idx].mlp
                )
                linear = getattr(parent, module)
                setattr(
                    parent,
                    module,
                    HQQLinear(
                        linear,
                        BaseQuantizeConfig(nbits=bits, group_size=128),
                        compute_dtype=torch.float16,
                    ),
                )
    else:
        for layer_idx, bits in enumerate(q_array):
            if bits >= 8:
                continue
            for module in TARGET_MODULES:
                parent = (
                    layers[layer_idx].self_attn
                    if module in {"q_proj", "k_proj", "v_proj", "o_proj"}
                    else layers[layer_idx].mlp
                )
                linear = getattr(parent, module)
                setattr(
                    parent,
                    module,
                    HQQLinear(
                        linear,
                        BaseQuantizeConfig(nbits=bits, group_size=128),
                        compute_dtype=torch.float16,
                    ),
                )
    return model


def main() -> None:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--task", default="winogrande")
    parser.add_argument("--shots", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    info_path = model_path / "model_info.json"

    if info_path.exists():
        info = json.loads(info_path.read_text())
        with open(info["qra_config"]) as f:
            qra = json.load(f)
        model_id = PRESETS.get(info["preset"], info["preset"])
        model = _reload_hqq_model(model_id, qra["q"])
        model = PeftModel.from_pretrained(model, info["adapter_dir"])
        model.eval()
        model.tie_weights = lambda: None
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[args.task],
        num_fewshot=args.shots,
        batch_size=args.batch_size,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
