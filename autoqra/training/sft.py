"""
Supervised fine-tuning entrypoint for a single (q, r) configuration.

Invoked as a subprocess by :class:`autoqra.evaluation.RealTaskEvaluator`
during Phase II's high-fidelity promotion step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, Trainer, TrainingArguments

from autoqra.training.data import load_training_data
from autoqra.training.lora import build_lora_config
from autoqra.training.quantize import PRESETS, quantize_with_hqq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True, choices=list(PRESETS.keys()))
    parser.add_argument("--dataset", default="alpaca")
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--qra_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    args = parser.parse_args()

    model_id = PRESETS[args.preset]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.qra_config) as f:
        qra = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = quantize_with_hqq(model_id, qra["q"])
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        model.enable_input_require_grads()

    lora_cfg = build_lora_config(qra["r"])
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = load_training_data(tokenizer, args.dataset, args.sample_ratio)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        save_steps=9_999_999,
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    peak_mem = torch.cuda.max_memory_allocated()
    elapsed = time.time() - t0

    final_dir = output_dir / "final_model"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    with open(output_dir / "train_profile.json", "w") as f:
        json.dump(
            {
                "peak_mem_bytes": peak_mem,
                "train_time_sec": elapsed,
                "num_samples": len(train_ds),
                "epochs": args.epochs,
                "sample_ratio": args.sample_ratio,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
