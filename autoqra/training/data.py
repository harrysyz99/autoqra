"""Instruction-tuning / pre-training data loaders."""

from __future__ import annotations

from datasets import Dataset, load_dataset


def load_training_data(
    tokenizer, dataset_name: str, sample_ratio: float, max_length: int = 512
) -> Dataset:
    """Load, format, and tokenize an SFT dataset.

    Currently supports:
      * ``alpaca`` — tatsu-lab/alpaca, formatted with an instruction /
        response template.
    """
    if dataset_name == "alpaca":
        raw = load_dataset("tatsu-lab/alpaca", split="train")
        rows = []
        for item in raw:
            text = f"### Instruction:\n{item['instruction']}"
            if item.get("input"):
                text += f"\n### Input:\n{item['input']}"
            text += f"\n### Response:\n{item['output']}"
            rows.append({"text": text})
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    n_samples = max(1, int(len(rows) * sample_ratio))
    dataset = Dataset.from_list(rows[:n_samples])

    def _tokenize(example):
        out = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    return dataset.map(_tokenize, remove_columns=dataset.column_names)
