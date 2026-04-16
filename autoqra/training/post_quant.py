"""
Post-training persistence step.

The HQQ-quantized base model + LoRA adapter is kept on disk as a small
metadata record that the evaluation script can use to reconstruct the
exact quantization layout before loading the adapter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--qra_config", required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--per_channel", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "adapter_dir": args.adapter_dir,
        "qra_config": args.qra_config,
        "preset": args.preset,
    }
    with open(out_dir / "model_info.json", "w") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
