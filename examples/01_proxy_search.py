"""Minimal end-to-end example using the in-process Python API.

Runs the full AutoQRA pipeline on a synthetic 8-layer importance signal
in proxy mode (no GPU, no HuggingFace download).

    python examples/01_proxy_search.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from autoqra import AutoQRAConfig, AutoQRA


def main() -> None:
    nl = 8
    rng = np.random.default_rng(0)
    bb = rng.random(nl)
    lr = rng.random(nl)
    payload = {
        "backbone_metric_per_layer": {str(i): float(bb[i]) for i in range(nl)},
        "lora_metric_per_layer": {str(i): float(lr[i]) for i in range(nl)},
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        imp_json = tmp / "imp.json"
        imp_json.write_text(json.dumps(payload))

        cfg = AutoQRAConfig(
            num_layers=nl,
            Q=[2, 4, 8],
            R=[4, 8, 16],
            seed=42,
            budget_bytes=1e8,
        )
        runner = AutoQRA(cfg, importance_json=imp_json)
        result = runner.run(
            outdir=tmp / "out",
            phase2_kwargs=dict(
                pop_size=10,
                generations=3,
                promote_k=3,
                gamma=1.5,
                lf_eval_mode="proxy",
            ),
            phase3_alpha=0.6,
        )

        pareto = result["phase2"]["pareto"]
        best = result["phase3_best"]
        print(f"Phase II Pareto points : {len(pareto)}")
        if best is not None:
            print(
                f"Phase III selected     : "
                f"phigh={best['phigh']:.4f}, mem={best['mem']:.2e}, "
                f"utility={best['utility']:.4f}"
            )


if __name__ == "__main__":
    main()
