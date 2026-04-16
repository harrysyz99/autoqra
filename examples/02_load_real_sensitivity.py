"""Run AutoQRA against the bundled Qwen3-4B sensitivity file.

    python examples/02_load_real_sensitivity.py
"""

from __future__ import annotations

from pathlib import Path

from autoqra import AutoQRAConfig, AutoQRA


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    imp_json = (
        repo_root
        / "sensitivity"
        / "Qwen3-4B_dataset_wikitext2_n_sample_128_seqlen_2048.json"
    )

    cfg = AutoQRAConfig(
        num_layers=36,
        Q=[2, 3, 4, 8],
        R=[4, 8, 16],
        seed=42,
        budget_bytes=4e9,
    )
    runner = AutoQRA(
        cfg,
        importance_json=imp_json,
        target_avg_bits=4.0,
        max_lowbit_fraction=0.5,
    )
    result = runner.run(
        outdir=repo_root / "results" / "qwen3_4b_proxy",
        phase2_kwargs=dict(
            pop_size=40,
            generations=12,
            promote_k=6,
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
            f"avg_bits={sum(best['q'])/len(best['q']):.2f}, "
            f"avg_rank={sum(best['r'])/len(best['r']):.2f}, "
            f"phigh={best['phigh']:.4f}, mem={best['mem']:.2e}"
        )


if __name__ == "__main__":
    main()
