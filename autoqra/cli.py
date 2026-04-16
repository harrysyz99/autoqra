"""Command-line entrypoint: ``python -m autoqra.cli ...``."""

from __future__ import annotations

import argparse
from pathlib import Path

from autoqra.core.config import AutoQRAConfig
from autoqra.autoqra_runner import AutoQRA


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="autoqra",
        description=(
            "AutoQRA: joint per-layer quantization and LoRA-rank "
            "optimization under a memory budget."
        ),
    )
    p.add_argument("--num_layers", type=int, default=28)
    p.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4, 6, 8])
    p.add_argument("--ranks", type=int, nargs="+", default=[4, 6, 8, 10, 12, 16])
    p.add_argument("--budget_bytes", type=float, default=None)
    p.add_argument("--importance_json", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--lf_eval_mode",
        choices=["proxy", "ptq", "real_task"],
        default="proxy",
    )
    p.add_argument("--preset", type=str, default=None)
    p.add_argument("--target_avg_bits", type=float, default=None)
    p.add_argument("--max_lowbit_fraction", type=float, default=0.5)

    p.add_argument("--phase2_pop", type=int, default=40)
    p.add_argument("--phase2_generations", type=int, default=12)
    p.add_argument("--phase2_promote", type=int, default=6)
    p.add_argument("--phase2_gamma", type=float, default=1.5)

    p.add_argument("--phase3_alpha", type=float, default=None)
    p.add_argument("--outdir", type=str, default="./results_autoqra")
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)

    cfg = AutoQRAConfig(
        num_layers=args.num_layers,
        Q=args.bits,
        R=args.ranks,
        seed=args.seed,
        budget_bytes=args.budget_bytes,
    )

    runner = AutoQRA(
        cfg,
        importance_json=Path(args.importance_json),
        target_avg_bits=args.target_avg_bits,
        max_lowbit_fraction=args.max_lowbit_fraction,
    )
    runner.run(
        outdir=Path(args.outdir),
        phase2_kwargs=dict(
            pop_size=args.phase2_pop,
            generations=args.phase2_generations,
            promote_k=args.phase2_promote,
            gamma=args.phase2_gamma,
            lf_eval_mode=args.lf_eval_mode,
        ),
        phase3_alpha=args.phase3_alpha,
    )


if __name__ == "__main__":
    main()
