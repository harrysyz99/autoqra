#!/usr/bin/env bash
# AutoQRA with real downstream evaluation (SFT + lm-eval).
# Requires a GPU and HF model access for the chosen --preset.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python -m autoqra.cli \
    --lf_eval_mode real_task \
    --preset qwen3-4b \
    --importance_json sensitivity/Qwen3-4B_dataset_wikitext2_n_sample_128_seqlen_2048.json \
    --num_layers 36 \
    --bits 2 3 4 8 \
    --ranks 4 8 16 \
    --budget_bytes 4e9 \
    --phase2_pop 20 \
    --phase2_generations 6 \
    --phase2_promote 3 \
    --phase3_alpha 0.6 \
    --outdir ./results/qwen3_4b_real_task
