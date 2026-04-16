#!/usr/bin/env bash
# AutoQRA end-to-end search on Qwen3-4B with proxy evaluation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python -m autoqra.cli \
    --importance_json sensitivity/Qwen3-4B_dataset_wikitext2_n_sample_128_seqlen_2048.json \
    --num_layers 36 \
    --bits 2 3 4 8 \
    --ranks 4 8 16 \
    --budget_bytes 4e9 \
    --phase2_pop 40 \
    --phase2_generations 12 \
    --phase2_promote 6 \
    --phase3_alpha 0.6 \
    --outdir ./results/qwen3_4b_proxy
