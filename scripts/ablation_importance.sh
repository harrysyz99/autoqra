#!/usr/bin/env bash
# Reproduce the I_q vs I_r orthogonality ablation (Section 5.2).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

python -m autoqra.experiments.ablation_importance \
    --importance_json sensitivity/Qwen3-4B_dataset_wikitext2_n_sample_128_seqlen_2048.json \
    --output_dir ./results/ablation_importance
