<h1 align="center">
AutoQRA: Joint Optimization of Mixed-Precision Quantization and Low-Rank Adapters for Efficient LLM Fine-Tuning
</h1>

<div align='center' style="font-size:18px;">
<p>
    <a href="[./paper.pdf](https://arxiv.org/abs/2602.22268)">
      <img src="https://img.shields.io/badge/Paper-ICML%202026-blue" alt="Paper"/>
    </a>
    <a href="#">
      <img src="https://img.shields.io/badge/Status-Accept-orange" alt="Status"/>
    </a>
    <a href="./LICENSE">
      <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
    </a>
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue" alt="Python"/>
    <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red" alt="PyTorch"/>
  </p>
</div>

## 🔥 Overview

We propose **AutoQRA**, a joint optimization framework that simultaneously optimizes the bit-width and LoRA rank configuration for each layer during mixed-precision quantized fine-tuning.

Existing sequential pipelines (quantize then fine-tune) fail to leverage the intricate interaction between quantization bit-width and LoRA rank. A carefully optimized quantization allocation with low quantization error does not always translate to strong fine-tuning performance, and different bit-width and rank configurations can lead to significantly varying outcomes under the same memory budget.

AutoQRA decomposes the optimization into **two stages**:

| Stage | Role | Module |
|-------|------|--------|
| **Stage 1: Global Multi-Fidelity Evolutionary Search** | Warm-start population from layer-wise importance priors; NSGA-II with surrogate-assisted candidate screening; multi-fidelity LF/HF promotion | `autoqra.search`, `autoqra.amq` |
| **Stage 2: Trust-Region Bayesian Optimization** | Local refinement on the Pareto front using Matern-5/2 GP + Expected Improvement; atomic-edit trust regions | `autoqra.search.phase3_bo` |

AutoQRA achieves performance close to full-precision fine-tuning with a memory footprint comparable to aggressive 2-bit quantization, enabling active compensation for quantization noise during training.

## 🗞️ News

- **`2026-04`**: Code and paper released.

## 🛠️ Installation

```bash
conda create -n autoqra python=3.10 -y
conda activate autoqra

git clone https://github.com/harrysyz99/autoqra.git
cd autoqra
pip install -e .
```

## 🚀 Quick Start

### CLI

```bash
autoqra \
    --importance_json sensitivity/Qwen3-4B_dataset_wikitext2_n_sample_128_seqlen_2048.json \
    --num_layers 36 \
    --bits 2 3 4 8 \
    --ranks 4 8 16 \
    --budget_bytes 4e9 \
    --phase2_pop 40 --phase2_generations 12 --phase2_promote 6 \
    --phase3_alpha 0.6 \
    --outdir ./results/qwen3_4b
```

### Python API

```python
from autoqra import AutoQRAConfig, AutoQRA

cfg = AutoQRAConfig(
    num_layers=36,
    Q=[2, 3, 4, 8],
    R=[4, 8, 16],
    seed=42,
    budget_bytes=4e9,
)

runner = AutoQRA(
    cfg,
    importance_json="sensitivity/Qwen3-4B_dataset_wikitext2_n_sample_128_seqlen_2048.json",
    target_avg_bits=4.0,
)
result = runner.run(
    outdir="./results",
    phase2_kwargs=dict(
        pop_size=40, generations=12, promote_k=6, gamma=1.5,
        lf_eval_mode="proxy",
    ),
    phase3_alpha=0.6,
)
print(result["phase3_best"])
```

### Outputs

| File | Content |
|------|---------|
| `phase2_pareto.json`  | Pareto front from Stage 1 evolutionary search |
| `phase3_selected.json`| Optimal configuration `(q*, r*)` from Stage 2 BO |
| `phase3_history.json` | Per-iteration BO trajectory |

## 📂 Repository Layout

```
autoqra/
├── autoqra/                          # Main Python package
│   ├── core/                         #   Config, importance, memory model, Pareto utilities
│   ├── amq/                          #   AMQ search framework (pymoo NSGA-II optimizer)
│   ├── evaluation/                   #   Proxy + real-task evaluators
│   ├── surrogate/                    #   Multi-fidelity MLP surrogate
│   ├── search/                       #   Stage 1 evolution + Stage 2 BO + operators
│   ├── training/                     #   HQQ quantization + per-layer LoRA SFT
│   ├── experiments/                  #   lm-eval harness + ablations
│   ├── utils/                        #   Numerical helpers, metrics, model loading
│   ├── cli.py                        #   `autoqra` CLI entry point
│   └── autoqra_runner.py             #   Top-level Stage 1 / 2 orchestrator
├── scripts/                          # Reproducible bash launchers
├── examples/                         # Self-contained Python examples
├── configs/                          # YAML configs (qwen3_4b, llama3_8b)
├── sensitivity/                      # Pre-computed layer sensitivity scores
├── tests/                            # Smoke tests for the public API
├── docs/                             # Architecture documentation
├── pyproject.toml
├── setup.py
├── requirements.txt
├── LICENSE
└── README.md
```

## 🧪 Testing

```bash
pip install pytest
pytest tests/ -v
```

## ⭐️ Citation

If you find this project useful, please cite us:

```bibtex
@inproceedings{autoqra2026,
  title     = {AutoQRA: Joint Optimization of Mixed-Precision Quantization
               and Low-Rank Adapters for Efficient LLM Fine-Tuning},
  author    = {Anonymous},
  booktitle = {Proceedings of the International Conference on Machine
               Learning (ICML)},
  year      = {2026}
}
```

## 🤝 Acknowledgement

This project builds upon [HQQ](https://github.com/mobiusml/hqq), [PEFT](https://github.com/huggingface/peft), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [pymoo](https://github.com/anyoptimization/pymoo), and [scikit-learn](https://github.com/scikit-learn/scikit-learn). We thank the authors of these projects.
