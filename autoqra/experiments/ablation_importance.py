"""
Ablation: orthogonality of the I_q and I_r sensitivity signals.

Demonstrates empirically that quantization sensitivity (Fidelity demand)
is essentially uncorrelated with adaptation sensitivity (Plasticity
demand), motivating the use of *separate* signals for warm-starting q
and r in Phase II rather than a single unified score.

Outputs:
  * Pearson, Spearman, cosine, and top-k-overlap statistics (JSON).
  * A 3-panel comparison figure (PNG/PDF).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from autoqra.utils.metrics import compute_orthogonality


def plot_comparison(I_q, I_r, ortho, output_dir: Path) -> None:
    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    nl = len(I_q)
    x = np.arange(nl)

    axes[0].bar(
        x - 0.175, I_q, 0.35, label="I_q (Fidelity)",
        color="#F18F01", alpha=0.8,
    )
    axes[0].bar(
        x + 0.175, I_r, 0.35, label="I_r (Plasticity)",
        color="#2E86AB", alpha=0.8,
    )
    axes[0].set_xlabel("Layer index")
    axes[0].set_ylabel("Importance")
    axes[0].set_title("Per-layer importance")
    axes[0].legend()

    axes[1].scatter(I_q, I_r, alpha=0.7, s=50, c=x, cmap="viridis")
    axes[1].set_xlabel("I_q (Fidelity)")
    axes[1].set_ylabel("I_r (Plasticity)")
    axes[1].set_title(f"Spearman ρ = {ortho['spearman']:.3f}")

    rank_q = np.argsort(np.argsort(I_q))[::-1] + 1
    rank_r = np.argsort(np.argsort(I_r))[::-1] + 1
    axes[2].scatter(rank_q, rank_r, alpha=0.7, s=50)
    axes[2].set_xlabel("Rank by I_q")
    axes[2].set_ylabel("Rank by I_r")
    axes[2].set_title(
        f"Top-{ortho['k']} overlap: {ortho['top_k_overlap']:.2f}"
    )

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "importance_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(output_dir / "importance_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--importance_json", required=True)
    parser.add_argument("--output_dir", default="results_importance_ablation")
    args = parser.parse_args()

    payload = json.loads(Path(args.importance_json).read_text())
    bb = payload.get("backbone_metric_per_layer", {})
    lr = payload.get("lora_metric_per_layer", {})
    n = max(len(bb), len(lr))
    I_q = np.array([float(bb.get(str(i), 0.0)) for i in range(n)])
    I_r = np.array([float(lr.get(str(i), 0.0)) for i in range(n)])
    I_q = I_q / max(I_q.sum(), 1e-8)
    I_r = I_r / max(I_r.sum(), 1e-8)

    ortho = compute_orthogonality(I_q, I_r)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "orthogonality.json", "w") as f:
        json.dump(ortho, f, indent=2)
    plot_comparison(I_q, I_r, ortho, out_dir)


if __name__ == "__main__":
    main()
