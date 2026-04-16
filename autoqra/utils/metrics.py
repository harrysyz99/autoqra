"""Orthogonality diagnostics for the sensitivity signals I_q and I_r."""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_orthogonality(I_q: np.ndarray, I_r: np.ndarray) -> Dict[str, float]:
    """Compute orthogonality metrics between two per-layer importance signals.

    Returns a dictionary with:
      - ``pearson`` / ``spearman`` : correlation coefficients.
      - ``cosine_similarity``      : cosine similarity of the raw signals.
      - ``top_k_overlap``          : Jaccard-style overlap of the top-k layers.
      - ``k``                      : number of top layers used for overlap.
    """
    from scipy import stats

    pearson, _ = stats.pearsonr(I_q, I_r)
    spearman, _ = stats.spearmanr(I_q, I_r)
    cosine = float(
        np.dot(I_q, I_r) / (np.linalg.norm(I_q) * np.linalg.norm(I_r))
    )

    k = max(1, len(I_q) // 3)
    top_k_q = set(np.argsort(I_q)[-k:])
    top_k_r = set(np.argsort(I_r)[-k:])
    rank_overlap = len(top_k_q & top_k_r) / k

    return {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "cosine_similarity": cosine,
        "top_k_overlap": float(rank_overlap),
        "k": k,
    }
