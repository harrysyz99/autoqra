"""Small numerical helpers shared across the package."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python and NumPy random number generators."""
    random.seed(seed)
    np.random.seed(seed)


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    """Min-max normalize a 1-D array to [0, 1].

    If the input is constant (vmax == vmin) or empty, returns an
    all-zeros array of matching shape.
    """
    if x.size == 0:
        return x
    vmin = float(np.min(x))
    vmax = float(np.max(x))
    if vmax == vmin:
        return np.zeros_like(x)
    return (x - vmin) / (vmax - vmin)
