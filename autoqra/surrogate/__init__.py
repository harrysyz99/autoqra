"""Surrogate models for multi-fidelity LF→HF performance prediction."""

from autoqra.surrogate.mlp import GELUSurrogateNet, SurrogateMLPPromotion

__all__ = ["GELUSurrogateNet", "SurrogateMLPPromotion"]
