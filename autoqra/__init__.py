"""
AutoQRA: Balancing Fidelity and Plasticity for Mixed-Precision Fine-Tuning.

Joint per-layer quantization bit-width and LoRA adapter rank optimization
via a three-phase framework:

  Phase I:   Importance-Guided Warm Start
  Phase II:  Multi-Fidelity Evolutionary Search (constrained NSGA-II)
  Phase III: Trust-Region Bayesian Optimization (Matern-5/2 GP + Expected Improvement)
"""

__version__ = "1.0.0"
__author__ = "AutoQRA Authors"

from autoqra.core.config import AutoQRAConfig, ConfigEncoding
from autoqra.core.importance import Importance
from autoqra.core.memory import MemoryModel
from autoqra.evaluation.proxy import ProxyEvaluator
from autoqra.evaluation.real_task import RealTaskEvaluator
from autoqra.surrogate.mlp import GELUSurrogateNet, SurrogateMLPPromotion
from autoqra.search.phase2_evolution import PhaseIIEvolution
from autoqra.search.phase3_bo import PhaseIIIBO
from autoqra.autoqra_runner import AutoQRA

__all__ = [
    "AutoQRAConfig",
    "ConfigEncoding",
    "Importance",
    "MemoryModel",
    "ProxyEvaluator",
    "RealTaskEvaluator",
    "GELUSurrogateNet",
    "SurrogateMLPPromotion",
    "PhaseIIEvolution",
    "PhaseIIIBO",
    "AutoQRA",
]
