"""Low-fidelity and high-fidelity evaluators for candidate configurations."""

from autoqra.evaluation.proxy import ProxyEvaluator
from autoqra.evaluation.real_task import RealTaskEvaluator

__all__ = ["ProxyEvaluator", "RealTaskEvaluator"]
