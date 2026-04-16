"""
Phase III: Trust-Region Bayesian Optimization.

A local Bayesian optimization loop with:

  * Scalarized utility  f(C; α) = α · P̂(C) − (1 − α) · M̂(C)     (Eq. 8)
  * Ordinal embedding   ψ(C) = [s_q, s_r, q_cov, r_cov]           (Eq. 9)
  * GP surrogate        Matérn-5/2 kernel + white noise
  * Acquisition         Expected Improvement                        (Eq. 10)
  * Search space        union of atomic-edit neighborhoods around
                        all Phase II Pareto points
  * Stopping            max EI < ε_ei                               (Eq. 13)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from autoqra.core.config import ConfigEncoding
from autoqra.core.importance import Importance
from autoqra.core.memory import MemoryModel
from autoqra.search.neighbors import generate_k_nearest_atomic_neighbors
from autoqra.search.operators import repair_to_budget


class PhaseIIIBO:
    """Trust-region Bayesian optimization warm-started from the Pareto front.

    Parameters
    ----------
    enc : ConfigEncoding
        Ordinal encoding for (Q, R).
    importance : Importance
        Sensitivity profiles; used inside the GP feature map.
    mem_model : MemoryModel
        Memory accountant used by ``repair_to_budget``.
    budget_bytes : float
        Hard memory budget.
    I_q, I_r : np.ndarray
        Per-layer sensitivity signals (passed into ``repair_to_budget``).
    k_neighbors : int
        Neighbors per Pareto point in the trust-region construction.
    epsilon_ei : float
        Convergence threshold on max Expected Improvement.
    max_iterations : int
        Maximum BO iterations.
    """

    def __init__(
        self,
        enc: ConfigEncoding,
        importance: Importance,
        mem_model: MemoryModel,
        budget_bytes: float,
        I_q: np.ndarray,
        I_r: np.ndarray,
        k_neighbors: int = 5,
        epsilon_ei: float = 1e-4,
        max_iterations: int = 3,
    ) -> None:
        self.enc = enc
        self.I = importance.score
        self.mem = mem_model
        self.budget_bytes = budget_bytes
        self.I_q = I_q
        self.I_r = I_r
        self.k_neighbors = k_neighbors
        self.epsilon_ei = epsilon_ei
        self.max_iterations = max_iterations

        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * Matern(length_scale=1.0, nu=2.5)
            + WhiteKernel(1e-3)
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, random_state=0
        )
        self.X_train: List[np.ndarray] = []
        self.y_train: List[float] = []
        self.best_y: float = -float("inf")

        # Filled in by ``fit``.
        self.perf_norm: Tuple[float, float] = (0.0, 1.0)
        self.mem_norm: Tuple[float, float] = (0.0, 1.0)
        self.alpha: float = 0.5

    # ------------------------------------------------------------------
    # Encoding and scalarization
    # ------------------------------------------------------------------
    def encode(self, q, r) -> np.ndarray:
        """Ordinal embedding ψ(C) (Eq. 9)."""
        qn = np.array([self.enc.s_q(x) for x in q], dtype=np.float64)
        rn = np.array([self.enc.s_r(x) for x in r], dtype=np.float64)
        q_cov = float(np.dot(self.I, qn) / len(qn))
        r_cov = float(np.dot(self.I, rn) / len(rn))
        return np.concatenate([qn, rn, [q_cov, r_cov]], axis=0)

    def scalarize(
        self,
        perf: float,
        mem: float,
        alpha: float,
        perf_norm: Tuple[float, float],
        mem_norm: Tuple[float, float],
    ) -> float:
        """Scalarized utility f(C; α) (Eq. 8)."""
        p_norm = (perf - perf_norm[0]) / max(perf_norm[1] - perf_norm[0], 1e-8)
        m_norm = (mem - mem_norm[0]) / max(mem_norm[1] - mem_norm[0], 1e-8)
        return alpha * p_norm - (1 - alpha) * m_norm

    # ------------------------------------------------------------------
    # Fitting and proposing
    # ------------------------------------------------------------------
    def fit(self, pareto: List[dict], alpha: float) -> None:
        """Warm-start the GP from the Phase II Pareto front."""
        perfs = np.array([p["phigh"] for p in pareto], dtype=np.float64)
        mems = np.array([p["mem"] for p in pareto], dtype=np.float64)
        self.perf_norm = (float(np.min(perfs)), float(np.max(perfs)))
        self.mem_norm = (float(np.min(mems)), float(np.max(mems)))
        self.alpha = alpha

        for p in pareto:
            x = self.encode(p["q"], p["r"])
            y = self.scalarize(
                p["phigh"], p["mem"], alpha, self.perf_norm, self.mem_norm
            )
            self.X_train.append(x)
            self.y_train.append(y)
            if y > self.best_y:
                self.best_y = y

        if self.X_train:
            self.gp.fit(np.vstack(self.X_train), np.array(self.y_train))

    def expected_improvement(
        self, mu: np.ndarray, sigma: np.ndarray
    ) -> np.ndarray:
        """Expected Improvement acquisition (Eq. 10)."""
        from scipy.stats import norm

        ei = np.zeros_like(mu)
        mask = sigma > 1e-8
        if not np.any(mask):
            return ei
        z = np.zeros_like(mu)
        z[mask] = (mu[mask] - self.best_y) / sigma[mask]
        ei[mask] = sigma[mask] * (
            z[mask] * norm.cdf(z[mask]) + norm.pdf(z[mask])
        )
        return ei

    def propose_multi_start(
        self,
        pareto_points: List[Tuple[List[int], List[int]]],
        k_neighbors: int = 5,
    ):
        """Select next candidate from the union of all Pareto trust regions."""
        neighbors = set()
        for q_p, r_p in pareto_points:
            for q, r in generate_k_nearest_atomic_neighbors(
                q_p, r_p, self.enc, k=k_neighbors
            ):
                q_rep, r_rep = repair_to_budget(
                    q, r, self.enc, self.I_q, self.I_r, self.mem,
                    self.budget_bytes,
                )
                if self.mem.total_memory_bytes(q_rep, r_rep) <= self.budget_bytes:
                    neighbors.add((tuple(q_rep), tuple(r_rep)))
            neighbors.add((tuple(q_p), tuple(r_p)))

        omega = [(list(q), list(r)) for q, r in neighbors]
        if not omega:
            q, r = pareto_points[0]
            return q, r, 0.0, 0

        X = np.vstack([self.encode(q, r) for q, r in omega])
        mu, sigma = self.gp.predict(X, return_std=True)
        ei = self.expected_improvement(mu, sigma)
        idx = int(np.argmax(ei))
        return omega[idx][0], omega[idx][1], float(ei[idx]), len(omega)

    def update(
        self, q: List[int], r: List[int], perf: float, mem: float
    ) -> bool:
        """Fold a new observation into the GP; return True if it improved."""
        x = self.encode(q, r)
        y = self.scalarize(perf, mem, self.alpha, self.perf_norm, self.mem_norm)
        improved = y > self.best_y
        if improved:
            self.best_y = y
        self.X_train.append(x)
        self.y_train.append(y)
        self.gp.fit(np.vstack(self.X_train), np.array(self.y_train))
        return improved

    def has_converged(self, max_ei: float) -> bool:
        """Convergence criterion (Eq. 13): max EI < ε_ei."""
        return max_ei < self.epsilon_ei
