"""
Multi-fidelity surrogate (Eq. 7).

A 2-layer MLP with GELU activation maps (P_low, M, descriptors of C) →
predicted P_high. Trained with Huber loss and L2 regularization:

    θ_s* = argmin_θ  Σ_i ρ(Φ_s(x_i) - P(C_i; b_S))  +  λ ||θ||²

Inputs are standardized (z-score) and early stopping is applied on a
validation split (20%). The surrogate is *re-trained each generation*
once enough (P_low, P_high) pairs have been observed.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from autoqra.core.config import ConfigEncoding


class GELUSurrogateNet(nn.Module):
    """2-layer MLP with GELU activation for P_low → P_high prediction."""

    def __init__(
        self, input_dim: int, hidden_dims: Tuple[int, int] = (64, 32)
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1), nn.GELU(),
            nn.Linear(h1, h2), nn.GELU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SurrogateMLPPromotion:
    """Multi-fidelity surrogate for LF→HF promotion (Eq. 7).

    Maintains an online training set of (features, P_high) pairs and refits
    the underlying MLP whenever ``update`` is called and the sample count
    exceeds ``min_samples``. A standard-scaler is fit on the entire buffer.

    Parameters
    ----------
    hidden_dims : tuple[int, int]
        MLP hidden layer sizes.
    val_fraction : float
        Fraction of the training buffer used for validation (early stopping).
    patience : int
        Early-stopping patience (epochs without validation-loss improvement).
    lr : float
        Adam learning rate.
    max_epochs : int
        Maximum training epochs per refit.
    min_samples : int
        Minimum number of pairs required before fitting starts.
    huber_delta : float
        Huber-loss threshold (robustness to low-budget outliers).
    l2_lambda : float
        L2 regularization coefficient.
    """

    def __init__(
        self,
        hidden_dims: Tuple[int, int] = (64, 32),
        val_fraction: float = 0.2,
        patience: int = 10,
        lr: float = 1e-3,
        max_epochs: int = 500,
        min_samples: int = 10,
        huber_delta: float = 1.0,
        l2_lambda: float = 0.01,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.val_fraction = val_fraction
        self.patience = patience
        self.lr = lr
        self.max_epochs = max_epochs
        self.min_samples = min_samples
        self.huber_delta = huber_delta
        self.l2_lambda = l2_lambda

        self.scaler = StandardScaler()
        self.model: Optional[GELUSurrogateNet] = None
        self.input_dim: Optional[int] = None
        self.X: List[List[float]] = []
        self.y: List[float] = []
        self.is_fitted: bool = False

    def _features(
        self,
        plow: float,
        mem: float,
        q: Sequence[int],
        r: Sequence[int],
        enc: ConfigEncoding,
        I: np.ndarray,
    ) -> List[float]:
        """Build the feature vector (P_low, M, q_cov, r_cov, q_hist, r_hist)."""
        q_list, r_list = list(q), list(r)
        q_hist = [q_list.count(v) / len(q_list) for v in enc.Q]
        r_hist = [r_list.count(v) / len(r_list) for v in enc.R]
        q_cov = float(np.dot(I, np.array([enc.s_q(x) for x in q])) / len(q))
        r_cov = float(np.dot(I, np.array([enc.s_r(x) for x in r])) / len(r))
        return [plow, mem, q_cov, r_cov] + q_hist + r_hist

    def update(
        self,
        plow: float,
        phigh: float,
        mem: float,
        q: Sequence[int],
        r: Sequence[int],
        enc: ConfigEncoding,
        I: np.ndarray,
    ) -> None:
        """Observe a new (P_low, P_high) pair and optionally refit."""
        self.X.append(self._features(plow, mem, q, r, enc, I))
        self.y.append(phigh)
        if len(self.y) >= self.min_samples:
            self._train()

    def _train(self) -> None:
        X = np.asarray(self.X, dtype=np.float64)
        y = np.asarray(self.y, dtype=np.float64)
        X_scaled = self.scaler.fit_transform(X)

        if len(X_scaled) < 5:
            X_train, X_val, y_train, y_val = X_scaled, X_scaled, y, y
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=self.val_fraction, random_state=42
            )

        if self.model is None:
            self.input_dim = X.shape[1]
            self.model = GELUSurrogateNet(self.input_dim, self.hidden_dims)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.HuberLoss(delta=self.huber_delta)

        best_val = float("inf")
        best_state = None
        patience_ctr = 0

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        for _ in range(self.max_epochs):
            self.model.train()
            optimizer.zero_grad()
            pred = self.model(X_train_t)
            loss = criterion(pred, y_train_t)

            l2 = sum(torch.norm(p, p=2) for p in self.model.parameters())
            loss = loss + self.l2_lambda * l2

            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                v_loss = criterion(self.model(X_val_t), y_val_t).item()

            if v_loss < best_val:
                best_val = v_loss
                patience_ctr = 0
                best_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.is_fitted = True

    def predict(
        self,
        plow: float,
        mem: float,
        q: Sequence[int],
        r: Sequence[int],
        enc: ConfigEncoding,
        I: np.ndarray,
    ) -> float:
        """Predict P_high from the configuration descriptors."""
        if not self.is_fitted or len(self.y) < self.min_samples:
            return plow  # identity fallback

        x = self._features(plow, mem, q, r, enc, I)
        x_scaled = self.scaler.transform([x])
        x_t = torch.tensor(x_scaled, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return float(self.model(x_t).item())
