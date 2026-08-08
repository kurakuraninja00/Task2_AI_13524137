"""
optimizers.py — Implementasi optimizer from scratch.

1. GradientDescent  — vanilla batch/mini-batch gradient descent
2. Adam             — Adaptive Moment Estimation

Interface seragam: optimizer.step(params, grads) → updated params.
"""

import numpy as np
from typing import List


class GradientDescent:
    """Vanilla (mini-batch) gradient descent."""

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def reset(self):
        """Reset state (tidak ada state untuk vanilla GD)."""
        pass

    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """Update parameters: params -= lr * grads."""
        updated = []
        for p, g in zip(params, grads):
            updated.append(p - self.lr * g)
        return updated


class Adam:
    """
    Adam optimizer (Kingma & Ba, 2014).

    Menyimpan first moment (m) dan second moment (v) per parameter.
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m: List[np.ndarray] = []
        self.v: List[np.ndarray] = []
        self._initialized = False

    def reset(self):
        """Reset moment estimates."""
        self.t = 0
        self.m = []
        self.v = []
        self._initialized = False

    def step(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """Update parameters menggunakan Adam."""
        if not self._initialized:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
            self._initialized = True

        self.t += 1
        updated = []

        for i, (p, g) in enumerate(zip(params, grads)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

            # Bias-corrected estimates
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update params
            updated.append(p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))

        return updated
