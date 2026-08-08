"""
logistic_regression.py — Logistic Regression from scratch.

Forward pass (sigmoid), binary cross-entropy loss, gradient manual.
Mendukung class weight dan L2 regularization.
Menggunakan optimizer dari src/optimizers.py.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple


class LogisticRegressionScratch:
    """
    Logistic Regression classifier — from scratch.

    Parameters
    ----------
    max_iter : int
        Jumlah maksimum iterasi training.
    batch_size : int or None
        Ukuran mini-batch. None = full batch.
    lambda_reg : float
        Koefisien L2 regularization.
    class_weight : str or dict, optional
        'balanced' untuk weight otomatis, atau dict {0: w0, 1: w1}.
    optimizer : object
        Optimizer dari src/optimizers.py (harus punya method step()).
    threshold : float
        Threshold untuk klasifikasi (default 0.5).
    """

    def __init__(
        self,
        max_iter: int = 1000,
        batch_size: Optional[int] = 256,
        lambda_reg: float = 0.01,
        class_weight: Optional[str | Dict[int, float]] = None,
        optimizer=None,
        threshold: float = 0.5,
    ):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.class_weight = class_weight
        self.optimizer = optimizer
        self.threshold = threshold

        self.w: Optional[np.ndarray] = None  # shape (n_features,)
        self.b: float = 0.0
        self.loss_history: List[float] = []
        self.w_history: List[np.ndarray] = []  # untuk visualisasi trajectory

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Hitung sample weights berdasarkan class_weight."""
        n = len(y)
        if self.class_weight is None:
            return np.ones(n)
        if self.class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            n_classes = len(classes)
            weights_map = {}
            for c, cnt in zip(classes, counts):
                weights_map[c] = n / (n_classes * cnt)
        else:
            weights_map = self.class_weight
        return np.array([weights_map[int(yi)] for yi in y])

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Compute sigmoid(X @ w + b)."""
        z = X @ self.w + self.b
        return self._sigmoid(z)

    def _loss(self, X: np.ndarray, y: np.ndarray, sample_w: np.ndarray) -> float:
        """Weighted binary cross-entropy + L2 regularization."""
        eps = 1e-15
        y_hat = self._forward(X)
        y_hat = np.clip(y_hat, eps, 1 - eps)
        bce = -sample_w * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        loss = bce.mean() + 0.5 * self.lambda_reg * np.sum(self.w ** 2)
        return loss

    def _gradients(
        self, X: np.ndarray, y: np.ndarray, sample_w: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Hitung gradien w dan b."""
        n = len(y)
        y_hat = self._forward(X)
        error = (y_hat - y) * sample_w  # weighted error

        dw = (X.T @ error) / n + self.lambda_reg * self.w
        db = error.mean()

        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        """Train logistic regression."""
        n, m = X.shape
        self.w = np.zeros(m)
        self.b = 0.0
        self.loss_history = []
        self.w_history = []

        sample_w = self._compute_sample_weights(y)

        if self.optimizer is not None:
            self.optimizer.reset()

        for epoch in range(self.max_iter):
            if self.batch_size is None or self.batch_size >= n:
                # Full batch
                X_batch, y_batch, sw_batch = X, y, sample_w
            else:
                # Mini-batch
                idx = np.random.choice(n, self.batch_size, replace=False)
                X_batch, y_batch, sw_batch = X[idx], y[idx], sample_w[idx]

            dw, db = self._gradients(X_batch, y_batch, sw_batch)

            if self.optimizer is not None:
                updated = self.optimizer.step(
                    [self.w, np.array([self.b])],
                    [dw, np.array([db])],
                )
                self.w = updated[0]
                self.b = updated[1][0]
            else:
                # Default: simple GD with lr=0.01
                self.w -= 0.01 * dw
                self.b -= 0.01 * db

            # Record loss (on full data, setiap 10 iterasi untuk efisiensi)
            if epoch % 10 == 0 or epoch == self.max_iter - 1:
                loss = self._loss(X, y, sample_w)
                self.loss_history.append(loss)
                self.w_history.append(self.w.copy())

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probabilitas kelas 1."""
        return self._forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediksi kelas berdasarkan threshold."""
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw score (sebelum sigmoid)."""
        return X @ self.w + self.b
