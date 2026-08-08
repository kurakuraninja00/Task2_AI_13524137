"""
svm.py — Linear Soft-Margin SVM from scratch (primal formulation).

Hinge loss + L2 regularization, dioptimasi lewat (mini-batch) gradient descent.
Formulasi primal dipilih karena tractable dengan numpy murni untuk ~28.8k samples
(formulasi dual memerlukan matriks kernel n×n yang terlalu besar).
"""

import numpy as np
from typing import Optional, Dict, List, Tuple


class LinearSVMScratch:
    """
    Linear Soft-Margin SVM — from scratch (primal).

    Minimisasi:
        L(w,b) = (1/2) ||w||^2 + C * (1/n) Σ max(0, 1 - y_i * (w·x_i + b)) * sample_weight_i

    Di mana label y ∈ {-1, +1} (dikonversi internal dari 0/1).

    Parameters
    ----------
    C : float
        Regularization parameter (trade-off margin vs misclassification).
    max_iter : int
        Jumlah iterasi training.
    batch_size : int or None
        Ukuran mini-batch. None = full batch.
    class_weight : str or dict, optional
        'balanced' untuk weight otomatis, atau dict {0: w0, 1: w1}.
    optimizer : object
        Optimizer dari src/optimizers.py.
    threshold : float
        Threshold pada decision function (default 0.0).
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        batch_size: Optional[int] = 256,
        class_weight: Optional[str | Dict[int, float]] = None,
        optimizer=None,
        threshold: float = 0.0,
    ):
        self.C = C
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.optimizer = optimizer
        self.threshold = threshold

        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0
        self.loss_history: List[float] = []

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Hitung sample weights. y sudah dalam {0, 1}."""
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

    @staticmethod
    def _to_svm_labels(y: np.ndarray) -> np.ndarray:
        """Konversi label 0/1 → -1/+1."""
        return 2 * y - 1

    def _hinge_loss(
        self, X: np.ndarray, y_svm: np.ndarray, sample_w: np.ndarray
    ) -> float:
        """Hitung total loss = 0.5 ||w||^2 + C * mean(weighted hinge loss)."""
        margins = y_svm * (X @ self.w + self.b)
        hinge = np.maximum(0, 1 - margins)
        loss = 0.5 * np.sum(self.w ** 2) + self.C * np.mean(sample_w * hinge)
        return loss

    def _gradients(
        self, X: np.ndarray, y_svm: np.ndarray, sample_w: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Subgradient dari hinge loss.

        dL/dw = w - C * (1/n) Σ y_i * x_i * sample_w_i   (jika margin < 1)
        dL/db =   - C * (1/n) Σ y_i * sample_w_i          (jika margin < 1)
        """
        n = len(y_svm)
        margins = y_svm * (X @ self.w + self.b)
        violated = margins < 1  # hinge active

        dw = self.w.copy()  # regularization gradient
        db = 0.0

        if violated.any():
            violated_mask = violated.astype(float)
            weighted = violated_mask * sample_w * y_svm
            dw -= self.C * (X.T @ weighted) / n
            db -= self.C * weighted.sum() / n

        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVMScratch":
        """Train linear SVM."""
        n, m = X.shape
        self.w = np.zeros(m)
        self.b = 0.0
        self.loss_history = []

        # Compute sample weights from original 0/1 labels
        sample_w = self._compute_sample_weights(y)
        y_svm = self._to_svm_labels(y)

        if self.optimizer is not None:
            self.optimizer.reset()

        for epoch in range(self.max_iter):
            if self.batch_size is None or self.batch_size >= n:
                X_batch, y_batch, sw_batch = X, y_svm, sample_w
            else:
                idx = np.random.choice(n, self.batch_size, replace=False)
                X_batch, y_batch, sw_batch = X[idx], y_svm[idx], sample_w[idx]

            dw, db = self._gradients(X_batch, y_batch, sw_batch)

            if self.optimizer is not None:
                updated = self.optimizer.step(
                    [self.w, np.array([self.b])],
                    [dw, np.array([db])],
                )
                self.w = updated[0]
                self.b = updated[1][0]
            else:
                self.w -= 0.001 * dw
                self.b -= 0.001 * db

            # Record loss setiap 10 iterasi
            if epoch % 10 == 0 or epoch == self.max_iter - 1:
                loss = self._hinge_loss(X, y_svm, sample_w)
                self.loss_history.append(loss)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return raw score w·x + b."""
        return X @ self.w + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediksi kelas 0/1 berdasarkan threshold pada decision function."""
        scores = self.decision_function(X)
        return (scores >= self.threshold).astype(int)
