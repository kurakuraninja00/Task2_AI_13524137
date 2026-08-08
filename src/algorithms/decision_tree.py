"""
decision_tree.py — Implementasi CART Decision Tree from scratch.

Menggunakan Gini impurity sebagai split criterion.
Support class weight untuk menangani imbalanced data.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List


class CARTNode:
    """Node pada CART decision tree."""

    def __init__(self):
        self.feature_idx: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["CARTNode"] = None
        self.right: Optional["CARTNode"] = None
        self.is_leaf: bool = False
        self.prediction: Optional[int] = None
        self.class_distribution: Optional[np.ndarray] = None  # [count_0, count_1]
        self.n_samples: int = 0
        self.gini: float = 0.0
        self.depth: int = 0


class DecisionTreeCART:
    """
    CART Decision Tree Classifier — from scratch.

    Alasan pemilihan CART:
    - Mayoritas fitur bersifat numerik kontinu → CART langsung split
      berbasis threshold tanpa diskritisasi manual.
    - Binary split → tree lebih balanced dan efisien.
    - Gini impurity → komputasi lebih cepat daripada entropy (tanpa log).

    Parameters
    ----------
    max_depth : int
        Kedalaman maksimum tree.
    min_samples_split : int
        Jumlah minimum sampel untuk melakukan split.
    min_samples_leaf : int
        Jumlah minimum sampel di setiap leaf.
    class_weight : str or dict, optional
        'balanced' untuk menghitung weight otomatis, atau dict {0: w0, 1: w1}.
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        class_weight: Optional[str | Dict[int, float]] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.root: Optional[CARTNode] = None
        self.n_features_: int = 0
        self.classes_: np.ndarray = np.array([0, 1])
        self.sample_weights_: Optional[np.ndarray] = None

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

    def _weighted_gini(self, y: np.ndarray, sample_w: np.ndarray) -> float:
        """Hitung weighted Gini impurity."""
        if len(y) == 0:
            return 0.0
        total_w = sample_w.sum()
        if total_w == 0:
            return 0.0
        gini = 1.0
        for c in self.classes_:
            mask = y == c
            p = sample_w[mask].sum() / total_w
            gini -= p ** 2
        return gini

    def _best_split(
        self, X: np.ndarray, y: np.ndarray, sample_w: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Cari split terbaik (feature_idx, threshold) yang meminimalkan
        weighted Gini impurity.
        """
        n, m = X.shape
        best_gini = float("inf")
        best_feat = None
        best_thresh = None

        parent_gini = self._weighted_gini(y, sample_w)
        total_w = sample_w.sum()

        for feat in range(m):
            # Sort by feature values
            sorted_idx = np.argsort(X[:, feat])
            X_sorted = X[sorted_idx, feat]
            y_sorted = y[sorted_idx]
            w_sorted = sample_w[sorted_idx]

            # Kumulatif weighted counts per kelas untuk left split
            left_w = 0.0
            left_class_w = np.zeros(len(self.classes_))

            for i in range(n - 1):
                c_idx = int(y_sorted[i])
                left_class_w[c_idx] += w_sorted[i]
                left_w += w_sorted[i]

                # Skip jika nilai sama (tidak bisa split di sini)
                if X_sorted[i] == X_sorted[i + 1]:
                    continue

                right_w = total_w - left_w
                n_left = i + 1
                n_right = n - n_left

                # Check min_samples_leaf
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                # Gini kiri
                if left_w > 0:
                    gini_left = 1.0 - np.sum((left_class_w / left_w) ** 2)
                else:
                    gini_left = 0.0

                # Gini kanan
                right_class_w = np.zeros(len(self.classes_))
                for ci, c in enumerate(self.classes_):
                    mask_c = y_sorted[i + 1 :] == c
                    right_class_w[ci] = w_sorted[i + 1 :][mask_c].sum()

                if right_w > 0:
                    gini_right = 1.0 - np.sum((right_class_w / right_w) ** 2)
                else:
                    gini_right = 0.0

                # Weighted average Gini
                weighted_gini = (left_w * gini_left + right_w * gini_right) / total_w

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feat = feat
                    best_thresh = (X_sorted[i] + X_sorted[i + 1]) / 2.0

        return best_feat, best_thresh, best_gini

    def _build_tree(
        self, X: np.ndarray, y: np.ndarray, sample_w: np.ndarray, depth: int
    ) -> CARTNode:
        """Rekursif membangun tree."""
        node = CARTNode()
        node.depth = depth
        node.n_samples = len(y)
        node.gini = self._weighted_gini(y, sample_w)

        # Class distribution (unweighted)
        node.class_distribution = np.array(
            [np.sum(y == c) for c in self.classes_]
        )

        # Majority class (weighted)
        weighted_counts = np.array([
            sample_w[y == c].sum() for c in self.classes_
        ])
        node.prediction = int(self.classes_[np.argmax(weighted_counts)])

        # Stopping criteria
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            node.is_leaf = True
            return node

        # Cari best split
        feat_idx, threshold, split_gini = self._best_split(X, y, sample_w)

        if feat_idx is None:
            node.is_leaf = True
            return node

        # Split data
        left_mask = X[:, feat_idx] <= threshold
        right_mask = ~left_mask

        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            node.is_leaf = True
            return node

        node.feature_idx = feat_idx
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], sample_w[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], sample_w[right_mask], depth + 1)

        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeCART":
        """Train decision tree."""
        self.n_features_ = X.shape[1]
        sample_w = self._compute_sample_weights(y)
        self.root = self._build_tree(X, y, sample_w, depth=0)
        return self

    def _predict_single(self, x: np.ndarray, node: CARTNode) -> int:
        """Prediksi satu sampel."""
        if node.is_leaf:
            return node.prediction
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediksi batch."""
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_proba_single(self, x: np.ndarray, node: CARTNode) -> np.ndarray:
        """Return probabilitas kelas di leaf."""
        if node.is_leaf:
            total = node.class_distribution.sum()
            if total == 0:
                return np.array([0.5, 0.5])
            return node.class_distribution / total
        if x[node.feature_idx] <= node.threshold:
            return self._predict_proba_single(x, node.left)
        else:
            return self._predict_proba_single(x, node.right)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities. Shape (n, 2)."""
        return np.array([self._predict_proba_single(x, self.root) for x in X])

    def get_tree_structure(self) -> dict:
        """
        Return tree structure sebagai nested dict untuk visualisasi.
        """
        def _to_dict(node: CARTNode) -> dict:
            d = {
                "n_samples": node.n_samples,
                "gini": round(node.gini, 4),
                "class_dist": node.class_distribution.tolist() if node.class_distribution is not None else [],
                "prediction": node.prediction,
                "depth": node.depth,
            }
            if not node.is_leaf:
                d["feature_idx"] = node.feature_idx
                d["threshold"] = round(node.threshold, 4)
                d["left"] = _to_dict(node.left)
                d["right"] = _to_dict(node.right)
            else:
                d["is_leaf"] = True
            return d

        return _to_dict(self.root)
