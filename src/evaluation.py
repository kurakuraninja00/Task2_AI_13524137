"""
evaluation.py — Cross-validation manual, metrics, threshold tuning.

Stratified K-Fold CV, macro F1, classification report.
Boleh pakai sklearn.metrics untuk cross-check.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable
from . import config


def stratified_k_fold_indices(
    y: np.ndarray, n_folds: int = 5, seed: int = None
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate stratified K-fold train/val indices.

    Setiap fold memiliki distribusi kelas yang proporsional.
    """
    if seed is None:
        seed = config.SEED
    rng = np.random.RandomState(seed)

    classes = np.unique(y)
    class_indices = {}
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        class_indices[c] = idx

    # Bagi tiap kelas ke n_folds
    folds = [[] for _ in range(n_folds)]
    for c in classes:
        idx = class_indices[c]
        fold_sizes = np.full(n_folds, len(idx) // n_folds, dtype=int)
        fold_sizes[: len(idx) % n_folds] += 1
        current = 0
        for i in range(n_folds):
            folds[i].extend(idx[current : current + fold_sizes[i]])
            current += fold_sizes[i]

    # Generate train/val pairs
    result = []
    for i in range(n_folds):
        val_idx = np.array(folds[i])
        train_idx = np.concatenate([np.array(folds[j]) for j in range(n_folds) if j != i])
        result.append((train_idx, val_idx))

    return result


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> Dict[str, float]:
    """Hitung precision, recall, F1 untuk satu kelas."""
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Hitung macro F1 (rata-rata F1 kelas 0 dan kelas 1)."""
    f1_0 = precision_recall_f1(y_true, y_pred, pos_label=0)["f1"]
    f1_1 = precision_recall_f1(y_true, y_pred, pos_label=1)["f1"]
    return (f1_0 + f1_1) / 2.0


def classification_report_manual(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Classification report per kelas + macro F1."""
    report = {}
    for c in [0, 1]:
        metrics = precision_recall_f1(y_true, y_pred, pos_label=c)
        support = np.sum(y_true == c)
        report[f"class_{c}"] = {**metrics, "support": int(support)}
    report["macro_f1"] = macro_f1_score(y_true, y_pred)
    report["accuracy"] = np.mean(y_true == y_pred)
    return report


def find_best_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray = None,
) -> Tuple[float, float]:
    """
    Cari threshold yang memaksimalkan macro F1.

    Parameters
    ----------
    y_true : array of 0/1
    scores : raw scores atau probabilitas
    thresholds : array of candidate thresholds (default linspace)

    Returns
    -------
    best_threshold, best_macro_f1
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 201)

    best_f1 = -1.0
    best_t = 0.5

    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        f1 = macro_f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


def cross_validate(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    tune_threshold: bool = False,
    use_decision_function: bool = False,
) -> Dict:
    """
    Stratified K-Fold Cross-Validation.

    Parameters
    ----------
    model_factory : callable
        Function yang mengembalikan model baru (fresh instance).
    X, y : data
    n_folds : jumlah fold
    tune_threshold : jika True, cari threshold optimal per fold
    use_decision_function : jika True, gunakan decision_function() bukan predict_proba()

    Returns
    -------
    dict dengan keys: fold_scores, mean_f1, std_f1, per_fold_details
    """
    folds = stratified_k_fold_indices(y, n_folds)
    fold_scores = []
    fold_details = []

    for i, (train_idx, val_idx) in enumerate(folds):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        model = model_factory()
        model.fit(X_train, y_train)

        if tune_threshold:
            if use_decision_function:
                scores = model.decision_function(X_val)
                thresholds = np.linspace(scores.min(), scores.max(), 201)
            else:
                scores = model.predict_proba(X_val)
                if scores.ndim == 2:
                    scores = scores[:, 1]
                thresholds = np.linspace(0.0, 1.0, 201)

            best_t, best_f1 = find_best_threshold(y_val, scores, thresholds)
            y_pred = (scores >= best_t).astype(int)
            f1 = best_f1
        else:
            y_pred = model.predict(X_val)
            f1 = macro_f1_score(y_val, y_pred)
            best_t = None

        report = classification_report_manual(y_val, y_pred)
        fold_scores.append(f1)
        fold_details.append({
            "fold": i,
            "macro_f1": f1,
            "threshold": best_t,
            "report": report,
        })

    return {
        "fold_scores": fold_scores,
        "mean_f1": np.mean(fold_scores),
        "std_f1": np.std(fold_scores),
        "per_fold_details": fold_details,
    }
