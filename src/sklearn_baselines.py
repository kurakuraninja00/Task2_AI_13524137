"""
sklearn_baselines.py — Wrapper sklearn untuk DTL, LR, SVM sebagai baseline pembanding SAJA.

Model dari file ini TIDAK BOLEH dipakai untuk menghasilkan submission.
Hanya untuk cross-check perbandingan macro F1 dengan implementasi from-scratch.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from . import config


def get_sklearn_dtl(max_depth: int = None, class_weight: str = "balanced"):
    """Baseline DecisionTreeClassifier sklearn."""
    if max_depth is None:
        max_depth = config.DTL_MAX_DEPTH
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_split=config.DTL_MIN_SAMPLES_SPLIT,
        min_samples_leaf=config.DTL_MIN_SAMPLES_LEAF,
        class_weight=class_weight,
        random_state=config.SEED,
    )


def get_sklearn_lr(max_iter: int = None, class_weight: str = "balanced"):
    """Baseline LogisticRegression sklearn."""
    if max_iter is None:
        max_iter = config.LR_MAX_ITER
    return LogisticRegression(
        max_iter=max_iter,
        C=1.0 / config.LR_LAMBDA_REG if config.LR_LAMBDA_REG > 0 else 1.0,
        class_weight=class_weight,
        solver="lbfgs",
        random_state=config.SEED,
    )


def get_sklearn_svm(class_weight: str = "balanced"):
    """Baseline LinearSVC sklearn."""
    return LinearSVC(
        C=config.SVM_C,
        class_weight=class_weight,
        max_iter=5000,
        random_state=config.SEED,
        dual="auto",
    )
