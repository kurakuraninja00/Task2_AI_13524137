"""
utils.py — Utilitas umum: set seed, save/load model artifacts.
"""

import os
import random
import pickle
import numpy as np

from . import config


def set_seed(seed: int = None) -> None:
    """Set random seed untuk reproducibility."""
    if seed is None:
        seed = config.SEED
    random.seed(seed)
    np.random.seed(seed)


def save_model(model: object, filename: str) -> str:
    """Simpan model ke pickle di MODELS_DIR. Return path."""
    path = os.path.join(config.MODELS_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(filename: str) -> object:
    """Load model dari MODELS_DIR."""
    path = os.path.join(config.MODELS_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)
