"""
predict.py — Generate submission file dari model from-scratch.

HANYA model from-scratch yang boleh dipakai di sini.
Model sklearn TIDAK BOLEH digunakan untuk submission.
"""

import os
import numpy as np
import pandas as pd
from . import config


def generate_submission(
    model,
    X_test: np.ndarray,
    test_ids: np.ndarray,
    filename: str = "submission.csv",
    threshold: float = None,
    use_decision_function: bool = False,
) -> str:
    """
    Generate file submission dari model from-scratch.

    Parameters
    ----------
    model : model from-scratch (harus punya predict() atau decision_function())
    X_test : fitur test yang sudah dipreprocess
    test_ids : array person_id dari test
    filename : nama file output
    threshold : threshold kustom (jika None, pakai model.predict())
    use_decision_function : gunakan decision_function() + threshold

    Returns
    -------
    path ke file submission
    """
    if threshold is not None:
        if use_decision_function and hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
        elif hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_test)
            if scores.ndim == 2:
                scores = scores[:, 1]
        else:
            scores = model.decision_function(X_test)
        y_pred = (scores >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    # Buat DataFrame submission
    submission = pd.DataFrame({
        config.ID_COL: test_ids.astype(int),
        config.TARGET_COL: y_pred.astype(int),
    })

    # Validasi
    assert len(submission) == 7200, f"Expected 7200 rows, got {len(submission)}"

    # Simpan
    path = os.path.join(config.SUBMISSIONS_DIR, filename)
    submission.to_csv(path, index=False)
    print(f"Submission saved: {path}")
    print(f"  Shape: {submission.shape}")
    print(f"  Prediction distribution: {submission[config.TARGET_COL].value_counts().to_dict()}")

    return path
