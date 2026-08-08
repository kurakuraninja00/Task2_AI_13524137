"""
data.py — Load dan validasi data train/test/sample_submission.
"""

import pandas as pd
from . import config


def load_train() -> pd.DataFrame:
    """Load train.csv, validasi shape & integritas."""
    df = pd.read_csv(config.TRAIN_PATH)
    assert df.shape[0] == 28800, f"Expected 28800 rows, got {df.shape[0]}"
    assert config.TARGET_COL in df.columns, "Target column missing"
    assert df.isnull().sum().sum() == 0, "Missing values detected"
    assert df[config.ID_COL].nunique() == len(df), "Duplicate person_id"
    return df


def load_test() -> pd.DataFrame:
    """Load test.csv, validasi shape & integritas."""
    df = pd.read_csv(config.TEST_PATH)
    assert df.shape[0] == 7200, f"Expected 7200 rows, got {df.shape[0]}"
    assert config.TARGET_COL not in df.columns, "Target column should not be in test"
    assert df.isnull().sum().sum() == 0, "Missing values detected"
    assert df[config.ID_COL].nunique() == len(df), "Duplicate person_id"
    return df


def load_sample_submission() -> pd.DataFrame:
    """Load sample_submission.csv."""
    df = pd.read_csv(config.SAMPLE_SUBMISSION_PATH)
    assert df.shape[0] == 7200, f"Expected 7200 rows, got {df.shape[0]}"
    return df
