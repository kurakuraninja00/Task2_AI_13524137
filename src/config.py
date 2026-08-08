"""
config.py — Konfigurasi sentral project Kaggle Loan Approval Classification.

Berisi path data, random seed, daftar kolom, bound cleaning, dan hyperparameter.
"""

import os

# ============================================================================
# Paths
# ============================================================================
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

TRAIN_PATH = os.path.join(DATA_DIR, "train (1).csv")
TEST_PATH = os.path.join(DATA_DIR, "test (1).csv")
SAMPLE_SUBMISSION_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

# Artifacts
ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "src", "artifacts")
MODELS_DIR = os.path.join(ARTIFACTS_DIR, "models")
FIGURES_DIR = os.path.join(ARTIFACTS_DIR, "figures")
SUBMISSIONS_DIR = os.path.join(ARTIFACTS_DIR, "submissions")

for _d in [ARTIFACTS_DIR, MODELS_DIR, FIGURES_DIR, SUBMISSIONS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ============================================================================
# Random seed
# ============================================================================
SEED = 42

# ============================================================================
# Kolom
# ============================================================================
ID_COL = "person_id"
TARGET_COL = "loan_status"

NUMERICAL_COLS = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]

CATEGORICAL_COLS = [
    "person_gender",
    "person_home_ownership",
    "previous_loan_defaults_on_file",
]

FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

# ============================================================================
# Cleaning bounds (fit dari train, lihat cleaning.py)
# ============================================================================
AGE_CAP = 90.0
EMP_EXP_CAP = 50
INCOME_LOWER_PERCENTILE = 1   # persentil ke-1
INCOME_UPPER_PERCENTILE = 99  # persentil ke-99

# ============================================================================
# Hyperparameter defaults
# ============================================================================

# Decision Tree (CART)
DTL_MAX_DEPTH = 10
DTL_MIN_SAMPLES_SPLIT = 10
DTL_MIN_SAMPLES_LEAF = 5

# Logistic Regression
LR_LEARNING_RATE = 0.01
LR_MAX_ITER = 1000
LR_BATCH_SIZE = 256
LR_LAMBDA_REG = 0.01  # L2 regularization

# SVM (linear soft-margin, primal)
SVM_LEARNING_RATE = 0.001
SVM_MAX_ITER = 1000
SVM_BATCH_SIZE = 256
SVM_C = 1.0  # regularization trade-off (inverse of lambda)

# Cross-validation
CV_N_FOLDS = 5
