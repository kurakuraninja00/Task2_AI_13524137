"""
Full cross-validation + submission generation script.
Runs all 3 from-scratch models + sklearn baselines.
Outputs results for the write-up.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from src.utils import set_seed
from src.data import load_train, load_test, load_sample_submission
from src.cleaning import DataCleaner
from src.preprocessing import Preprocessor
from src.optimizers import GradientDescent, Adam
from src.algorithms.decision_tree import DecisionTreeCART
from src.algorithms.logistic_regression import LogisticRegressionScratch
from src.algorithms.svm import LinearSVMScratch
from src.evaluation import cross_validate, macro_f1_score, find_best_threshold, stratified_k_fold_indices
from src.sklearn_baselines import get_sklearn_dtl, get_sklearn_lr, get_sklearn_svm
from src.predict import generate_submission
from src import config

set_seed(42)

# ============================================================
# Load & preprocess
# ============================================================
print("=" * 60)
print("LOADING & PREPROCESSING")
print("=" * 60)
train_df = load_train()
test_df = load_test()

cleaner = DataCleaner()
train_clean = cleaner.fit_transform(train_df)
test_clean = cleaner.transform(test_df)

preprocessor = Preprocessor()
X_train, y_train = preprocessor.fit_transform(train_clean)
X_test = preprocessor.transform(test_clean)
test_ids = test_df[config.ID_COL].values

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"Features: {preprocessor.feature_names_}")

all_results = {}

# ============================================================
# 1. DECISION TREE (CART) — From Scratch
# ============================================================
print("\n" + "=" * 60)
print("1. DECISION TREE (CART) — FROM SCRATCH")
print("=" * 60)

def dtl_factory():
    return DecisionTreeCART(
        max_depth=10, min_samples_split=10, min_samples_leaf=5,
        class_weight="balanced"
    )

dtl_cv = cross_validate(dtl_factory, X_train, y_train, n_folds=5)
print(f"  Fold scores: {[f'{s:.4f}' for s in dtl_cv['fold_scores']]}")
print(f"  Mean Macro F1: {dtl_cv['mean_f1']:.4f} ± {dtl_cv['std_f1']:.4f}")
all_results["DTL"] = {"scratch": dtl_cv['mean_f1']}

# DTL sklearn baseline
print("\n  --- DTL Sklearn Baseline ---")
def dtl_sklearn_factory():
    return get_sklearn_dtl(max_depth=10)

dtl_sk_cv = cross_validate(dtl_sklearn_factory, X_train, y_train, n_folds=5)
print(f"  Sklearn Mean Macro F1: {dtl_sk_cv['mean_f1']:.4f} ± {dtl_sk_cv['std_f1']:.4f}")
all_results["DTL"]["sklearn"] = dtl_sk_cv['mean_f1']

# ============================================================
# 2. LOGISTIC REGRESSION — From Scratch
# ============================================================
print("\n" + "=" * 60)
print("2. LOGISTIC REGRESSION — FROM SCRATCH")
print("=" * 60)

# With Adam optimizer
def lr_factory():
    return LogisticRegressionScratch(
        max_iter=500, batch_size=512, lambda_reg=0.01,
        class_weight="balanced", optimizer=Adam(lr=0.01)
    )

lr_cv = cross_validate(lr_factory, X_train, y_train, n_folds=5, tune_threshold=True)
print(f"  Fold scores: {[f'{s:.4f}' for s in lr_cv['fold_scores']]}")
print(f"  Mean Macro F1: {lr_cv['mean_f1']:.4f} ± {lr_cv['std_f1']:.4f}")
all_results["LR"] = {"scratch": lr_cv['mean_f1']}

# LR sklearn baseline
print("\n  --- LR Sklearn Baseline ---")
def lr_sklearn_factory():
    return get_sklearn_lr()

lr_sk_cv = cross_validate(lr_sklearn_factory, X_train, y_train, n_folds=5)
print(f"  Sklearn Mean Macro F1: {lr_sk_cv['mean_f1']:.4f} ± {lr_sk_cv['std_f1']:.4f}")
all_results["LR"]["sklearn"] = lr_sk_cv['mean_f1']

# ============================================================
# 3. SVM — From Scratch
# ============================================================
print("\n" + "=" * 60)
print("3. SVM (LINEAR) — FROM SCRATCH")
print("=" * 60)

def svm_factory():
    return LinearSVMScratch(
        C=1.0, max_iter=500, batch_size=512,
        class_weight="balanced", optimizer=Adam(lr=0.001)
    )

svm_cv = cross_validate(
    svm_factory, X_train, y_train, n_folds=5,
    tune_threshold=True, use_decision_function=True
)
print(f"  Fold scores: {[f'{s:.4f}' for s in svm_cv['fold_scores']]}")
print(f"  Mean Macro F1: {svm_cv['mean_f1']:.4f} ± {svm_cv['std_f1']:.4f}")
all_results["SVM"] = {"scratch": svm_cv['mean_f1']}

# SVM sklearn baseline
print("\n  --- SVM Sklearn Baseline ---")
def svm_sklearn_factory():
    return get_sklearn_svm()

svm_sk_cv = cross_validate(svm_sklearn_factory, X_train, y_train, n_folds=5)
print(f"  Sklearn Mean Macro F1: {svm_sk_cv['mean_f1']:.4f} ± {svm_sk_cv['std_f1']:.4f}")
all_results["SVM"]["sklearn"] = svm_sk_cv['mean_f1']

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY — Model Comparison")
print("=" * 60)
print(f"{'Model':<20} {'From-Scratch':<18} {'Sklearn':<18}")
print("-" * 56)
for m in all_results:
    s = all_results[m].get('scratch', '-')
    sk = all_results[m].get('sklearn', '-')
    s_str = f"{s:.4f}" if isinstance(s, float) else s
    sk_str = f"{sk:.4f}" if isinstance(sk, float) else sk
    print(f"{m:<20} {s_str:<18} {sk_str:<18}")

# ============================================================
# SELECT BEST & GENERATE SUBMISSION
# ============================================================
print("\n" + "=" * 60)
print("GENERATING SUBMISSION")
print("=" * 60)

# Find best from-scratch model
best_model_name = max(all_results, key=lambda k: all_results[k]['scratch'])
print(f"Best from-scratch model: {best_model_name} (F1={all_results[best_model_name]['scratch']:.4f})")

# Train on full training data
if best_model_name == "DTL":
    final_model = dtl_factory()
    final_model.fit(X_train, y_train)
    y_test_pred = final_model.predict(X_test)
elif best_model_name == "LR":
    final_model = lr_factory()
    final_model.fit(X_train, y_train)
    # Find best threshold on training data (using last fold's threshold as proxy)
    scores_train = final_model.predict_proba(X_train)
    best_t, _ = find_best_threshold(y_train, scores_train)
    print(f"  Best threshold (LR): {best_t:.4f}")
    y_test_pred = (final_model.predict_proba(X_test) >= best_t).astype(int)
else:  # SVM
    final_model = svm_factory()
    final_model.fit(X_train, y_train)
    scores_train = final_model.decision_function(X_train)
    t_range = np.linspace(scores_train.min(), scores_train.max(), 201)
    best_t, _ = find_best_threshold(y_train, scores_train, t_range)
    print(f"  Best threshold (SVM): {best_t:.4f}")
    y_test_pred = (final_model.decision_function(X_test) >= best_t).astype(int)

# Generate submission
sub_path = generate_submission(
    final_model, X_test, test_ids,
    filename="submission.csv"
)

# Save results to JSON for write-up
results_path = os.path.join(config.ARTIFACTS_DIR, "cv_results.json")
# Convert numpy types for JSON
serializable = {}
for k, v in all_results.items():
    serializable[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv for kk, vv in v.items()}
serializable["best_model"] = best_model_name
serializable["dtl_cv_folds"] = [float(x) for x in dtl_cv['fold_scores']]
serializable["lr_cv_folds"] = [float(x) for x in lr_cv['fold_scores']]
serializable["svm_cv_folds"] = [float(x) for x in svm_cv['fold_scores']]

with open(results_path, "w") as f:
    json.dump(serializable, f, indent=2)
print(f"\nResults saved to {results_path}")

print("\n=== DONE ===")
