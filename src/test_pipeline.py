"""
Quick end-to-end test of the from-scratch ML pipeline.
Run from project root: python -m src.test_pipeline
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.utils import set_seed
from src.data import load_train, load_test
from src.cleaning import DataCleaner
from src.preprocessing import Preprocessor
from src.optimizers import GradientDescent, Adam
from src.algorithms.decision_tree import DecisionTreeCART
from src.algorithms.logistic_regression import LogisticRegressionScratch
from src.algorithms.svm import LinearSVMScratch
from src.evaluation import macro_f1_score, cross_validate, stratified_k_fold_indices

set_seed(42)

print("=== Loading data ===")
train_df = load_train()
test_df = load_test()
print(f"  Train: {train_df.shape}, Test: {test_df.shape}")

print("\n=== Cleaning ===")
cleaner = DataCleaner()
train_clean = cleaner.fit_transform(train_df)
test_clean = cleaner.transform(test_df)
print(f"  person_age max (train): {train_clean['person_age'].max()}")
print(f"  person_emp_exp max (train): {train_clean['person_emp_exp'].max()}")

print("\n=== Preprocessing ===")
preprocessor = Preprocessor()
X_train, y_train = preprocessor.fit_transform(train_clean)
X_test = preprocessor.transform(test_clean)
print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  Features: {preprocessor.feature_names_[:5]}...")

# Quick single-fold test
print("\n=== Single split test ===")
folds = stratified_k_fold_indices(y_train, n_folds=5)
tr_idx, val_idx = folds[0]
X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
X_val, y_val = X_train[val_idx], y_train[val_idx]

# DTL
print("\n--- Decision Tree (CART) ---")
dtl = DecisionTreeCART(max_depth=8, min_samples_split=20, min_samples_leaf=10, class_weight="balanced")
dtl.fit(X_tr, y_tr)
y_pred_dtl = dtl.predict(X_val)
f1_dtl = macro_f1_score(y_val, y_pred_dtl)
print(f"  Macro F1: {f1_dtl:.4f}")
print(f"  Pred distribution: 0={np.sum(y_pred_dtl==0)}, 1={np.sum(y_pred_dtl==1)}")

# LR
print("\n--- Logistic Regression ---")
lr_model = LogisticRegressionScratch(
    max_iter=300, batch_size=512, lambda_reg=0.01,
    class_weight="balanced", optimizer=Adam(lr=0.01)
)
lr_model.fit(X_tr, y_tr)
y_pred_lr = lr_model.predict(X_val)
f1_lr = macro_f1_score(y_val, y_pred_lr)
print(f"  Macro F1: {f1_lr:.4f}")
print(f"  Pred distribution: 0={np.sum(y_pred_lr==0)}, 1={np.sum(y_pred_lr==1)}")
print(f"  Loss history length: {len(lr_model.loss_history)}")

# SVM
print("\n--- SVM (linear) ---")
svm_model = LinearSVMScratch(
    C=1.0, max_iter=300, batch_size=512,
    class_weight="balanced", optimizer=Adam(lr=0.001)
)
svm_model.fit(X_tr, y_tr)
y_pred_svm = svm_model.predict(X_val)
f1_svm = macro_f1_score(y_val, y_pred_svm)
print(f"  Macro F1: {f1_svm:.4f}")
print(f"  Pred distribution: 0={np.sum(y_pred_svm==0)}, 1={np.sum(y_pred_svm==1)}")

print("\n=== ALL TESTS PASSED ===")
print(f"  DTL: {f1_dtl:.4f}  |  LR: {f1_lr:.4f}  |  SVM: {f1_svm:.4f}")
