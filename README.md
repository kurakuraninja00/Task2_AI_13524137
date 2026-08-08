# Task 2 — Seleksi Lab AI (13524137)

## Kaggle: Loan Approval Classification (From-Scratch)

Implementasi tiga algoritma machine learning **from scratch** (tanpa library ML) untuk memprediksi persetujuan pinjaman.

### Algoritma yang Diimplementasikan

1. **CART Decision Tree** (`src/algorithms/decision_tree.py`)
   - Varian: **CART** — dipilih karena mayoritas fitur numerik kontinu sehingga binary split berbasis threshold langsung applicable tanpa diskritisasi manual (berbeda dari ID3). Gini impurity dipilih karena komputasi lebih cepat daripada entropy.
   - Fitur: weighted Gini impurity, class weight balanced, max_depth/min_samples stopping criteria.

2. **Logistic Regression** (`src/algorithms/logistic_regression.py`)
   - Sigmoid, binary cross-entropy + L2 regularization, gradient manual.
   - Threshold tuning untuk macro F1 optimization.

3. **Linear SVM** (`src/algorithms/svm.py`)
   - Formulasi primal, hinge loss + L2 regularization, subgradient descent.
   - Formulasi primal dipilih karena dual membutuhkan matriks kernel n×n (~28.8k²) yang tidak feasible.

### Optimizer (`src/optimizers.py`)

- **Gradient Descent**: Vanilla mini-batch GD.
- **Adam** (Kingma & Ba, 2014): Adaptive moment estimation dengan bias correction.

### Hasil (5-Fold Stratified CV)

| Model | From-Scratch | Sklearn Baseline |
|-------|-------------|-----------------|
| CART Decision Tree | **0.8485** ± 0.0082 | 0.8483 ± 0.0083 |
| Logistic Regression | **0.8433** ± 0.0050 | 0.8170 ± 0.0036 |
| Linear SVM | **0.8223** ± 0.0067 | 0.8146 ± 0.0030 |

**Model final**: CART Decision Tree (macro F1 tertinggi).

### Struktur Project

```
src/
├── config.py               # path, seed, hyperparameter
├── data.py                 # load & validasi data
├── cleaning.py             # capping outlier (fit dari train)
├── preprocessing.py        # one-hot encoding & standardization manual
├── optimizers.py            # GD & Adam from scratch
├── algorithms/
│   ├── decision_tree.py    # CART from scratch
│   ├── logistic_regression.py  # LR from scratch
│   └── svm.py              # Linear SVM from scratch
├── evaluation.py           # stratified K-fold CV, macro F1, threshold tuning
├── sklearn_baselines.py    # baseline sklearn (pembanding SAJA)
├── visualization.py        # tree plot, loss contour, convergence
├── predict.py              # generate submission (from-scratch only)
└── artifacts/
    ├── models/             # saved model artifacts
    ├── figures/            # tree visualization, loss plots
    └── submissions/        # submission.csv

notebooks/
├── 01_eda.ipynb
├── 02_dtl_scratch_vs_sklearn.ipynb
├── 03_logistic_regression_scratch_vs_sklearn.ipynb
├── 04_svm_scratch_vs_sklearn.ipynb
└── 05_final_submission.ipynb
```

### Cara Menjalankan

```bash
pip install -r requirements.txt
```

Jalankan notebook secara berurutan:
1. `notebooks/01_eda.ipynb` — EDA & data exploration
2. `notebooks/02_dtl_scratch_vs_sklearn.ipynb` — CART Decision Tree
3. `notebooks/03_logistic_regression_scratch_vs_sklearn.ipynb` — Logistic Regression
4. `notebooks/04_svm_scratch_vs_sklearn.ipynb` — SVM
5. `notebooks/05_final_submission.ipynb` — Perbandingan & submission

Atau jalankan pipeline lengkap sekaligus:
```bash
python src/run_full_cv.py
```

### Catatan Penting

- Semua implementasi inti **hanya menggunakan numpy** (+ pandas untuk data handling).
- scikit-learn hanya dipakai sebagai **baseline pembanding** dan cross-check metric.
- Submission wajib dari model **from-scratch** (bukan sklearn).
- Semua bound preprocessing di-fit dari train saja (no data leakage).