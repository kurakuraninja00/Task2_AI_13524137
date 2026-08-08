# Prompt: Bangun Solusi Kaggle Competition — Loan Approval Classification (Task #2 Seleksi Lab AI)
### (Implementasi From-Scratch)

Kamu bertindak sebagai Data Scientist / ML Engineer. Tugasmu adalah membangun **project Python yang lengkap dan rapi** untuk menyelesaikan kompetisi Kaggle di bawah ini, dengan struktur kode modular dan reproducible, di mana **seluruh algoritma machine learning diimplementasikan from scratch**.

## 1. Konteks Kompetisi & Batasan Algoritma

- Tugas: memprediksi `loan_status` (0 = ditolak, 1 = disetujui) untuk setiap pemohon pinjaman pada data uji.
- Metric evaluasi: **macro F1-score** (rata-rata F1 dari kelas 0 dan kelas 1, tanpa memperhitungkan proporsi jumlah data tiap kelas) — penting karena distribusi kelas tidak seimbang.

### Batasan model — WAJIB DIPATUHI

Ketiga algoritma berikut **wajib diimplementasikan from scratch** (tanpa memakai kelas/fungsi model siap pakai dari scikit-learn atau library ML lain):

1. **Decision Tree Learning** — pilih **salah satu** dari **ID3**, **C4.5**, atau **CART**, dan jelaskan alasan pemilihannya di README/notebook.
   - **Rekomendasi**: **CART** paling praktis untuk dataset ini karena mayoritas fitur bersifat numerik kontinu (`person_age`, `person_income`, `loan_amnt`, `loan_int_rate`, `credit_score`, dst.) sehingga bisa langsung di-split berbasis threshold tanpa diskritisasi manual. Jika memilih **ID3**, fitur numerik **wajib** didiskritisasi terlebih dahulu (dokumentasikan strategi binning-nya). **C4.5** juga bisa dipilih jika ingin menangani fitur kontinu + gain ratio + pruning, dengan konsekuensi implementasi lebih kompleks.
2. **Logistic Regression** — diimplementasikan manual: forward pass (sigmoid), loss (binary cross-entropy), gradien, dan update parameter.
3. **Support Vector Machine (SVM)** — diimplementasikan manual. Mengingat ukuran dataset (~28,8 ribu baris train), **direkomendasikan** SVM linear soft-margin dengan hinge loss yang dioptimasi lewat (mini-batch) gradient descent (formulasi **primal**) — jauh lebih tractable dengan numpy murni dibanding SVM kernel/dual (matriks kernel n×n untuk ~28,8k baris terlalu besar untuk dihitung naif). Eksplorasi kernel non-linear (jika diinginkan) dilakukan sebagai eksperimen tambahan opsional pada subsample data, bukan model utama.

**Aturan library**:
- Library yang boleh dipakai untuk **implementasi inti** ketiga algoritma di atas **hanya library komputasi matematis murni** — utamanya `numpy`. `pandas` boleh dipakai untuk manipulasi data di luar logika algoritma, `matplotlib`/`seaborn` untuk visualisasi.
- **Dilarang** memakai `scikit-learn`, `scipy.optimize`, atau library ML/optimizer siap pakai lain **di dalam** implementasi from-scratch itu sendiri (termasuk untuk encoding/scaling di jalur from-scratch — implementasikan manual dengan numpy/pandas, lihat Bagian 5 poin 4).
- `scikit-learn` (`DecisionTreeClassifier`, `LogisticRegression`, `SVC`/`LinearSVC`, dst.) **hanya boleh dipakai sebagai baseline pembanding**, dan boleh dipakai bebas untuk metric verifikasi (`sklearn.metrics.f1_score`, dsb.) sebagai cross-check.
- **WAJIB**: bandingkan hasil (macro F1, precision, recall per kelas) dari ketiga implementasi from-scratch dengan hasil algoritma sejenis dari scikit-learn, pada skema evaluasi yang sama (CV/split identik) agar perbandingan adil.
- **Dilarang** menggunakan ensemble method (voting, bagging, boosting, stacking, dst.) atau algoritma apa pun di luar tiga yang disebutkan.
- **Submission Kaggle hanya boleh dihasilkan dari model buatan sendiri (from scratch)** — model scikit-learn sama sekali tidak boleh dipakai untuk menghasilkan file submission, murni sebagai pembanding/sanity check.

### Requirement visualisasi & eksperimen tambahan

- **Sertakan gambar percabangan tree** hasil implementasi Decision Tree Learning from scratch (bukan hasil sklearn).
- **Sertakan visualisasi proses training untuk Logistic Regression**, minimal: kontur fungsi loss dan lintasan parameter selama training.
- **Implementasikan algoritma optimasi tambahan di luar gradient descent dasar** (misal SGD dengan momentum atau Adam) untuk training Logistic Regression (dan gunakan optimizer yang sama untuk SVM), lalu bandingkan konvergensinya dengan gradient descent biasa.

## 2. Data

File yang tersedia (nama file: `train.csv`, `test.csv`, `sample_submission.csv`, path dirujuk lewat `src/config.py` — lihat Bagian 4 soal struktur folder):

- `train.csv`: **28.800 baris**, kolom lengkap + label `loan_status`.
  - Distribusi kelas: **22.400 baris label 0 / ditolak (77.8%)**, **6.400 baris label 1 / disetujui (22.2%)** → imbalance ratio ~3.5:1.
- `test.csv`: **7.200 baris**, tanpa kolom `loan_status`.
- `sample_submission.csv`: format submission, kolom `person_id, loan_status`.
- **Tidak ada missing value** di kolom manapun (train maupun test) — terverifikasi.
- **Tidak ada baris duplikat**, tidak ada `person_id` duplikat, dan **tidak ada overlap `person_id`** antara train & test — terverifikasi.

Skema kolom (nama kolom terverifikasi dari file asli):

| Kolom | Tipe | Keterangan |
|---|---|---|
| `person_id` | int | ID unik pemohon. **Hanya untuk join/submission, bukan fitur prediktif.** |
| `person_age` | float | Usia pemohon (tahun) |
| `person_gender` | kategorik | `male`, `female` |
| `person_income` | float | Pendapatan tahunan |
| `person_emp_exp` | int | Lama pengalaman kerja (tahun) |
| `person_home_ownership` | kategorik | `RENT`, `OWN`, `MORTGAGE`, `OTHER` |
| `loan_amnt` | float | Jumlah pinjaman yang diajukan |
| `loan_int_rate` | float | Suku bunga pinjaman (%) |
| `loan_percent_income` | float | Rasio jumlah pinjaman terhadap pendapatan pemohon |
| `cb_person_cred_hist_length` | float | Lama riwayat kredit (tahun) |
| `credit_score` | int | Skor kredit pemohon (rentang realistis, mirip FICO 300–850) |
| `previous_loan_defaults_on_file` | kategorik | `Yes`, `No` |
| `loan_status` | int | **Target.** 0 = ditolak, 1 = disetujui. Hanya ada di `train.csv` |

Format submission (harus persis sesuai `sample_submission.csv`):

```
person_id,loan_status
8121,0
22683,0
17123,0
...
```

## 3. Temuan EDA & Kualitas Data — WAJIB Ditangani Saat Cleaning/Preprocessing

Hasil investigasi langsung terhadap `train.csv`/`test.csv` (verifikasi ulang secara terprogram di notebook EDA):

1. **Outlier ekstrem pada `person_age` & `person_emp_exp`**: 7 baris di train dan 1 baris di test punya `person_age` tidak masuk akal (sampai 144 tahun) dan `person_emp_exp` yang mustahil (sampai 121–125 tahun pengalaman kerja). Semua 7 baris outlier di train punya `loan_status = 0`.
   - **Penanganan**: JANGAN drop baris test manapun. Lakukan capping/winsorizing pada `person_age` dan `person_emp_exp` ke batas wajar (misal cap `person_age` di ~90, `person_emp_exp` di ~50), dengan bound yang **di-fit dari train saja** lalu diterapkan konsisten ke test. Opsional: tambahkan fitur biner `is_outlier_age_exp` sebelum capping.
2. **`person_income` long-tail ekstrem**: 144 baris train (~0.5%, di atas persentil 99.5) punya income sangat ekstrem, sampai ~7,2 juta — jauh dari IQR normal (median ~67 ribu).
   - **Penanganan**: winsorize/clip `person_income` pada persentil tertentu (misal p1–p99, fit dari train) **atau** transformasi `log1p`. Penting terutama untuk Logistic Regression & SVM yang sensitif skala/outlier.
3. **Tidak ada nilai negatif** di kolom numerik manapun — aman.
4. **`credit_score` dalam rentang wajar** (train: 418–850, test: 390–807) — tidak perlu cleaning.
5. **Kategori `OTHER` pada `person_home_ownership` sangat jarang** (~0.3% baris). Bukan error, tapi perhatikan saat encoding manual agar kategori tak dikenal di test tidak menyebabkan error (tangani eksplisit, misal kolom "unknown" fallback).
6. **Temuan paling signifikan** — `previous_loan_defaults_on_file`: dari seluruh 14.629 baris train dengan nilai `Yes`, **0% disetujui** (`loan_status` selalu 0). Predictor yang sangat dominan/near-deterministic terhadap kelas 0.
   - **Wajib** ditonjolkan di EDA (crosstab/bar chart approval rate per kategori).
   - Biarkan model mempelajari bobot fitur ini secara natural lewat training — **jangan** membuat aturan if-else hardcode yang meng-override prediksi model.

## 4. Instruksi Wajib — Struktur Project

**Struktur folder project hanya terdiri dari dua folder utama yang sudah ada: `src/` dan `notebooks/`.** Jangan membuat folder top-level baru seperti `data/` atau `outputs/` — asumsikan lokasi file data (`train.csv`, `test.csv`, `sample_submission.csv`) sudah tersedia dan cukup dirujuk lewat path yang dikonfigurasi di `src/config.py`.

- Pisahkan tegas: seluruh logika inti (loading, cleaning, preprocessing manual, implementasi algoritma from scratch, optimizer, baseline sklearn, evaluasi, visualisasi, prediksi) ditulis sebagai fungsi/kelas modular di `src/`, lengkap docstring & type hints.
- Notebook di `notebooks/` **hanya** untuk orkestrasi (memanggil fungsi/kelas dari `src`), visualisasi, dan interpretasi hasil — **tidak boleh** mendefinisikan logika inti (termasuk algoritma from scratch) langsung di cell notebook.
- Notebook mengimpor `src` sebagai package lokal.
- Artefak yang dihasilkan (model tersimpan, gambar struktur tree, plot loss contour, file submission) disimpan di sub-folder **di dalam** `src/` (misal `src/artifacts/models/`, `src/artifacts/figures/`, `src/artifacts/submissions/`) — bukan folder top-level baru.
- Random seed konsisten di seluruh project.
- Semua bound cleaning/preprocessing (capping, encoding, scaling) wajib di-fit dari train saja, diterapkan konsisten ke test — tidak boleh ada data leakage.

### Struktur folder

```
src/
├── __init__.py
├── config.py                     # path, seed, daftar kolom, bound cleaning, hyperparameter
├── data.py                       # load_train(), load_test(), load_sample_submission()
├── cleaning.py                   # penanganan outlier (Bagian 3), fit dari train saja
├── preprocessing.py              # encoding & scaling MANUAL (numpy/pandas) untuk jalur from-scratch
├── algorithms/
│   ├── __init__.py
│   ├── decision_tree.py          # implementasi from-scratch (ID3/C4.5/CART — pilih satu)
│   ├── logistic_regression.py    # implementasi from-scratch, pakai src/optimizers.py
│   └── svm.py                    # implementasi from-scratch (linear soft-margin, hinge loss)
├── optimizers.py                 # gradient descent biasa + minimal satu optimizer tambahan (mis. momentum/Adam)
├── sklearn_baselines.py          # wrapper DecisionTreeClassifier/LogisticRegression/SVC untuk pembanding SAJA
├── evaluation.py                 # cross-validation manual, macro F1, threshold tuning, classification report
├── visualization.py              # plot_tree_structure(), plot_loss_contour(), plot_parameter_trajectory()
├── predict.py                    # generate_submission() — HANYA dari model from-scratch
└── utils.py                      # set_seed(), save/load artefak model

notebooks/
├── 01_eda.ipynb
├── 02_dtl_scratch_vs_sklearn.ipynb
├── 03_logistic_regression_scratch_vs_sklearn.ipynb
├── 04_svm_scratch_vs_sklearn.ipynb
└── 05_final_submission.ipynb
```

## 5. Tahapan Pipeline yang Harus Diimplementasikan

1. **Data loading & validasi** (`src/data.py`): load train/test, verifikasi ulang shape (28.800/7.200), tidak ada missing value/duplikat.
2. **EDA** (`notebooks/01_eda.ipynb`): distribusi target, distribusi fitur, boxplot outlier (Bagian 3), crosstab `previous_loan_defaults_on_file` vs `loan_status`.
3. **Data Cleaning** (`src/cleaning.py`): capping/winsorizing `person_age`, `person_emp_exp`, `person_income` — bound fit dari train, terapkan ke test, tidak drop baris test.
4. **Preprocessing manual** (`src/preprocessing.py`): implementasikan sendiri fungsi one-hot encoding & standardization (numpy/pandas) untuk jalur from-scratch — **tidak** memakai `ColumnTransformer`/`OneHotEncoder`/`StandardScaler` dari sklearn di jalur ini (untuk jalur baseline sklearn, boleh pakai tools bawaan sklearn seperti biasa di `sklearn_baselines.py`). `person_id` dikeluarkan dari fitur.
5. **Decision Tree Learning from scratch** (`src/algorithms/decision_tree.py` + notebook `02`):
   - Implementasikan varian terpilih (ID3/C4.5/CART) — split criterion, stopping criteria (`max_depth`, `min_samples_split`, dst.), traversal untuk prediksi, penanganan class imbalance (misal class weight pada criterion atau weighted voting di leaf).
   - **Visualisasikan struktur tree** hasil implementasi from-scratch sebagai gambar (`src/visualization.py` → `plot_tree_structure()`, digambar manual dengan matplotlib karena bukan model sklearn). Simpan ke `src/artifacts/figures/`.
   - Latih baseline `DecisionTreeClassifier` sklearn dengan hyperparameter senada (`src/sklearn_baselines.py`), bandingkan macro F1 (CV) antara versi from-scratch vs sklearn di notebook.
6. **Logistic Regression from scratch** (`src/algorithms/logistic_regression.py` + notebook `03`):
   - Forward pass (sigmoid), binary cross-entropy loss (dengan opsi class weight), gradien manual.
   - Implementasikan **minimal dua optimizer** (`src/optimizers.py`): (a) batch/mini-batch gradient descent biasa, (b) minimal satu optimizer tambahan (mis. momentum atau Adam). Bandingkan kecepatan konvergensi (loss vs iterasi) antar optimizer.
   - **Visualisasikan proses training**: kurva loss vs iterasi per optimizer, plus kontur fungsi loss + lintasan parameter — karena dimensi parameter >2, proyeksikan ke 2 dimensi terpilih (misal bias & koefisien `previous_loan_defaults_on_file`, atau 2 komponen PCA dari parameter) dan dokumentasikan simplifikasi ini secara eksplisit.
   - Latih baseline `LogisticRegression` sklearn dengan setting senada, bandingkan macro F1 (CV).
7. **SVM from scratch** (`src/algorithms/svm.py` + notebook `04`):
   - Soft-margin linear SVM (primal, hinge loss + regularisasi L2), dioptimasi dengan optimizer dari `src/optimizers.py` (gradient descent + optimizer tambahan yang sama seperti poin 6).
   - Class imbalance ditangani manual (class weight pada hinge loss).
   - Latih baseline `SVC`/`LinearSVC` sklearn dengan setting senada, bandingkan macro F1 (CV).
8. **Threshold tuning** (`src/evaluation.py`): untuk Logistic Regression (probabilistik) dan SVM (skor/decision function), cari threshold yang memaksimalkan macro F1 di data validasi/CV — implementasi pencarian threshold manual (boleh pakai `sklearn.metrics.f1_score` sekadar cross-check hasil).
9. **Perbandingan & pemilihan model final** (`notebooks/05_final_submission.ipynb`): rangkum macro F1 (CV) ketiga model from-scratch vs baseline sklearn-nya dalam satu tabel/plot, pilih model **from-scratch terbaik** (bukan sklearn) sebagai model final.
10. **Prediksi & submission** (`src/predict.py`): jalankan model from-scratch terpilih pada `test.csv`, hasilkan `submission.csv` (kolom `person_id, loan_status`, 7.200 baris) sesuai format `sample_submission.csv`. **Tidak boleh** memakai model sklearn untuk langkah ini. Simpan ke `src/artifacts/submissions/`.
11. Simpan parameter/model hasil training from-scratch (misal `pickle`/`numpy.save`, bukan `joblib` khusus sklearn) ke `src/artifacts/models/`.

## 6. Deliverables

- Seluruh struktur `src/` dan `notebooks/` di atas, berjalan end-to-end.
- `requirements.txt` (numpy, pandas, matplotlib/seaborn, scikit-learn — untuk baseline & metric cross-check saja).
- Gambar struktur tree, plot loss contour & parameter trajectory Logistic Regression (`src/artifacts/figures/`).
- Tabel/plot perbandingan macro F1 from-scratch vs scikit-learn untuk ketiga algoritma.
- `README.md` berisi: cara menjalankan notebook secara berurutan, alasan pemilihan varian Decision Tree (ID3/C4.5/CART), ringkasan optimizer yang diimplementasikan, ringkasan hasil perbandingan from-scratch vs sklearn, dan model final yang dipakai untuk submission.

## 7. Catatan Penting

- Implementasi inti (tree, logistic regression, SVM, optimizer, encoding/scaling di jalur from-scratch) **wajib** hanya pakai numpy (+pandas untuk data handling) — tidak boleh scikit-learn di dalamnya.
- scikit-learn hanya untuk baseline pembanding & cross-check metric, **tidak untuk submission**.
- Tidak boleh ensemble method atau model apa pun di luar DTL, Logistic Regression, SVM.
- Submission akhir wajib dari model from-scratch.
- Jangan drop baris test — submission wajib 7.200 baris.
- Bound cleaning/preprocessing wajib fit dari train saja.
- Manfaatkan temuan `previous_loan_defaults_on_file` secara alami lewat model, bukan hardcode override.
- Struktur project hanya `src/` dan `notebooks/` — jangan buat folder top-level baru; artefak (gambar, model, submission) taruh di sub-folder dalam `src/`.