"""
preprocessing.py — Encoding & scaling MANUAL (numpy/pandas).

One-hot encoding dan standardization diimplementasikan sendiri,
TANPA sklearn ColumnTransformer/OneHotEncoder/StandardScaler.
Semua parameter (categories, mean, std) di-fit dari train saja.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from . import config


class ManualOneHotEncoder:
    """One-hot encoding manual — fit categories dari train."""

    def __init__(self):
        self.categories_: Dict[str, List[str]] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "ManualOneHotEncoder":
        """Fit: catat unique categories per kolom dari train."""
        for col in columns:
            self.categories_[col] = sorted(df[col].unique().tolist())
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform: buat kolom one-hot, drop kolom asli."""
        assert self._fitted, "Call fit() first"
        df = df.copy()
        for col, cats in self.categories_.items():
            for cat in cats:
                df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
            df = df.drop(columns=[col])
        return df

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return self.fit(df, columns).transform(df)

    def get_feature_names(self) -> List[str]:
        """Return daftar nama kolom one-hot yang dihasilkan."""
        names = []
        for col, cats in self.categories_.items():
            for cat in cats:
                names.append(f"{col}_{cat}")
        return names


class ManualStandardScaler:
    """Standardization manual (z-score) — fit mean/std dari train."""

    def __init__(self):
        self.mean_: Dict[str, float] = {}
        self.std_: Dict[str, float] = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame, columns: List[str]) -> "ManualStandardScaler":
        """Fit: hitung mean dan std per kolom dari train."""
        for col in columns:
            self.mean_[col] = df[col].mean()
            self.std_[col] = df[col].std()
            # Hindari pembagian nol
            if self.std_[col] == 0:
                self.std_[col] = 1.0
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform: standardize kolom numerik."""
        assert self._fitted, "Call fit() first"
        df = df.copy()
        for col in self.mean_:
            df[col] = (df[col] - self.mean_[col]) / self.std_[col]
        return df

    def fit_transform(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        return self.fit(df, columns).transform(df)


class Preprocessor:
    """
    Pipeline preprocessing lengkap:
    1. Drop person_id
    2. One-hot encode kategorik
    3. Standardize numerik

    Fit dari train, transform train & test.
    """

    def __init__(self):
        self.encoder = ManualOneHotEncoder()
        self.scaler = ManualStandardScaler()
        self.feature_names_: List[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Fit encoder & scaler dari train dataframe."""
        self.encoder.fit(df, config.CATEGORICAL_COLS)
        self.scaler.fit(df, config.NUMERICAL_COLS)
        # Compute feature names
        self.feature_names_ = list(config.NUMERICAL_COLS) + self.encoder.get_feature_names()
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform dataframe → numpy array fitur.
        Drop person_id dan target jika ada. Return (X, y) atau (X,).
        """
        assert self._fitted, "Call fit() first"
        df = df.copy()

        # Drop ID
        if config.ID_COL in df.columns:
            df = df.drop(columns=[config.ID_COL])

        # Pisahkan target jika ada
        y = None
        if config.TARGET_COL in df.columns:
            y = df[config.TARGET_COL].values.astype(np.float64)
            df = df.drop(columns=[config.TARGET_COL])

        # Encode kategorik
        df = self.encoder.transform(df)

        # Standardize numerik
        df = self.scaler.transform(df)

        # Susun fitur sesuai urutan: numerik dulu, lalu one-hot
        X = df[self.feature_names_].values.astype(np.float64)

        if y is not None:
            return X, y
        return X

    def fit_transform(self, df: pd.DataFrame):
        """Fit dari train lalu transform."""
        self.fit(df)
        return self.transform(df)
