"""
cleaning.py — Penanganan outlier: capping/winsorizing.

Semua bound di-fit dari train saja, lalu diterapkan konsisten ke test.
"""

import numpy as np
import pandas as pd
from . import config


class DataCleaner:
    """Fit bounds dari train, transform train & test secara konsisten."""

    def __init__(self):
        self.income_lower: float = None
        self.income_upper: float = None
        self._fitted = False

    def fit(self, df_train: pd.DataFrame) -> "DataCleaner":
        """Fit cleaning bounds dari data train."""
        self.income_lower = np.percentile(
            df_train["person_income"], config.INCOME_LOWER_PERCENTILE
        )
        self.income_upper = np.percentile(
            df_train["person_income"], config.INCOME_UPPER_PERCENTILE
        )
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Terapkan capping/winsorizing ke dataframe. Return copy."""
        assert self._fitted, "Call fit() first"
        df = df.copy()

        # 1. Cap person_age
        df["person_age"] = df["person_age"].clip(upper=config.AGE_CAP)

        # 2. Cap person_emp_exp
        df["person_emp_exp"] = df["person_emp_exp"].clip(upper=config.EMP_EXP_CAP)

        # 3. Winsorize person_income (bounds dari train)
        df["person_income"] = df["person_income"].clip(
            lower=self.income_lower, upper=self.income_upper
        )

        return df

    def fit_transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
        """Fit dari train lalu transform train."""
        return self.fit(df_train).transform(df_train)
