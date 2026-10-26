"""xG *proxy*: expected goals from shots, via a plain linear regression.

    goals ≈ a · (shots on target) + b · (shots off target)

This is NOT real xG (no shot location / type). It is fitted on a fixed early
period only (default seasons <= 2019) so that no later information enters it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class XGProxy:
    def __init__(self) -> None:
        self.model = LinearRegression(fit_intercept=False, positive=True)

    @staticmethod
    def _design(shots: pd.Series, sot: pd.Series) -> np.ndarray:
        return np.column_stack([sot, shots - sot])

    def fit(self, shots: pd.Series, sot: pd.Series, goals: pd.Series) -> "XGProxy":
        ok = shots.notna() & sot.notna() & goals.notna()
        self.model.fit(self._design(shots[ok], sot[ok]), goals[ok])
        return self

    @property
    def coefs(self) -> dict[str, float]:
        return {"per_shot_on_target": float(self.model.coef_[0]),
                "per_shot_off_target": float(self.model.coef_[1])}

    def predict(self, shots: pd.Series, sot: pd.Series) -> pd.Series:
        ok = shots.notna() & sot.notna()
        out = pd.Series(np.nan, index=shots.index)
        out[ok] = self.model.predict(self._design(shots[ok], sot[ok]))
        return out
