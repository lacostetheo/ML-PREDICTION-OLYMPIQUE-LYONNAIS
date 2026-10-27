"""Baseline probabilistic models. Common interface:

    model.fit(train_df) -> self
    model.predict_proba(df) -> (n, 3) array, columns = [away, draw, home]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, PoissonRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = np.logspace(-4, 1, 8)


class FrequencyModel:
    """Predicts the training-set class frequencies for every match (naive baseline)."""

    def fit(self, df: pd.DataFrame) -> "FrequencyModel":
        self.p_ = np.bincount(df["y"], minlength=3) / len(df)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.tile(self.p_, (len(df), 1))


class LogisticModel:
    """Multinomial logistic regression on selected columns.

    Median imputation (+ missing indicator) and standardisation are fitted on the
    training data only. The L2 strength C is chosen by time-ordered
    cross-validation *inside the training set* (TimeSeriesSplit, log loss)."""

    def __init__(self, cols: list[str], n_splits: int = 5):
        self.cols, self.n_splits = list(cols), n_splits

    def fit(self, df: pd.DataFrame) -> "LogisticModel":
        clf = LogisticRegressionCV(Cs=C_GRID, cv=TimeSeriesSplit(self.n_splits),
                                   scoring="neg_log_loss", max_iter=3000)
        self.pipe_ = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                   StandardScaler(), clf)
        self.pipe_.fit(df[self.cols], df["y"])
        self.C_ = float(clf.C_[0])
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipe_.predict_proba(df[self.cols])


class PoissonGoalModel:
    """Independent Poisson goals (log-linear regression) -> 1X2 probabilities.

    Home goals ~ Poisson(λ_h), away goals ~ Poisson(λ_a), each fitted by a Poisson
    GLM. Match outcome probabilities come from the score matrix. Independence
    typically under-estimates draws; it is kept as a classical reference and is
    also what the season simulation can use to draw scorelines."""

    def __init__(self, home_cols: list[str], away_cols: list[str], max_goals: int = 10):
        self.home_cols, self.away_cols, self.max_goals = list(home_cols), list(away_cols), max_goals

    def _pipe(self):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             PoissonRegressor(alpha=1e-3, max_iter=1000))

    def fit(self, df: pd.DataFrame) -> "PoissonGoalModel":
        self.home_ = self._pipe().fit(df[self.home_cols], df["home_goals"])
        self.away_ = self._pipe().fit(df[self.away_cols], df["away_goals"])
        return self

    def rates(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.home_.predict(df[self.home_cols]), self.away_.predict(df[self.away_cols])

    def score_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """(n, G+1, G+1) array: P(home goals = i, away goals = j)."""
        lam_h, lam_a = self.rates(df)
        g = np.arange(self.max_goals + 1)
        ph = poisson.pmf(g[None, :], lam_h[:, None])
        pa = poisson.pmf(g[None, :], lam_a[:, None])
        return ph[:, :, None] * pa[:, None, :]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        S = self.score_matrix(df)
        i, j = np.indices(S.shape[1:])
        P = np.column_stack([(S * (i < j)).sum(axis=(1, 2)),
                             (S * (i == j)).sum(axis=(1, 2)),
                             (S * (i > j)).sum(axis=(1, 2))])
        return P / P.sum(axis=1, keepdims=True)


class BookmakerModel:
    """Reference only (not a model we fit): Bet365 odds converted to probabilities by
    proportional normalisation, which removes the bookmaker's margin (overround)."""

    def fit(self, df: pd.DataFrame) -> "BookmakerModel":
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        inv = 1.0 / df[["b365_a", "b365_d", "b365_h"]].to_numpy()
        return inv / inv.sum(axis=1, keepdims=True)
