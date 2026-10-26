"""Classical ML models (scikit-learn) with time-ordered hyper-parameter search.

All searches use ``TimeSeriesSplit`` on the *training seasons only* (each fold
trains on the past and scores on the following block, scoring = log loss), so no
validation/test data and no future data enters the choice of hyper-parameters.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

N_SPLITS = 5


class TunedModel:
    """Wraps a search object: ``fit(df)`` tunes on ``df`` then refits the best
    configuration on all of ``df``. Exposes the interface of baselines.py."""

    def __init__(self, name: str, cols: list[str], search):
        self.name, self.cols, self.search = name, list(cols), search

    def fit(self, df: pd.DataFrame) -> "TunedModel":
        start = time.time()
        self.search.fit(df[self.cols], df["y"])
        self.fit_seconds_ = time.time() - start
        self.best_params_ = self.search.best_params_
        self.cv_log_loss_ = -float(self.search.best_score_)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.search.predict_proba(df[self.cols])

    def tuning_table(self) -> pd.DataFrame:
        res = pd.DataFrame(self.search.cv_results_)
        out = pd.DataFrame({"model": self.name, "cv_log_loss": -res["mean_test_score"],
                            "cv_std": res["std_test_score"], "params": res["params"].astype(str)})
        return out.sort_values("cv_log_loss").reset_index(drop=True)


def random_forest(cols, seed=0, n_iter=12) -> TunedModel:
    pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                         RandomForestClassifier(random_state=seed, n_jobs=-1))
    space = {"randomforestclassifier__n_estimators": [200, 400],
             "randomforestclassifier__max_depth": [4, 6, 8, 12, None],
             "randomforestclassifier__min_samples_leaf": [10, 25, 50, 100],
             "randomforestclassifier__max_features": ["sqrt", 0.3, 0.5]}
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=TimeSeriesSplit(N_SPLITS),
                                scoring="neg_log_loss", random_state=seed, refit=True)
    return TunedModel("Random Forest", cols, search)


def gradient_boosting(cols, seed=0, n_iter=20) -> TunedModel:
    # HistGradientBoosting = scikit-learn's histogram-based gradient boosting
    # (same algorithm family as LightGBM). Early stopping is off: sklearn's built-in
    # one uses a *random* validation split, which would be temporally leaky.
    pipe = HistGradientBoostingClassifier(early_stopping=False, random_state=seed)
    space = {"learning_rate": loguniform(0.01, 0.2),
             "max_iter": [50, 100, 200, 400],
             "max_depth": [2, 3, 4, None],
             "max_leaf_nodes": [8, 15, 31],
             "min_samples_leaf": [20, 50, 100, 200],
             "l2_regularization": [0.0, 0.1, 1.0, 10.0]}
    search = RandomizedSearchCV(pipe, space, n_iter=n_iter, cv=TimeSeriesSplit(N_SPLITS),
                                scoring="neg_log_loss", random_state=seed, refit=True, n_jobs=1)
    return TunedModel("Gradient Boosting", cols, search)


def svm_rbf(cols) -> TunedModel:
    """RBF SVM on a small feature set (training cost grows ~ n²)."""
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=0))
    grid = {"svc__C": [0.3, 1.0], "svc__gamma": [0.02, "scale"]}
    search = GridSearchCV(pipe, grid, cv=TimeSeriesSplit(3), scoring="neg_log_loss",
                          refit=True, n_jobs=4)
    return TunedModel("SVM (RBF)", cols, search)


class Blend:
    """Equal-weight average of the probabilities of fitted models (no parameters)."""

    def __init__(self, models: list):
        self.models = models

    def fit(self, df: pd.DataFrame) -> "Blend":
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_proba(df) for m in self.models], axis=0)
