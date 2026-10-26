"""Neural networks with scikit-learn's ``MLPClassifier``.

scikit-learn has no Dropout or Batch Normalisation. Regularisation therefore comes
from (a) an L2 penalty (``alpha``), (b) small networks, (c) a limited number of
epochs chosen by time-ordered cross-validation, and (d) averaging several random
initialisations (``SeedEnsemble``), which reduces variance much as dropout does.
``early_stopping`` is deliberately OFF: scikit-learn's version holds out a random
validation set, which would mix future and past matches.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from olpred.models.ml import N_SPLITS, TunedModel

warnings.filterwarnings("ignore", category=ConvergenceWarning)

SEEDS = (0, 1, 2, 3, 4)


def _pipeline(seed: int = 0):
    mlp = MLPClassifier(activation="relu", solver="adam", batch_size=256,
                        early_stopping=False, random_state=seed)
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), mlp)


def _space(architectures: list[tuple[int, ...]]) -> dict:
    return {"mlpclassifier__hidden_layer_sizes": architectures,
            "mlpclassifier__alpha": loguniform(1e-1, 1e2),          # strong L2: data are very noisy
            "mlpclassifier__learning_rate_init": loguniform(1e-4, 3e-3),
            "mlpclassifier__max_iter": [10, 20, 40, 80]}            # epochs = the other regulariser


def simple_mlp(cols, seed=0, n_iter=20) -> TunedModel:
    """Input -> Dense(ReLU) -> softmax output; one hidden layer."""
    search = RandomizedSearchCV(_pipeline(seed), _space([(8,), (16,), (32,), (64,)]), n_iter=n_iter,
                                cv=TimeSeriesSplit(N_SPLITS), scoring="neg_log_loss",
                                random_state=seed, n_jobs=4, refit=True)
    return TunedModel("MLP simple (1 hidden layer)", cols, search)


def deep_mlp(cols, seed=0, n_iter=20) -> TunedModel:
    """Input -> Dense(ReLU) x 2-3 -> softmax output."""
    search = RandomizedSearchCV(_pipeline(seed), _space([(32, 16), (64, 32), (64, 32, 16), (128, 64, 32)]),
                                n_iter=n_iter, cv=TimeSeriesSplit(N_SPLITS), scoring="neg_log_loss",
                                random_state=seed, n_jobs=4, refit=True)
    return TunedModel("MLP deep (2-3 hidden layers)", cols, search)


class SeedEnsemble:
    """Average of the same tuned network trained from several random initialisations."""

    def __init__(self, tuned: TunedModel, seeds=SEEDS):
        self.cols, self.seeds = tuned.cols, seeds
        self.template = tuned.search.best_estimator_

    def fit(self, df: pd.DataFrame) -> "SeedEnsemble":
        self.members_ = [clone(self.template).set_params(mlpclassifier__random_state=s)
                         .fit(df[self.cols], df["y"]) for s in self.seeds]
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_proba(df[self.cols]) for m in self.members_], axis=0)
