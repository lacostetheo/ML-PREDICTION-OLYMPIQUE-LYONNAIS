"""Final model line-up (curated features, hyper-parameters frozen from Phases 4-6)."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from olpred.features.columns import curated_cols
from olpred.models.baselines import BookmakerModel, FrequencyModel, LogisticModel, PoissonGoalModel
from olpred.models.neural import FixedMLP

GB_PARAMS = {"learning_rate": 0.0294, "max_iter": 100, "max_depth": 2, "max_leaf_nodes": 15,
             "min_samples_leaf": 200, "l2_regularization": 1.0}  # Phase 5


class FrozenGB:
    def __init__(self, cols):
        self.cols = list(cols)

    def fit(self, df: pd.DataFrame) -> "FrozenGB":
        self.est_ = HistGradientBoostingClassifier(early_stopping=False, random_state=0, **GB_PARAMS
                                                   ).fit(df[self.cols], df["y"])
        return self

    def predict_proba(self, df: pd.DataFrame):
        return self.est_.predict_proba(df[self.cols])


NAIVE, ELO, POISSON, LOGREG, GB, MLP, ENSEMBLE, MARKET = (
    "Class frequency (naive)", "Elo (LogReg)", "Poisson goals", "LogReg (final)",
    "Gradient boosting", "MLP deep (5 seeds)", "Ensemble LogReg+GB+MLP", "Bookmaker Bet365 (reference)")


def make_models(df: pd.DataFrame) -> dict[str, object]:
    cols = curated_cols(df)
    return {
        NAIVE: FrequencyModel(),
        ELO: LogisticModel(["elo_diff"]),
        POISSON: PoissonGoalModel(["h_gf_10", "a_ga_10", "elo_diff"], ["a_gf_10", "h_ga_10", "elo_diff"]),
        LOGREG: LogisticModel(cols),
        GB: FrozenGB(cols),
        MLP: FixedMLP(cols),
        MARKET: BookmakerModel(),
    }
