import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from olpred.evaluation.metrics import brier, paired_block_bootstrap, per_match_log_loss, rps
from olpred.evaluation.splits import TEST, TRAIN, VAL, split
from olpred.models.baselines import BookmakerModel, FrequencyModel, PoissonGoalModel


def test_log_loss_matches_sklearn():
    rng = np.random.default_rng(0)
    P = rng.dirichlet([2, 1, 2], size=200)
    y = rng.integers(0, 3, 200)
    assert np.isclose(per_match_log_loss(y, P).mean(), log_loss(y, P, labels=[0, 1, 2]))


def test_perfect_and_worst_forecasts():
    y = np.array([0, 1, 2])
    perfect = np.eye(3)[y]
    assert brier(y, perfect) == 0 and rps(y, perfect) == 0
    assert rps(np.array([2]), np.array([[1.0, 0.0, 0.0]])) == 1.0  # maximal ordinal miss


def test_bootstrap_identical_losses_is_not_significant():
    loss = np.random.default_rng(1).random(300)
    res = paired_block_bootstrap(loss, loss.copy(), np.arange(300) // 10, n_boot=500)
    assert res["diff"] == 0 and not res["significant"]


def test_bootstrap_detects_clear_difference():
    rng = np.random.default_rng(2)
    a = rng.random(500)
    res = paired_block_bootstrap(a, a + 0.2, np.arange(500) // 10, n_boot=500)
    assert res["significant"] and res["ci_high"] < 0


def test_splits_are_chronological_and_disjoint():
    seasons = np.repeat(np.arange(2010, 2027), 5)
    df = pd.DataFrame({"season": seasons, "date": pd.to_datetime(seasons, format="%Y"),
                       "match_id": np.arange(len(seasons))})
    parts = split(df)
    assert parts["train"]["season"].between(*TRAIN).all()
    assert parts["val"]["season"].between(*VAL).all()
    assert parts["test"]["season"].between(*TEST).all()
    assert parts["train"]["date"].max() < parts["val"]["date"].min() < parts["test"]["date"].min()
    assert 2010 not in set().union(*[set(p["season"]) for p in parts.values()])  # warm-up excluded


def test_poisson_probabilities_are_valid():
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame({"x": rng.normal(size=n)})
    df["home_goals"] = rng.poisson(np.exp(0.3 + 0.2 * df.x))
    df["away_goals"] = rng.poisson(np.exp(0.0 - 0.2 * df.x))
    model = PoissonGoalModel(["x"], ["x"]).fit(df)
    P = model.predict_proba(df)
    assert np.allclose(P.sum(axis=1), 1) and (P >= 0).all()
    assert P[:, 2].mean() > P[:, 0].mean()  # home advantage recovered
    assert np.allclose(model.score_matrix(df).sum(axis=(1, 2)), 1, atol=1e-3)


def test_frequency_and_bookmaker_models():
    df = pd.DataFrame({"y": [2, 2, 1, 0], "b365_h": [2.0, 1.5, 3.0, 4.0],
                       "b365_d": [3.5, 4.0, 3.2, 3.6], "b365_a": [4.0, 6.0, 2.4, 1.8]})
    assert np.allclose(FrequencyModel().fit(df).predict_proba(df)[0], [0.25, 0.25, 0.5])
    P = BookmakerModel().predict_proba(df)
    assert np.allclose(P.sum(axis=1), 1)
    assert P[1, 2] > P[1, 0]  # shorter home odds -> higher home probability
