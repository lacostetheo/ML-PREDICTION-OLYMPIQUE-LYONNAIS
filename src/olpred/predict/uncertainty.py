"""Prediction uncertainty for single matches.

Two sources, kept separate:
 - model uncertainty: the logistic regression is refit on week-block bootstrap
   resamples of the training matches (how much do the probabilities move if the
   history had been slightly different?);
 - rating uncertainty: the true strengths are not observed exactly, so each team's
   Elo is shaken by N(0, sigma) points.
Outcome randomness (the match itself) is not an interval: it is what the probabilities say.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from olpred.models.online import refit_logistic, week_ids
from olpred.predict.simulate import LinearSim


def bootstrap_models(train: pd.DataFrame, cols: list[str], C: float, n_boot: int = 100, seed: int = 0):
    """Logistic regressions refit on bootstrap resamples (whole calendar weeks resampled)."""
    rng = np.random.default_rng(seed)
    weeks = week_ids(train["date"])
    groups = {w: np.flatnonzero(weeks == w) for w in np.unique(weeks)}
    keys = np.array(list(groups))
    models = []
    for _ in range(n_boot):
        pick = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[w] for w in pick])
        pipe = refit_logistic(cols, C, train.iloc[idx])
        models.append(SimpleNamespace(pipe_=pipe, cols=cols))
    return models


def match_intervals(models, X: pd.DataFrame, sigma: float, n_shocks: int = 20, seed: int = 0,
                    level: float = 0.90) -> pd.DataFrame:
    """Point forecast and intervals for each fixture row of ``X``.

    Returns, per fixture: p_away/p_draw/p_home (central, from the bootstrap mean),
    expected points of the home and away side, and the ``level`` interval from
    (a) model uncertainty only and (b) model + rating uncertainty."""
    rng = np.random.default_rng(seed)
    n = len(X)
    model_only, total = [], []
    for m in models:
        sim = LinearSim(m, X)
        zero = np.zeros((1, n))
        model_only.append(sim.probs(zero, zero)[0])                          # (n, 3)
        total.append(sim.probs(rng.normal(0, sigma, (n_shocks, n)), rng.normal(0, sigma, (n_shocks, n))))
    mo = np.stack(model_only)                                                 # (B, n, 3)
    tt = np.concatenate(total, axis=0)                                        # (B*S, n, 3)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    out = pd.DataFrame({"p_away": mo[..., 0].mean(0), "p_draw": mo[..., 1].mean(0),
                        "p_home": mo[..., 2].mean(0)})
    ep_h = lambda P: 3 * P[..., 2] + P[..., 1]  # noqa: E731
    ep_a = lambda P: 3 * P[..., 0] + P[..., 1]  # noqa: E731
    out["exp_pts_home"], out["exp_pts_away"] = ep_h(mo).mean(0), ep_a(mo).mean(0)
    for tag, P in (("model", mo), ("total", tt)):
        for name, k in (("home", 2), ("draw", 1), ("away", 0)):
            out[f"p_{name}_{tag}_lo"], out[f"p_{name}_{tag}_hi"] = np.percentile(P[..., k], [lo, hi], axis=0)
        out[f"exp_pts_home_{tag}_lo"], out[f"exp_pts_home_{tag}_hi"] = np.percentile(ep_h(P), [lo, hi], axis=0)
        out[f"exp_pts_away_{tag}_lo"], out[f"exp_pts_away_{tag}_hi"] = np.percentile(ep_a(P), [lo, hi], axis=0)
    return out
