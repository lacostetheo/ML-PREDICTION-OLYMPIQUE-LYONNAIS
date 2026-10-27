"""Monte Carlo simulation of the rest of a league season.

Each simulated season:
 1. every team's rating is shaken by N(0, sigma) Elo points (uncertainty about true
    strength over the rest of the season; the Elo features of every remaining fixture
    move accordingly and the probabilities are recomputed exactly);
 2. the outcome of every remaining match is drawn from the model's win/draw/loss
    probabilities;
 3. a scoreline consistent with that outcome is drawn from the Poisson goal model
    (needed for goal-difference tie-breaks);
 4. the table is ranked on points, goal difference, goals scored (random last resort).
The logistic model is linear, so changing the Elo columns only changes the logits
additively: that is why the 10,000 simulations are cheap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from olpred.predict.state import HFA

ELO_NAMES = ["elo_home", "elo_away", "elo_diff", "elo_exp_home"]
G = 11  # scores 0..10


def remaining_pairs(played_season: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    """Ordered pairs (home, away) not yet played: in a double round robin each pair
    meets once at each venue."""
    done = set(zip(played_season["home"], played_season["away"]))
    return pd.DataFrame([(h, a) for h in teams for a in teams if h != a and (h, a) not in done],
                        columns=["home", "away"])


class LinearSim:
    """Vectorised class probabilities of a fitted ``LogisticModel`` under Elo shocks."""

    def __init__(self, model, X: pd.DataFrame):
        prep, clf = model.pipe_[:-1], model.pipe_[-1]
        names = list(prep.get_feature_names_out(model.cols))
        self.W, self.b = clf.coef_, clf.intercept_
        self.Z0 = prep.transform(X[model.cols])
        self.logit0 = self.Z0 @ self.W.T + self.b
        scaler = prep[-1]
        self.idx = [names.index(n) for n in ELO_NAMES]
        self.mu, self.sd = scaler.mean_[self.idx], scaler.scale_[self.idx]
        self.eh, self.ea = X["elo_home"].to_numpy(), X["elo_away"].to_numpy()

    def probs(self, shock_h: np.ndarray, shock_a: np.ndarray) -> np.ndarray:
        """shock_h, shock_a: (S, F) Elo shocks of the home / away team of each fixture."""
        eh, ea = self.eh[None] + shock_h, self.ea[None] + shock_a
        diff = eh - ea
        cols = [eh, ea, diff, 1.0 / (1.0 + 10 ** (-(diff + HFA) / 400.0))]
        logit = np.broadcast_to(self.logit0[None], (*eh.shape, 3)).copy()
        for c, x, j in zip(self.idx, cols, range(4)):
            dz = (x - self.mu[j]) / self.sd[j] - self.Z0[:, c][None]
            logit += dz[..., None] * self.W[:, c][None, None, :]
        logit -= logit.max(axis=-1, keepdims=True)
        p = np.exp(logit)
        return p / p.sum(axis=-1, keepdims=True)


def conditional_score_cdfs(score_matrix: np.ndarray) -> np.ndarray:
    """(F, 3, G*G) cumulative distributions of the scoreline given the outcome
    (0 = away win, 1 = draw, 2 = home win)."""
    i, j = np.indices(score_matrix.shape[1:])
    out = np.empty((score_matrix.shape[0], 3, G * G))
    for k, mask in enumerate([i < j, i == j, i > j]):
        pmf = (score_matrix * mask).reshape(len(score_matrix), -1)
        pmf = pmf / np.clip(pmf.sum(axis=1, keepdims=True), 1e-15, None)
        out[:, k] = np.cumsum(pmf, axis=1)
    return out


@dataclass
class SeasonInput:
    teams: list[str]
    table_now: pd.DataFrame          # index team; columns pts, gd, gf, played
    fixtures: pd.DataFrame           # columns home, away
    sim: LinearSim
    score_cdfs: np.ndarray


def simulate_season(inp: SeasonInput, n_sims: int = 10_000, sigma: float = 30.0, seed: int = 0,
                    chunk: int = 1000) -> dict[str, np.ndarray]:
    """Returns ``positions`` (n_sims, T; 1 = champion) and final ``points`` (n_sims, T)."""
    rng = np.random.default_rng(seed)
    T, F = len(inp.teams), len(inp.fixtures)
    t_index = {t: i for i, t in enumerate(inp.teams)}
    hi = inp.fixtures["home"].map(t_index).to_numpy()
    ai = inp.fixtures["away"].map(t_index).to_numpy()
    Hm, Am = np.zeros((F, T)), np.zeros((F, T))
    Hm[np.arange(F), hi] = 1
    Am[np.arange(F), ai] = 1
    base = inp.table_now.reindex(inp.teams)
    pts0, gd0, gf0 = (base[c].to_numpy(float) for c in ("pts", "gd", "gf"))

    pos_all, pts_all = [], []
    for start in range(0, n_sims, chunk):
        S = min(chunk, n_sims - start)
        shock = rng.normal(0.0, sigma, size=(S, T)) if sigma > 0 else np.zeros((S, T))
        P = inp.sim.probs(shock[:, hi], shock[:, ai])               # (S, F, 3)
        u = rng.random((S, F))
        cum = P.cumsum(axis=-1)
        outcome = (u[..., None] > cum[..., :2]).sum(axis=-1)         # 0 away, 1 draw, 2 home
        v = rng.random((S, F))
        gh, ga = np.zeros((S, F)), np.zeros((S, F))
        for f in range(F):                                           # scoreline given outcome
            for k in range(3):
                rows = np.where(outcome[:, f] == k)[0]
                if rows.size:
                    cell = np.searchsorted(inp.score_cdfs[f, k], v[rows, f])
                    cell = np.minimum(cell, G * G - 1)
                    gh[rows, f], ga[rows, f] = cell // G, cell % G
        ph = np.select([outcome == 2, outcome == 1], [3, 1], 0)
        pa = np.select([outcome == 0, outcome == 1], [3, 1], 0)
        pts = pts0 + ph @ Hm + pa @ Am
        gd = gd0 + (gh - ga) @ Hm + (ga - gh) @ Am
        gf = gf0 + gh @ Hm + ga @ Am
        key = pts * 1e6 + (gd + 1000) * 1e3 + gf + rng.random((S, T))
        rank = np.argsort(np.argsort(-key, axis=1), axis=1) + 1
        pos_all.append(rank.astype(np.int16)); pts_all.append(pts)
    return {"positions": np.vstack(pos_all), "points": np.vstack(pts_all)}


def summarise(res: dict[str, np.ndarray], teams: list[str], spots: dict[str, int]) -> pd.DataFrame:
    """Per-team probabilities: title, top-k spots, relegation; expected points and 80 % interval."""
    pos, pts = res["positions"], res["points"]
    T = len(teams)
    out = pd.DataFrame({"team": teams,
                        "exp_points": pts.mean(0), "pts_p10": np.percentile(pts, 10, axis=0),
                        "pts_p90": np.percentile(pts, 90, axis=0),
                        "exp_position": pos.mean(0), "p_title": (pos == 1).mean(0)})
    for name, k in spots.items():
        if name.startswith("relegation"):
            out[f"p_{name}"] = (pos > T - k).mean(0)
        else:
            out[f"p_{name}"] = (pos <= k).mean(0)
    return out.set_index("team")
