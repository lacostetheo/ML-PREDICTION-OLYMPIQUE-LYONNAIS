"""Team "state" at an arbitrary date, and features for future fixtures.

To score a match that has not been played, its features must be computed from the
history only. We reuse the *exact* training pipeline (``build_features``) by adding
phantom matches on the ``as_of`` date: each team appears in one phantom match (as the
home side in set H, as the away side in set A), with no result. Because every feature
is built from strictly earlier matches, a phantom row's features are precisely the
team's pre-match features on ``as_of``. A fixture "home h vs away a" then takes the
home-side features of h from set H and the away-side features of a from set A.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from olpred.features.build import build_features
from olpred.features.squad_value import squad_features

HFA = 65.0  # Elo home advantage, same as olpred.features.elo.compute_elo
NEUTRAL_REST, NEUTRAL_CONGESTION = 7.0, 1.5


def _phantoms(played: pd.DataFrame, as_of: pd.Timestamp, reverse: bool) -> pd.DataFrame:
    season = int(played["season"].max())
    rows, next_id = [], int(played["match_id"].max()) + 1
    cur = played[played["season"] == season]
    for league, g in cur.groupby("league"):
        teams = sorted(set(g["home"]) | set(g["away"]))
        if len(teams) % 2:
            teams = teams[:-1]
        for a, b in zip(teams[0::2], teams[1::2]):
            h, aw = (b, a) if reverse else (a, b)
            rows.append({"match_id": next_id, "date": as_of, "season": season, "league": league,
                         "home": h, "away": aw, "home_goals": 0.0, "away_goals": 0.0,
                         "result": "D", "y": 1, "pts_home": 1, "pts_away": 1})
            next_id += 1
    ph = pd.DataFrame(rows)
    for c in played.columns:
        if c not in ph:
            ph[c] = np.nan
    return ph[played.columns]


def team_state(played: pd.DataFrame, as_of: pd.Timestamp, club_map: pd.DataFrame, snaps: pd.DataFrame
               ) -> pd.DataFrame:
    """Pre-match features of every team of the current season on ``as_of``.

    Returns a frame indexed by (league, team) with columns ``h_*`` (team playing at home),
    ``a_*`` (team playing away), ``elo`` (rating)."""
    played = played[played["date"] < as_of].reset_index(drop=True)
    home_rows, away_rows = [], []
    for reverse in (False, True):
        # each pass: a team is in exactly one phantom match, at home or away; the two
        # passes swap the venues, so every team gets both its home-side and away-side features
        aug = pd.concat([played, _phantoms(played, as_of, reverse)], ignore_index=True)
        feats = build_features(aug, squad_features(aug, club_map, snaps))
        ph = feats[feats["match_id"] > played["match_id"].max()]
        h_cols = [c for c in ph.columns if c.startswith("h_")] + ["elo_home"]
        a_cols = [c for c in ph.columns if c.startswith("a_")] + ["elo_away"]
        home_rows.append(ph.set_index(["league", "home"])[h_cols].rename_axis(["league", "team"]))
        away_rows.append(ph.set_index(["league", "away"])[a_cols].rename_axis(["league", "team"]))
    st = pd.concat(home_rows).join(pd.concat(away_rows), how="inner")
    st["elo"] = st["elo_home"]
    return st.drop(columns=["elo_home", "elo_away"])


def fixture_frame(fixtures: pd.DataFrame, state: pd.DataFrame, neutral_schedule: bool = False) -> pd.DataFrame:
    """Feature matrix for fixtures (columns league, home, away) from a team state."""
    h = state.loc[list(zip(fixtures["league"], fixtures["home"]))].reset_index(drop=True)
    a = state.loc[list(zip(fixtures["league"], fixtures["away"]))].reset_index(drop=True)
    out = fixtures[["league", "home", "away"]].reset_index(drop=True).copy()
    base = [c[2:] for c in state.columns if c.startswith("h_")]
    for f in base:
        out[f"h_{f}"] = h[f"h_{f}"].to_numpy()
        out[f"a_{f}"] = a[f"a_{f}"].to_numpy()
    if neutral_schedule:
        for f, v in (("rest_days", NEUTRAL_REST), ("matches_14d", NEUTRAL_CONGESTION)):
            out[f"h_{f}"], out[f"a_{f}"] = v, v
    for f in base:
        if f not in ("n_prior", "games_played_season", "sv_n"):
            out[f"d_{f}"] = out[f"h_{f}"] - out[f"a_{f}"]
    out["elo_home"], out["elo_away"] = h["elo"].to_numpy(), a["elo"].to_numpy()
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    out["elo_exp_home"] = 1.0 / (1.0 + 10 ** (-(out["elo_diff"] + HFA) / 400.0))
    return out
