"""Glue between the data tables and the season simulator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from olpred.models.baselines import PoissonGoalModel
from olpred.predict.simulate import LinearSim, SeasonInput, conditional_score_cdfs, remaining_pairs
from olpred.predict.state import fixture_frame

ROOT = Path(__file__).resolve().parents[3]
LEAGUES = {k: v for k, v in json.loads((ROOT / "configs" / "leagues.json").read_text()).items()
           if not k.startswith("_")}
POISSON_COLS = (["h_gf_10", "a_ga_10", "elo_diff"], ["a_gf_10", "h_ga_10", "elo_diff"])


def spots_for(league: str) -> dict[str, int]:
    c = LEAGUES[league]
    return {"top4": 4, "top6": 6, "champions_league": c["champions_league"], "europe": c["europe"],
            "relegation": c["relegation"]}


def table_from(played_season: pd.DataFrame) -> pd.DataFrame:
    """Points, goal difference, goals for and matches played per team."""
    h = pd.DataFrame({"team": played_season["home"], "pts": played_season["pts_home"],
                      "gf": played_season["home_goals"], "ga": played_season["away_goals"]})
    a = pd.DataFrame({"team": played_season["away"], "pts": played_season["pts_away"],
                      "gf": played_season["away_goals"], "ga": played_season["home_goals"]})
    t = pd.concat([h, a]).groupby("team").agg(pts=("pts", "sum"), gf=("gf", "sum"), ga=("ga", "sum"),
                                              played=("pts", "size"))
    t["gd"] = t["gf"] - t["ga"]
    return t


def league_input(league: str, played_season: pd.DataFrame, state: pd.DataFrame, lr_model,
                 poisson: PoissonGoalModel, fixtures: pd.DataFrame | None = None) -> SeasonInput:
    """Everything the simulator needs for one league on the state date."""
    teams = sorted(set(played_season["home"]) | set(played_season["away"]))
    fx = (fixtures if fixtures is not None else remaining_pairs(played_season, teams)).copy()
    fx["league"] = league
    X = fixture_frame(fx, state, neutral_schedule=True)
    sim = LinearSim(lr_model, X)
    cdfs = conditional_score_cdfs(poisson.score_matrix(X))
    return SeasonInput(teams=teams, table_now=table_from(played_season), fixtures=fx[["home", "away"]],
                       sim=sim, score_cdfs=cdfs)


def fit_poisson(train: pd.DataFrame) -> PoissonGoalModel:
    return PoissonGoalModel(*POISSON_COLS).fit(train)


def choose_sigma(sel: pd.DataFrame) -> float:
    """Rating-uncertainty sigma from the tuning-season backtest table (index = sigma):
    lowest mean Brier of title/top-4/relegation probabilities, with a small penalty for
    80 % intervals whose coverage is far from 0.8."""
    crit = (sel[["brier_title", "brier_top4", "brier_relegation"]].mean(axis=1)
            + (sel["coverage_80"] - 0.8).abs() * 0.2)
    return float(crit.idxmin())


def selected_sigma(tables: Path = ROOT / "reports" / "tables") -> float:
    return choose_sigma(pd.read_csv(tables / "backtest_sigma_selection.csv", index_col=0))
