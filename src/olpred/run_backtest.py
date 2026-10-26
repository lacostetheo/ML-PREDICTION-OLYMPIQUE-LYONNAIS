"""Backtest of the season simulator.

For each past season, simulate the rest of the season from 31 October using only what
was known then (data before that date, models fitted on earlier seasons), then compare
with the real final tables. Tuning seasons (2020-2022) choose the rating-uncertainty
sigma; evaluation seasons (2023-2025, the test period) are scored with that sigma.
Run:  python -m olpred.run_backtest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from olpred.features.build import squad_resources
from olpred.features.columns import curated_cols
from olpred.models.baselines import LogisticModel
from olpred.predict.league import LEAGUES, choose_sigma, fit_poisson, league_input, spots_for, table_from
from olpred.predict.simulate import simulate_season, summarise
from olpred.predict.state import team_state

TABLES = Path("reports/tables")
SIGMAS = [0.0, 15.0, 30.0, 45.0, 60.0]
TUNE_SEASONS, EVAL_SEASONS = [2020, 2021, 2022], [2023, 2024, 2025]
N_SIMS = 2000


def final_ranking(season_matches: pd.DataFrame) -> pd.DataFrame:
    t = table_from(season_matches)
    t["key"] = t["pts"] * 1e6 + (t["gd"] + 1000) * 1e3 + t["gf"]
    t["final_pos"] = t["key"].rank(ascending=False, method="first").astype(int)
    return t


def run_season(season, matches, feats, club_map, snaps, sigmas) -> pd.DataFrame:
    as_of = pd.Timestamp(f"{season}-10-31")
    played = matches[matches["date"] < as_of]
    train = feats[(feats["season"] >= 2011) & (feats["season"] < season)]
    lr = LogisticModel(curated_cols(feats)).fit(train)
    poisson = fit_poisson(train)
    state = team_state(played, as_of, club_map, snaps)
    rows = []
    for league in LEAGUES:
        sm = matches[(matches["season"] == season) & (matches["league"] == league)]
        ps = played[(played["season"] == season) & (played["league"] == league)]
        inp = league_input(league, ps, state, lr, poisson)
        assert len(inp.fixtures) == len(sm) - len(ps), "remaining pairs differ from the real calendar"
        actual = final_ranking(sm)
        n = len(inp.teams)
        for sigma in sigmas:
            res = simulate_season(inp, n_sims=N_SIMS, sigma=sigma, seed=season)
            s = summarise(res, inp.teams, spots_for(league))
            s = s.join(actual[["pts", "final_pos"]].rename(columns={"pts": "actual_points"}))
            s["now_points"] = inp.table_now["pts"].reindex(s.index)
            s["played"] = inp.table_now["played"].reindex(s.index)
            s["ppg_forecast"] = s["now_points"] / s["played"] * (2 * (n - 1))
            s["inside_80"] = (s["actual_points"] >= s["pts_p10"]) & (s["actual_points"] <= s["pts_p90"])
            for name, k in spots_for(league).items():
                s[f"o_{name}"] = (s["final_pos"] > n - k) if name == "relegation" else (s["final_pos"] <= k)
            s["o_title"] = s["final_pos"] == 1
            s["league"], s["season"], s["sigma"], s["n_teams"] = league, season, sigma, n
            rows.append(s.reset_index())
        print(season, league, "done", flush=True)
    return pd.concat(rows, ignore_index=True)


def score(df: pd.DataFrame) -> pd.Series:
    out = {"mae_points": (df["exp_points"] - df["actual_points"]).abs().mean(),
           "rmse_points": float(np.sqrt(((df["exp_points"] - df["actual_points"]) ** 2).mean())),
           "mae_points_ppg_baseline": (df["ppg_forecast"] - df["actual_points"]).abs().mean(),
           "coverage_80": df["inside_80"].mean()}
    for name in ["title", "top4", "top6", "europe", "relegation", "champions_league"]:
        p, o = df[f"p_{name}"], df[f"o_{name}"].astype(float)
        base = df.groupby("league")[f"o_{name}"].transform("mean")  # climatological rate (spots / teams)
        out[f"brier_{name}"] = ((p - o) ** 2).mean()
        out[f"brier_{name}_climatology"] = ((base - o) ** 2).mean()
    return pd.Series(out)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    matches = pd.read_parquet("data/processed/matches.parquet")
    feats = pd.read_parquet("data/processed/features.parquet")
    club_map, snaps = squad_resources(matches, end="2026-12-01")
    cache = TABLES / "backtest_sigma_selection.csv"
    if "--reuse-tuning" in sys.argv and cache.exists():
        sel = pd.read_csv(cache, index_col=0)
    else:
        tune = pd.concat([run_season(s, matches, feats, club_map, snaps, SIGMAS) for s in TUNE_SEASONS])
        sel = tune.groupby("sigma").apply(score, include_groups=False)
        sel.round(4).to_csv(cache)
    sigma = choose_sigma(sel)
    print(sel.round(4).T.to_string()); print("selected sigma:", sigma, flush=True)

    ev = pd.concat([run_season(s, matches, feats, club_map, snaps, [sigma]) for s in EVAL_SEASONS])
    ev.to_parquet(TABLES / "backtest_eval_teams.parquet")
    by_season = ev.groupby("season").apply(score, include_groups=False)
    total = score(ev).rename("all test seasons").to_frame().T
    pd.concat([by_season, total]).round(4).to_csv(TABLES / "backtest_eval.csv")
    pd.set_option("display.width", 220)
    print(pd.concat([by_season, total]).round(4).T.to_string())


if __name__ == "__main__":
    main()
