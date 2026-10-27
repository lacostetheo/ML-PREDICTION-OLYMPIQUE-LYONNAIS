"""Phase 9: Monte Carlo simulation of the rest of the season, all five leagues.

Uses the production logistic regression and the team state saved by olpred.run_ol.
Run:  python -m olpred.run_season_sim [n_sims]
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from olpred.predict.league import LEAGUES, fit_poisson, league_input, selected_sigma, spots_for, table_from
from olpred.predict.simulate import simulate_season, summarise

TABLES = Path("reports/tables")
PROCESSED = Path("data/processed")


def main() -> None:
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    prod = joblib.load(Path("models_store") / "production.joblib")
    matches = pd.read_parquet(PROCESSED / "matches.parquet")
    feats = pd.read_parquet(PROCESSED / "features.parquet")
    state = pd.read_parquet(PROCESSED / "team_state.parquet")
    sigma = selected_sigma()
    poisson = fit_poisson(feats[feats["season"] >= 2011])
    season = int(matches["season"].max())
    print(f"{n_sims} simulations per league, Elo sigma {sigma}, state {prod['as_of'].date()}", flush=True)

    for league, cfg in LEAGUES.items():
        played = matches[(matches["season"] == season) & (matches["league"] == league)]
        inp = league_input(league, played, state, prod["logreg"], poisson)
        res = simulate_season(inp, n_sims=n_sims, sigma=sigma, seed=2026)
        summary = summarise(res, inp.teams, spots_for(league))
        summary = summary.join(inp.table_now[["pts", "played"]].rename(columns={"pts": "points_now",
                                                                                 "played": "matches_played"}))
        summary.insert(0, "league", league)
        summary.sort_values("exp_position").round(4).to_csv(TABLES / f"sim_{league}_summary.csv")
        T = len(inp.teams)
        pos_probs = pd.DataFrame({k: (res["positions"] == k).mean(axis=0) for k in range(1, T + 1)},
                                 index=inp.teams)
        pos_probs.round(4).to_csv(TABLES / f"sim_{league}_position_probs.csv")
        np.savez_compressed(PROCESSED / f"sim_{league}.npz", positions=res["positions"], points=res["points"],
                            teams=np.array(inp.teams))
        print(f"\n== {cfg['name']} ({len(inp.fixtures)} matches left)")
        print(summary.sort_values("exp_position")[["points_now", "exp_points", "pts_p10", "pts_p90",
                                                   "p_title", "p_top4", "p_top6", "p_relegation"]].round(3).to_string())
        if league == "F1":  # OL points distribution
            d = pd.Series(res["points"][:, inp.teams.index("Lyon")]).value_counts(normalize=True).sort_index()
            d.rename("probability").rename_axis("final_points").round(5).to_csv(TABLES / "sim_ol_points_distribution.csv")


if __name__ == "__main__":
    main()
