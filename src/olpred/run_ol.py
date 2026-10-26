"""Phase 8: Olympique Lyonnais — current strength and match predictions.

Production model: the final logistic regression refit on every played match
(2011 -> latest); state of all teams on ``as_of``; predictions for the next round
(with the bookmaker's prices alongside) and for every remaining OL fixture, with
uncertainty intervals.   Run:  python -m olpred.run_ol [YYYY-MM-DD]
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from olpred.data import transfermarkt as tm
from olpred.data.football_data import download_fixtures, load_fixtures
from olpred.features.build import squad_resources
from olpred.features.columns import curated_cols
from olpred.features.squad_value import POSITIONS, position_profile
from olpred.models.baselines import LogisticModel
from olpred.predict.league import fit_poisson, selected_sigma
from olpred.predict.simulate import remaining_pairs
from olpred.predict.state import fixture_frame, team_state
from olpred.predict.uncertainty import bootstrap_models, match_intervals

TABLES = Path("reports/tables")
STORE = Path("models_store")
OL, LEAGUE = "Lyon", "F1"
N_BOOT = 100


def market_probs(df: pd.DataFrame) -> pd.DataFrame:
    inv = 1.0 / df[["B365A", "B365D", "B365H"]].to_numpy()
    p = inv / inv.sum(axis=1, keepdims=True)
    return pd.DataFrame(p, columns=["mkt_away", "mkt_draw", "mkt_home"], index=df.index)


def percentile_rank(s: pd.Series, team: str, higher_is_better: bool = True) -> tuple[float, int, int]:
    s = s.dropna()
    rank = int(s.rank(ascending=not higher_is_better, method="min")[team])
    return 100.0 * (len(s) - rank) / max(len(s) - 1, 1), rank, len(s)


def strength_profile(state: pd.DataFrame, matches: pd.DataFrame, club_map: pd.DataFrame,
                     as_of: pd.Timestamp) -> pd.DataFrame:
    """OL's standing among the league's clubs (rank 1 = best) on each dimension."""
    lg = state.loc[LEAGUE]
    dims = {"Elo rating": (lg["elo"], True),
            "Recent form: points/match, last 10": (lg["h_pts_10"], True),
            "Recent attack: goals scored, last 10": (lg["h_gf_10"], True),
            "Recent attack: shots on target, last 10": (lg["h_sot_10"], True),
            "Recent defence: goals conceded, last 10": (lg["h_ga_10"], False),
            "Recent defence: shots on target against, last 10": (lg["h_sot_against_10"], False)}
    rows = []
    for name, (s, hib) in dims.items():
        pct, rank, n = percentile_rank(s, OL, hib)
        rows.append({"dimension": name, "value": float(s[OL]), "rank": rank, "of": n, "percentile": pct})
    # position-level market values (descriptive)
    season_teams = sorted(lg.index)
    ids = club_map[(club_map["league"] == LEAGUE) & club_map["team"].isin(season_teams)]
    prof = position_profile(tm.load("player_valuations"), tm.load("players"), set(ids["club_id"]), as_of)
    by_team = prof.join(ids.set_index("club_id")["team"]).set_index("team")
    labels = {"Attack": "Squad value: attackers", "Midfield": "Squad value: midfielders",
              "Defender": "Squad value: defenders", "Goalkeeper": "Squad value: goalkeepers",
              "Depth (players 12-22)": "Squad depth (value of players 12-22)", "Total": "Squad value: total"}
    for col, name in labels.items():
        pct, rank, n = percentile_rank(by_team[col], OL, True)
        rows.append({"dimension": name, "value": float(by_team.loc[OL, col]) / 1e6, "rank": rank,
                     "of": n, "percentile": pct})
    return pd.DataFrame(rows)


def main() -> None:
    as_of = pd.Timestamp(sys.argv[1]) if len(sys.argv) > 1 else pd.Timestamp.today().normalize()
    TABLES.mkdir(parents=True, exist_ok=True); STORE.mkdir(exist_ok=True)
    matches = pd.read_parquet("data/processed/matches.parquet")
    feats = pd.read_parquet("data/processed/features.parquet")
    cols = curated_cols(feats)
    train = feats[feats["season"] >= 2011]
    last_played = matches["date"].max()
    print(f"as_of {as_of.date()}, last played match in data {last_played.date()}", flush=True)

    club_map, snaps = squad_resources(matches, end=(as_of + pd.offsets.MonthBegin(1)).strftime("%Y-%m-01"))
    lr = LogisticModel(cols).fit(train)
    poisson = fit_poisson(train)
    joblib.dump({"logreg": lr, "poisson": poisson, "as_of": as_of, "cols": cols}, STORE / "production.joblib")
    state = team_state(matches, as_of, club_map, snaps)
    state.to_parquet(Path("data/processed") / "team_state.parquet")
    sigma = selected_sigma()
    print("Elo-uncertainty sigma from backtest:", sigma, flush=True)

    boot = bootstrap_models(train, cols, lr.C_, n_boot=N_BOOT)
    joblib.dump({"models": boot, "sigma": sigma}, STORE / "bootstrap_logreg.joblib")  # used by the dashboard

    # 1. next round (real fixture list, with bookmaker prices)
    try:
        download_fixtures()
    except Exception as exc:  # noqa: BLE001
        print("offline, using cached fixtures:", exc)
    fx = load_fixtures()
    nxt = fx[(fx["league"] == LEAGUE) & (fx["date"] >= as_of - pd.Timedelta(days=3))].reset_index(drop=True)
    X = fixture_frame(nxt, state)
    res = match_intervals(boot, X, sigma)
    out = pd.concat([nxt[["date", "home", "away"]], res, market_probs(nxt)], axis=1)
    out["involves_OL"] = (out["home"] == OL) | (out["away"] == OL)
    out.round(4).to_csv(TABLES / "ol_next_round.csv", index=False)

    # 2. every remaining OL fixture (order unknown): today's state, neutral schedule
    cur = matches[(matches["season"] == matches["season"].max()) & (matches["league"] == LEAGUE)]
    teams = sorted(set(cur["home"]) | set(cur["away"]))
    rem = remaining_pairs(cur, teams)
    rem = rem[(rem["home"] == OL) | (rem["away"] == OL)].assign(league=LEAGUE).reset_index(drop=True)
    Xr = fixture_frame(rem, state, neutral_schedule=True)
    rr = match_intervals(boot, Xr, sigma)
    rem_out = pd.concat([rem[["home", "away"]], rr], axis=1)
    rem_out["venue"] = np.where(rem_out["home"] == OL, "home", "away")
    rem_out["opponent"] = np.where(rem_out["home"] == OL, rem_out["away"], rem_out["home"])
    rem_out["ol_win"] = np.where(rem_out["venue"] == "home", rem_out["p_home"], rem_out["p_away"])
    rem_out["ol_draw"] = rem_out["p_draw"]
    rem_out["ol_loss"] = np.where(rem_out["venue"] == "home", rem_out["p_away"], rem_out["p_home"])
    rem_out["ol_exp_pts"] = np.where(rem_out["venue"] == "home", rem_out["exp_pts_home"], rem_out["exp_pts_away"])
    for tag in ("model", "total"):
        for side in ("lo", "hi"):
            rem_out[f"ol_exp_pts_{tag}_{side}"] = np.where(rem_out["venue"] == "home",
                                                           rem_out[f"exp_pts_home_{tag}_{side}"],
                                                           rem_out[f"exp_pts_away_{tag}_{side}"])
    rem_out.round(4).to_csv(TABLES / "ol_remaining_fixtures.csv", index=False)

    # 3. current team strength profile
    prof = strength_profile(state, matches, club_map, as_of)
    prof.round(3).to_csv(TABLES / "ol_strength_profile.csv", index=False)

    pd.set_option("display.width", 220)
    print(out[["date", "home", "away", "p_home", "p_draw", "p_away", "mkt_home", "mkt_draw", "mkt_away"]].round(3).to_string())
    print(prof.round(2).to_string())
    print(rem_out[["opponent", "venue", "ol_win", "ol_draw", "ol_loss", "ol_exp_pts"]].round(3).to_string())


if __name__ == "__main__":
    main()
