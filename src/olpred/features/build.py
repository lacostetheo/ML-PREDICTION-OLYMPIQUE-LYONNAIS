"""Assemble the model-ready, leakage-free match-level feature table.

Run:  python -m olpred.features.build
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from olpred.data import transfermarkt as tm
from olpred.data.clean import build_matches
from olpred.data.football_data import load_raw
from olpred.features.elo import compute_elo
from olpred.features.form import ROLL_STATS, VENUE_STATS, VENUE_WINDOW, WINDOWS, add_form, add_table, \
    league_prior, to_long
from olpred.features.squad_value import build_club_map, monthly_snapshots, squad_features
from olpred.features.xg_proxy import XGProxy

XG_TRAIN_END = 2019  # xG-proxy coefficients fitted on seasons <= this only
PROCESSED = Path("data/processed")

TEAM_FEATURES = (
    [f"{s}_{w}" for s in ROLL_STATS for w in WINDOWS]
    + ["gd_5"] + [f"{s}_venue_{VENUE_WINDOW}" for s in VENUE_STATS]
    + ["rest_days", "matches_14d", "ppg_season", "gdpg_season", "pos_norm",
       "games_played_season", "n_prior"]
)
SQUAD_FEATURES = ["sv_total_rel", "sv_top11_rel", "sv_n"]
META = ["match_id", "date", "season", "league", "home", "away", "home_goals", "away_goals",
        "result", "y", "pts_home", "pts_away",
        "b365_h", "b365_d", "b365_a", "avg_h", "avg_d", "avg_a"]


def build_features(matches: pd.DataFrame, squad: pd.DataFrame | None = None,
                   xg_train_end: int = XG_TRAIN_END) -> pd.DataFrame:
    """One row per match: metadata, targets and home/away/difference features."""
    long = to_long(matches)

    train = long[long["season"] <= xg_train_end]
    proxy = XGProxy().fit(train["shots"], train["sot"], train["gf"])
    long["xg"] = proxy.predict(long["shots"], long["sot"])
    long["xga"] = proxy.predict(long["shots_against"], long["sot_against"])

    long = add_form(long)
    long = add_table(long)
    for col in ["ppg_season", "gdpg_season"]:  # season start: fall back to league prior
        stat = "pts" if col == "ppg_season" else "gd"
        long[col] = long[col].fillna(league_prior(long.assign(gd=long.gf - long.ga), stat))
    long["pos_norm"] = long["pos_norm"].fillna(0.5)

    feats = list(TEAM_FEATURES)
    if squad is not None:
        long["month"] = long["date"].dt.to_period("M").dt.to_timestamp()
        long = long.merge(squad, on=["league", "team", "month"], how="left")
        feats += SQUAD_FEATURES

    home = long[long["is_home"] == 1].set_index("match_id")[feats].add_prefix("h_")
    away = long[long["is_home"] == 0].set_index("match_id")[feats].add_prefix("a_")
    wide = home.join(away)
    for f in feats:
        if f not in ("n_prior", "games_played_season", "sv_n"):
            wide[f"d_{f}"] = wide[f"h_{f}"] - wide[f"a_{f}"]

    elo = compute_elo(matches).set_index("match_id")
    elo["elo_diff"] = elo["elo_home"] - elo["elo_away"]

    out = matches[META].set_index("match_id").join([elo, wide]).reset_index()
    return out.sort_values(["date", "match_id"]).reset_index(drop=True)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    matches = build_matches(load_raw())
    matches.to_parquet(PROCESSED / "matches.parquet")

    tm.download()
    club_map = build_club_map(matches, tm.load("games"))
    club_map.to_csv(PROCESSED / "club_map.csv", index=False)
    snaps = monthly_snapshots(tm.load("player_valuations"), set(club_map["club_id"]),
                              start=f"{matches['season'].min() - 1}-07-01",
                              end=matches["date"].max().strftime("%Y-%m-01"))
    squad = squad_features(matches, club_map, snaps)

    features = build_features(matches, squad)
    features.to_parquet(PROCESSED / "features.parquet")
    print(f"matches: {len(matches)}  features table: {features.shape}")


if __name__ == "__main__":
    main()
