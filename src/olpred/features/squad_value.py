"""Squad market-value features from the Transfermarkt dataset.

Leakage control: the squad value used for a match on date T comes from a
monthly snapshot G = first day of T's month, built only from valuations dated
strictly before G (<= T). Each player contributes his latest valuation of the
12 months before G, attributed to the club recorded on that valuation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TM_COMP = {"E0": "GB1", "SP1": "ES1", "I1": "IT1", "D1": "L1", "F1": "FR1"}
WINDOW_DAYS = 365
MIN_VALUED_PLAYERS = 11


def build_club_map(matches: pd.DataFrame, tm_games: pd.DataFrame) -> pd.DataFrame:
    """Map football-data (league, team) -> Transfermarkt club_id.

    Join both match tables on (competition, date, score); keep only matches
    unambiguous on both sides; majority vote per team. Teams with no match in the
    Transfermarkt data (it only covers top-flight clubs, from 2012) stay unmapped
    and get no squad value; a fuzzy name fallback was tried and rejected
    (it mapped "Arles" to "RC Lens").
    """
    fd = matches.assign(comp=matches["league"].map(TM_COMP))
    tm = tm_games.assign(date=pd.to_datetime(tm_games["date"]))
    key = ["comp", "date", "home_goals", "away_goals"]
    tm = tm.rename(columns={"competition_id": "comp", "home_club_goals": "home_goals",
                            "away_club_goals": "away_goals"})
    fd_u = fd[~fd.duplicated(key, keep=False)]
    tm_u = tm[~tm.duplicated(key, keep=False)]
    j = fd_u.merge(tm_u[key + ["home_club_id", "away_club_id"]], on=key)
    votes = pd.concat([
        j[["league", "home", "home_club_id"]].set_axis(["league", "team", "club_id"], axis=1),
        j[["league", "away", "away_club_id"]].set_axis(["league", "team", "club_id"], axis=1)])
    counts = votes.groupby(["league", "team", "club_id"]).size().rename("votes").reset_index()
    best = counts.sort_values("votes", ascending=False).drop_duplicates(["league", "team"])

    return best[["league", "team", "club_id", "votes"]].reset_index(drop=True)


def monthly_snapshots(valuations: pd.DataFrame, club_ids: set[int], start: str, end: str
                      ) -> pd.DataFrame:
    """Per club and month start G: total value, mean of top-11 values, #players."""
    v = valuations[valuations["current_club_id"].isin(club_ids)].copy()
    v["date"] = pd.to_datetime(v["date"])
    v = v.dropna(subset=["market_value_in_eur"]).sort_values("date")
    rows = []
    for g in pd.date_range(start, end, freq="MS"):
        w = v[(v["date"] < g) & (v["date"] >= g - pd.Timedelta(days=WINDOW_DAYS))]
        w = w.groupby("player_id").tail(1)  # latest valuation per player before G
        for club, vals in w.groupby("current_club_id")["market_value_in_eur"]:
            top = np.sort(vals.to_numpy())[::-1][:11]
            rows.append((club, g, vals.sum(), top.mean(), len(vals)))
    return pd.DataFrame(rows, columns=["club_id", "month", "sv_total", "sv_top11", "sv_n"])


def squad_features(matches: pd.DataFrame, club_map: pd.DataFrame, snaps: pd.DataFrame
                   ) -> pd.DataFrame:
    """Per (league, team, month) log squad value relative to the league mean
    in that month (removes market-wide value inflation)."""
    tm = pd.concat([
        matches[["league", "home", "date"]].set_axis(["league", "team", "date"], axis=1),
        matches[["league", "away", "date"]].set_axis(["league", "team", "date"], axis=1)])
    tm["month"] = tm["date"].dt.to_period("M").dt.to_timestamp()
    tm = tm[["league", "team", "month"]].drop_duplicates()
    tm = tm.merge(club_map[["league", "team", "club_id"]], how="left").merge(
        snaps, on=["club_id", "month"], how="left")
    for col, name in [("sv_total", "sv_total"), ("sv_top11", "sv_top11")]:
        lg = np.log(tm[col].where(tm[col] > 0))
        tm[f"{name}_rel"] = lg - lg.groupby([tm["league"], tm["month"]]).transform("mean")
    tm["sv_n"] = tm["sv_n"].astype(float)
    thin = tm["sv_n"] < MIN_VALUED_PLAYERS  # too few valued players = unreliable total
    tm.loc[thin, ["sv_total_rel", "sv_top11_rel"]] = np.nan
    return tm[["league", "team", "month", "sv_total_rel", "sv_top11_rel", "sv_n"]]


POSITIONS = ["Goalkeeper", "Defender", "Midfield", "Attack"]


def position_profile(valuations: pd.DataFrame, players: pd.DataFrame, club_ids: set[int],
                     as_of: pd.Timestamp) -> pd.DataFrame:
    """Descriptive squad-value profile per club on ``as_of`` (same snapshot logic as the
    model features: latest valuation of each player in the 12 months before the first day
    of the month): value by position group, total, and 'depth' = value of the players
    ranked 12th to 22nd in the squad. Not used as a model input."""
    g = as_of.replace(day=1)
    v = valuations[valuations["current_club_id"].isin(club_ids)].copy()
    v["date"] = pd.to_datetime(v["date"])
    v = v[(v["date"] < g) & (v["date"] >= g - pd.Timedelta(days=WINDOW_DAYS))].dropna(
        subset=["market_value_in_eur"]).sort_values("date")
    v = v.groupby("player_id").tail(1).merge(players[["player_id", "position"]], on="player_id", how="left")
    rows = {}
    for club, d in v.groupby("current_club_id"):
        d = d.sort_values("market_value_in_eur", ascending=False)
        r = {p: d.loc[d["position"] == p, "market_value_in_eur"].sum() for p in POSITIONS}
        r["Total"] = d["market_value_in_eur"].sum()
        r["Depth (players 12-22)"] = d["market_value_in_eur"].iloc[11:22].sum()
        r["Valued players"] = len(d)
        rows[club] = r
    return pd.DataFrame(rows).T.rename_axis("club_id")
