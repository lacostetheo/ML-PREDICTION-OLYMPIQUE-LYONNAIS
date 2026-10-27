"""Team-level rolling features. Every feature for a match on date T is built
from matches strictly before T (``shift(1)`` before any rolling/expanding op)."""
from __future__ import annotations

import numpy as np
import pandas as pd

ROLL_STATS = ["pts", "gf", "ga", "xg", "xga", "sot", "sot_against"]
WINDOWS = (3, 5, 10)
VENUE_STATS = ["pts", "gf", "ga"]
VENUE_WINDOW = 5
REST_CAP = 14.0


def to_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match (one per team), sorted by team then date."""
    common = ["match_id", "date", "season", "league"]
    home = matches[common].assign(
        team=matches["home"], opp=matches["away"], is_home=1,
        gf=matches["home_goals"], ga=matches["away_goals"],
        shots=matches["hs"], sot=matches["hst"],
        shots_against=matches["as_"], sot_against=matches["ast"])
    away = matches[common].assign(
        team=matches["away"], opp=matches["home"], is_home=0,
        gf=matches["away_goals"], ga=matches["home_goals"],
        shots=matches["as_"], sot=matches["ast"],
        shots_against=matches["hs"], sot_against=matches["hst"])
    long = pd.concat([home, away], ignore_index=True)
    long["pts"] = np.select([long.gf > long.ga, long.gf == long.ga], [3, 1], 0)
    return long.sort_values(["league", "team", "date"], kind="stable").reset_index(drop=True)


def league_prior(long: pd.DataFrame, stat: str) -> pd.Series:
    """League-wide mean of ``stat`` over matches on strictly earlier dates
    (used to fill teams with no history yet)."""
    daily = long.groupby(["league", "date"])[stat].agg(["sum", "count"]).reset_index()
    daily = daily.sort_values(["league", "date"])
    g = daily.groupby("league")
    daily["prior"] = (g["sum"].cumsum() - daily["sum"]) / (g["count"].cumsum() - daily["count"])
    daily["prior"] = daily["prior"].replace([np.inf, -np.inf], np.nan)
    merged = long[["league", "date"]].merge(daily[["league", "date", "prior"]], how="left")
    return pd.Series(merged["prior"].to_numpy(), index=long.index)


def add_form(long: pd.DataFrame) -> pd.DataFrame:
    """Rolling means over the last 3/5/10 matches + venue-specific form."""
    long = long.copy()
    long["gd"] = long["gf"] - long["ga"]
    team = [long["league"], long["team"]]
    new: dict[str, pd.Series] = {}

    priors = {s: league_prior(long, s) for s in ROLL_STATS}
    for s in ROLL_STATS:
        shifted = long.groupby(team)[s].shift(1)
        for w in WINDOWS:
            roll = shifted.groupby(team).rolling(w, min_periods=1).mean()
            roll = roll.reset_index(level=[0, 1], drop=True).sort_index()
            new[f"{s}_{w}"] = roll.fillna(priors[s])
    new["gd_5"] = new["gf_5"] - new["ga_5"]

    venue = [long["league"], long["team"], long["is_home"]]
    for s in VENUE_STATS:
        shifted = long.groupby(venue)[s].shift(1)
        roll = shifted.groupby(venue).rolling(VENUE_WINDOW, min_periods=1).mean()
        roll = roll.reset_index(level=[0, 1, 2], drop=True).sort_index()
        new[f"{s}_venue_{VENUE_WINDOW}"] = roll.fillna(priors[s])

    # Games played before this match (history depth) and rest / congestion
    long["n_prior"] = long.groupby(team).cumcount()
    days = long.groupby(team)["date"].diff().dt.days
    new["rest_days"] = days.clip(upper=REST_CAP).fillna(REST_CAP)
    congestion = np.zeros(len(long))
    for _, idx in long.groupby(team).indices.items():
        d = long["date"].to_numpy()[idx]
        congestion[idx] = np.arange(len(idx)) - np.searchsorted(d, d - np.timedelta64(14, "D"), side="left")
    new["matches_14d"] = pd.Series(congestion, index=long.index)

    return pd.concat([long, pd.DataFrame(new)], axis=1)


def add_table(long: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date league table *before* each match:
    points per game, goal difference per game and normalised position (0 = top)."""
    long = long.copy()
    ppg = np.full(len(long), np.nan)
    gdpg = np.full(len(long), np.nan)
    pos = np.full(len(long), np.nan)
    played = np.zeros(len(long))

    for (_, _), block in long.groupby(["league", "season"]):
        teams = {t: i for i, t in enumerate(block["team"].unique())}
        n = len(teams)
        pts = np.zeros(n); gd = np.zeros(n); gp = np.zeros(n); gf = np.zeros(n)
        for date, day in block.groupby("date"):
            ti = day["team"].map(teams).to_numpy()
            # standings before this date's matches (lexsort: last key is primary)
            order = np.lexsort((-gf, -gd, -pts))
            rank = np.empty(n); rank[order] = np.arange(n)
            ix = day.index.to_numpy()
            played[ix] = gp[ti]
            has = gp[ti] > 0
            ppg[ix] = np.where(has, pts[ti] / np.maximum(gp[ti], 1), np.nan)
            gdpg[ix] = np.where(has, gd[ti] / np.maximum(gp[ti], 1), np.nan)
            pos[ix] = np.where(has, rank[ti] / max(n - 1, 1), np.nan)
            pts[ti] += day["pts"].to_numpy()
            gd[ti] += day["gf"].to_numpy() - day["ga"].to_numpy()
            gf[ti] += day["gf"].to_numpy()
            gp[ti] += 1

    long["games_played_season"] = played
    long["ppg_season"] = ppg
    long["gdpg_season"] = gdpg
    long["pos_norm"] = pos
    return long
