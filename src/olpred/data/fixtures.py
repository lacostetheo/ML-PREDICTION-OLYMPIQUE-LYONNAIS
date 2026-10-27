"""Matches still to play in the five leagues, with their published dates.

Calendars: openfootball (E0, SP1, I1, D1) and Maxifoot (F1, see olpred.data.calendar).  A match counts as played
when it is in the results (football-data.co.uk), which are always more recent than the calendars' scores.
Dates can change: the leagues move matches for television and European fixtures.
"""
from __future__ import annotations

import pandas as pd

from olpred.data import calendar as cal
from olpred.data import openfootball as of


def league_calendar(code: str, teams: set[str], refresh: bool = True) -> pd.DataFrame:
    if code in of.SOURCES:
        return of.load(code, refresh=refresh)
    if refresh or not cal.RAW.exists():
        try:
            cal.download()
        except Exception as exc:  # noqa: BLE001
            print(f"offline, using cached {code} calendar ({exc})")
    df = cal.parse(cal.RAW.read_text(errors="ignore"), teams)
    cal.validate(df, n_teams=len(teams))
    df = repair_duplicates(df, code)
    df.insert(0, "league", code)
    df["time"] = ""
    return df[["league", "matchday", "date", "time", "home", "away"]]


def repair_duplicates(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """A published calendar can list one ordered pair twice (home/away typo) while its reverse is missing.
    The later duplicate is then the return leg with the venues inverted: swap it back."""
    df = df.sort_values(["date", "matchday"]).reset_index(drop=True)
    seen = set()
    for i, r in df.iterrows():
        if (r["home"], r["away"]) in seen and (r["away"], r["home"]) not in set(zip(df["home"], df["away"])):
            print(f"{code}: matchday {r['matchday']} lists {r['home']} - {r['away']} twice; return leg restored as {r['away']} - {r['home']}")
            df.loc[i, ["home", "away"]] = [r["away"], r["home"]]
        seen.add((df.loc[i, "home"], df.loc[i, "away"]))
    return df


def remaining(matches: pd.DataFrame, leagues, refresh: bool = True) -> pd.DataFrame:
    """league, matchday, date, time, home, away for every fixture of the current season not yet played."""
    season = matches["season"].max()
    out = []
    for code in leagues:
        cur = matches[(matches["season"] == season) & (matches["league"] == code)]
        teams = set(cur["home"]) | set(cur["away"])
        c = league_calendar(code, teams, refresh)
        played = set(zip(cur["home"], cur["away"]))
        unknown = played - set(zip(c["home"], c["away"]))
        if unknown:
            print(f"{code}: {len(unknown)} played matches are not in the calendar (e.g. {sorted(unknown)[:2]})")
        left = c[[(h, a) not in played for h, a in zip(c["home"], c["away"])]]
        n = len(teams) * (len(teams) - 1)
        assert len(left) == n - len(cur), f"{code}: {len(left)} fixtures left but {n - len(cur)} expected"
        out.append(left)
    return pd.concat(out, ignore_index=True).sort_values(["league", "date", "matchday", "home"]).reset_index(drop=True)
