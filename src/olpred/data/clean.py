"""Turn raw football-data CSVs into one tidy, validated match table."""
from __future__ import annotations

import numpy as np
import pandas as pd

RESULT_TO_Y = {"A": 0, "D": 1, "H": 2}  # ordinal: 0 = away win, 1 = draw, 2 = home win

_STATS = {"HS": "hs", "AS": "as_", "HST": "hst", "AST": "ast",
          "HC": "hc", "AC": "ac", "HF": "hf", "AF": "af",
          "HY": "hy", "AY": "ay", "HR": "hr", "AR": "ar"}
_ODDS = {"B365H": "b365_h", "B365D": "b365_d", "B365A": "b365_a",
         "AvgH": "avg_h", "AvgD": "avg_d", "AvgA": "avg_a"}


def build_matches(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per match, sorted by date, with a stable ``match_id``.

    Results are re-derived from the score and checked against the source ``FTR``.
    Odds are kept in separate columns: they are *never* model features
    (see docs/project_design.md, leakage rule 5).
    """
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
        "season": raw["season"].astype(int),
        "league": raw["league"],
        "home": raw["HomeTeam"].str.strip(),
        "away": raw["AwayTeam"].str.strip(),
        "home_goals": pd.to_numeric(raw["FTHG"]),
        "away_goals": pd.to_numeric(raw["FTAG"]),
    })
    for src, dst in {**_STATS, **_ODDS}.items():
        out[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw else np.nan

    out = out.dropna(subset=["home_goals", "away_goals"])
    out["result"] = np.select(
        [out.home_goals > out.away_goals, out.home_goals < out.away_goals], ["H", "A"], "D")
    mismatch = (out["result"] != raw.loc[out.index, "FTR"]).sum()
    assert mismatch == 0, f"{mismatch} rows where FTR disagrees with the score"

    out = out.drop_duplicates(["league", "date", "home", "away"])
    out = out.sort_values(["date", "league", "home"], kind="stable").reset_index(drop=True)
    out.insert(0, "match_id", np.arange(len(out)))
    out["y"] = out["result"].map(RESULT_TO_Y)
    out["pts_home"] = out["y"].map({0: 0, 1: 1, 2: 3})
    out["pts_away"] = out["y"].map({0: 3, 1: 1, 2: 0})
    return out
