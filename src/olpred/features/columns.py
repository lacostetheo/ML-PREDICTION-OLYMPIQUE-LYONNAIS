"""Model input columns.

``legacy_cols`` = every candidate column (used in Phases 4-6).
``curated_cols`` = the final feature set: drops *clock proxies* — variables that
mostly measure how much calendar time has passed rather than team quality:

- ``n_prior`` / ``games_played_season``: number of matches seen so far; grows by
  ~30 per season for every team (mean 18 in 2010 -> 431 in 2026), so a model can use
  it as a time trend and then extrapolates outside the training range;
- ``sv_n``: number of players with a market valuation, a data-coverage artefact.

Removing them costs +0.0011 log loss on validation (95 % CI [-0.0008, +0.0031], not
significant) — see docs/phase7_interpretability.md.
"""
from __future__ import annotations

import pandas as pd

CLOCK_PROXIES = ("n_prior", "games_played_season", "sv_n")


def legacy_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("h_", "a_", "d_")) or c.startswith("elo")]


def curated_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in legacy_cols(df)
            if not (c.startswith(("h_", "a_", "d_")) and c.split("_", 1)[1] in CLOCK_PROXIES)]
