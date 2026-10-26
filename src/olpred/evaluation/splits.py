"""Temporal splits (see docs/features.md). Season = year in which it starts.

2010 is warm-up only (Elo / rolling features need history). The test set is
reserved for the final model comparison and must not be used for model
selection."""
from __future__ import annotations

import pandas as pd

WARMUP = 2010
TRAIN = (2011, 2020)
VAL = (2021, 2022)
TEST = (2023, 9999)


def _between(df: pd.DataFrame, bounds: tuple[int, int]) -> pd.DataFrame:
    return df[(df["season"] >= bounds[0]) & (df["season"] <= bounds[1])]


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return ``{'train', 'val', 'test'}``, each sorted by date."""
    return {name: _between(df, b).sort_values(["date", "match_id"]).reset_index(drop=True)
            for name, b in [("train", TRAIN), ("val", VAL), ("test", TEST)]}
