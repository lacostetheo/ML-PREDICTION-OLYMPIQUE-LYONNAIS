"""Download and load match data from football-data.co.uk.

Terms: free for private, non-commercial use only. Raw files are cached in
data/raw/ (git-ignored, never redistributed) and fetched politely, once.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://football-data.co.uk/mmz4281/{season}/{league}.csv"
LEAGUES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}
FIRST_SEASON_START = 2010
RAW_DIR = Path("data/raw/football_data")


def season_code(start_year: int) -> str:
    """2010 -> '1011'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def download(
    first: int = FIRST_SEASON_START,
    last: int = 2026,
    raw_dir: Path = RAW_DIR,
    refresh_current: bool = True,
    pause: float = 1.0,
) -> list[Path]:
    """Download one CSV per (league, season). Past seasons are cached forever;
    the most recent season is re-downloaded when ``refresh_current`` is True."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for start in range(first, last + 1):
        code = season_code(start)
        for league in LEAGUES:
            path = raw_dir / f"{code}_{league}.csv"
            if path.exists() and not (refresh_current and start == last):
                paths.append(path)
                continue
            resp = requests.get(BASE_URL.format(season=code, league=league), timeout=30)
            if resp.status_code == 200 and resp.content:
                path.write_bytes(resp.content)
                paths.append(path)
            else:
                print(f"skip {code}/{league}: HTTP {resp.status_code}")
            time.sleep(pause)
    return paths


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Concatenate all cached CSVs into one frame with season/league columns."""
    frames = []
    for path in sorted(raw_dir.glob("*.csv")):
        code, league = path.stem.split("_")
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
        df = df.dropna(subset=["HomeTeam", "AwayTeam"]).copy()
        df["season"] = 2000 + int(code[:2])
        df["league"] = league
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
