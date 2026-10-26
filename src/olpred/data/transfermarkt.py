"""Download the dcaribou 'transfermarkt-datasets' snapshot (CC0 on Kaggle).

Only the public bucket linked from the dataset README is used; Transfermarkt
itself is never scraped. Files go to data/raw/transfermarkt/ (git-ignored).
"""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.zip"
RAW_DIR = Path("data/raw/transfermarkt")
BIG5 = ["GB1", "ES1", "IT1", "L1", "FR1"]


def download(raw_dir: Path = RAW_DIR, force: bool = False) -> Path:
    """Fetch and extract the dataset zip (~240 MB) unless already cached."""
    if (raw_dir / "player_valuations.csv.gz").exists() and not force:
        return raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, timeout=600)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(raw_dir)
    return raw_dir


def load(table: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load one table, e.g. 'player_valuations', 'game_lineups', 'games'."""
    return pd.read_csv(raw_dir / f"{table}.csv.gz")
