"""Published Ligue 1 calendar (display only: it is NOT a model input).

football-data.co.uk lists only the next round, so matches 2 and 3 of a team come from
Maxifoot's public calendar page (robots.txt allows it; downloaded once, cached in
data/raw/calendar/, fixtures are subject to change by the league). Parsing is validated:
every matchday must contain each club exactly once.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
import requests

URL = "https://www.maxifoot.fr/calendrier-ligue1.htm"
RAW = Path("data/raw/calendar/ligue1.html")
SKIP = {"^", "top", "emplacement publicitaire"}
HEADER = re.compile(r"(\d+)e journ\S*e, \w+ (\d{2}/\d{2}/\d{4})")
SCORE = re.compile(r"^\d+-\d+$")
SHORT_DATE = re.compile(r"^\d{2}/\d{2}$")


def download(path: Path = RAW) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(URL, timeout=30, headers={"User-Agent": "Mozilla/5.0 (academic project)"})
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def parse(raw_html: str, teams: set[str] | None = None) -> pd.DataFrame:
    """Matchday, date, home, away, score (NaN if not played) for the whole season."""
    text = re.sub(r"<script.*?</script>|<style.*?</style>", "", raw_html, flags=re.S)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", text))
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    rows, day, names = [], None, []

    for ln in lines:
        m = HEADER.search(ln)
        if m:
            day = (int(m.group(1)), pd.to_datetime(m.group(2), dayfirst=True))
            names = []
            continue
        if day is None or ln in SKIP or ln.lower().startswith("mis "):
            continue
        if SCORE.match(ln) and rows and rows[-1]["matchday"] == day[0]:
            rows[-1]["score"] = ln
        elif SHORT_DATE.match(ln) and rows and rows[-1]["matchday"] == day[0]:
            d, mth = ln.split("/")
            year = day[1].year + (1 if int(mth) < day[1].month - 6 else 0)  # season spans two calendar years
            rows[-1]["date"] = pd.Timestamp(year=year, month=int(mth), day=int(d))
        elif (ln in teams) if teams is not None else (len(ln) < 30 and not ln[0].isdigit()):
            names.append(ln)
            if len(names) == 2:
                rows.append({"matchday": day[0], "date": day[1], "home": names[0], "away": names[1], "score": None})
                names = []
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame, n_teams: int = 18) -> None:
    for md, g in df.groupby("matchday"):
        names = list(g["home"]) + list(g["away"])
        assert len(names) == n_teams and len(set(names)) == n_teams, f"matchday {md} is not a full round"
