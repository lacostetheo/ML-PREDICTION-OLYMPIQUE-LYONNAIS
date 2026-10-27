"""Full 2026/27 calendars of four leagues from the openfootball project (public domain, CC0).

https://github.com/openfootball  - plain-text fixtures with dates, times and scores.  Used for display
(upcoming matches) and to know which matches are left; it is never a model input.  Ligue 1 is not
published there for this season: its calendar comes from olpred.data.calendar.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

RAW = Path("data/raw/calendar")
SOURCES = {"E0": ("england", "1-premierleague"), "SP1": ("espana", "1-liga"),
           "I1": ("italy", "1-seriea"), "D1": ("deutschland", "1-bundesliga")}
URL = "https://raw.githubusercontent.com/openfootball/{repo}/master/2026-27/{file}.txt"

# openfootball club name -> football-data.co.uk name (the names used everywhere else in the project)
ALIASES = {
    "E0": {"AFC Bournemouth": "Bournemouth", "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa", "Brentford FC": "Brentford",
           "Brighton & Hove Albion FC": "Brighton", "Chelsea FC": "Chelsea", "Coventry City FC": "Coventry",
           "Crystal Palace FC": "Crystal Palace", "Everton FC": "Everton", "Fulham FC": "Fulham", "Hull City AFC": "Hull",
           "Ipswich Town FC": "Ipswich", "Leeds United FC": "Leeds", "Liverpool FC": "Liverpool", "Manchester City FC": "Man City",
           "Manchester United FC": "Man United", "Newcastle United FC": "Newcastle", "Nottingham Forest FC": "Nott'm Forest",
           "Sunderland AFC": "Sunderland", "Tottenham Hotspur FC": "Tottenham"},
    "SP1": {"Athletic Club": "Ath Bilbao", "CA Osasuna": "Osasuna", "Club Atlético de Madrid": "Ath Madrid",
            "Deportivo Alavés": "Alaves", "Elche CF": "Elche", "FC Barcelona": "Barcelona", "Getafe CF": "Getafe",
            "Levante UD": "Levante", "Málaga CF": "Malaga", "RC Celta de Vigo": "Celta", "RC Deportivo La Coruña": "La Coruna",
            "RCD Espanyol de Barcelona": "Espanol", "Rayo Vallecano de Madrid": "Vallecano", "Real Betis Balompié": "Betis",
            "Real Madrid CF": "Real Madrid", "Real Racing Club de Santander": "Santander", "Real Sociedad de Fútbol": "Sociedad",
            "Sevilla FC": "Sevilla", "Valencia CF": "Valencia", "Villarreal CF": "Villarreal"},
    "I1": {"AC Milan": "Milan", "AC Monza": "Monza", "ACF Fiorentina": "Fiorentina", "AS Roma": "Roma", "Atalanta BC": "Atalanta",
           "Bologna FC 1909": "Bologna", "Cagliari Calcio": "Cagliari", "Como 1907": "Como", "FC Internazionale Milano": "Inter",
           "Frosinone Calcio": "Frosinone", "Genoa CFC": "Genoa", "Juventus FC": "Juventus", "Parma Calcio 1913": "Parma",
           "SS Lazio": "Lazio", "SSC Napoli": "Napoli", "Torino FC": "Torino", "US Lecce": "Lecce",
           "US Sassuolo Calcio": "Sassuolo", "Udinese Calcio": "Udinese", "Venezia FC": "Venezia"},
    "D1": {"1. FC Köln": "FC Koln", "1. FC Union Berlin": "Union Berlin", "1. FSV Mainz 05": "Mainz", "Bayer 04 Leverkusen": "Leverkusen",
           "Borussia Dortmund": "Dortmund", "Borussia Mönchengladbach": "M'gladbach", "Eintracht Frankfurt": "Ein Frankfurt",
           "FC Augsburg": "Augsburg", "FC Bayern München": "Bayern Munich", "FC Schalke 04": "Schalke 04", "Hamburger SV": "Hamburg",
           "RB Leipzig": "RB Leipzig", "SC Freiburg": "Freiburg", "SC Paderborn 07": "Paderborn", "SV 07 Elversberg": "Elversberg",
           "SV Werder Bremen": "Werder Bremen", "TSG 1899 Hoffenheim": "Hoffenheim", "VfB Stuttgart": "Stuttgart"},
}
DATE = re.compile(r"^\s+(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) (\w{3}) (\d{1,2})(?: (\d{4}))?\s*$")
MATCH = re.compile(r"^\s+(?:(\d{1,2}:\d{2})\s+)?(.+?)\s+v\s+(.+?)(?:\s+(\d+)-(\d+)(?:\s*\(.*\))?)?\s*$")
MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def download(code: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    repo, file = SOURCES[code]
    resp = requests.get(URL.format(repo=repo, file=file), timeout=60)
    resp.raise_for_status()
    path = RAW / f"{code}_openfootball.txt"
    path.write_text(resp.text, encoding="utf8")
    return path


def parse(text: str, code: str) -> pd.DataFrame:
    """league, matchday, date, time, home, away, hg, ag (NaN while unplayed), with football-data names."""
    alias = ALIASES[code]
    rows, md, year, last_month, day, last_time = [], None, None, None, None, ""
    for line in text.split("\n"):
        if line.startswith("▪"):
            md = int(re.search(r"(\d+)", line).group(1))
            continue
        m = DATE.match(line)
        if m:
            mon = MONTHS[m.group(1)]
            if m.group(3):
                year = int(m.group(3))
            elif last_month is not None and mon < last_month:
                year += 1  # the season runs over two calendar years
            last_month = mon
            day = pd.Timestamp(year=year, month=mon, day=int(m.group(2)))
            last_time = ""
            continue
        m = MATCH.match(line)
        if m and md is not None and day is not None:
            h, a = m.group(2).strip(), m.group(3).strip()
            last_time = m.group(1) or last_time  # a blank time means "same slot as the line above"
            rows.append({"league": code, "matchday": md, "date": day, "time": last_time,
                         "home": alias[h], "away": alias[a],
                         "hg": float(m.group(4)) if m.group(4) else float("nan"),
                         "ag": float(m.group(5)) if m.group(5) else float("nan")})
    return pd.DataFrame(rows)


def load(code: str, refresh: bool = True) -> pd.DataFrame:
    path = RAW / f"{code}_openfootball.txt"
    if refresh or not path.exists():
        try:
            download(code)
        except Exception as exc:  # noqa: BLE001
            print(f"offline, using cached {code} calendar ({exc})")
    df = parse(path.read_text(encoding="utf8"), code)
    n = len(ALIASES[code])
    assert len(df) == n * (n - 1) and df.groupby("matchday").size().eq(n // 2).all(), f"{code}: incomplete calendar"
    return df
