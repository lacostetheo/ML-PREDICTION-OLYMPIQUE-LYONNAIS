"""UEFA club competitions 2026/27: league-phase fixtures and results from Wikipedia.

Only scores and dates are used (facts). The wikitext of the league-phase page is cached in
data/raw/calendar/ and parsed into one row per match. Wikipedia can lag behind the real results,
so the number of recorded matches is returned with the data.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

RAW = Path("data/raw/calendar")
TITLES = {"ucl": "2026–27_UEFA_Champions_League_league_phase",
          "uel": "2026–27_UEFA_Europa_League_league_phase"}
URL = "https://en.wikipedia.org/w/index.php?title={title}&action=raw"


def download(comp: str, tries: int = 4) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    for k in range(tries):  # Wikipedia answers 429 when asked too often: wait and retry politely
        resp = requests.get(URL.format(title=TITLES[comp]), timeout=60, headers={"User-Agent": "academic-project (OL-LAB)"})
        if resp.status_code != 429:
            break
        time.sleep(float(resp.headers.get("Retry-After", 15 * (k + 1))))
    resp.raise_for_status()
    path = RAW / f"{comp}_league_phase.wiki"
    path.write_text(resp.text, encoding="utf8")
    return path


def _name(s: str) -> str:
    s = re.sub(r"\{\{fbaicon\|\w+\}\}", "", s)
    s = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", s)
    return s.strip()


def parse(wikitext: str) -> pd.DataFrame:
    """md, date, time (local text, may be empty), home, away, hg, ag (NaN if not played)."""
    body = wikitext[wikitext.index("==Matches=="):]
    rows = []
    for blk in re.split(r"\n(?====Matchday)", body):
        m = re.match(r"===Matchday (\d)", blk.strip())
        if not m:
            continue
        md = int(m.group(1))
        for b in blk.split("{{#invoke:Football box|main")[1:]:
            g = lambda k: (re.search(r"\|" + k + r"\s*=([^\n]*)", b) or [None, ""])[1].strip()
            d = re.search(r"Start date\|(\d+)\|(\d+)\|(\d+)", b)
            sc = re.match(r"(\d+)\s*[–-]\s*(\d+)", g("score"))
            t = re.match(r"(\d{1,2}:\d{2})", g("time"))
            rows.append({"md": md, "date": f"{d[1]}-{int(d[2]):02d}-{int(d[3]):02d}", "time": t.group(1) if t else "",
                         "home": _name(g("team1")), "away": _name(g("team2")),
                         "hg": int(sc[1]) if sc else None, "ag": int(sc[2]) if sc else None})
    return pd.DataFrame(rows)


def load(comp: str, refresh: bool = True) -> pd.DataFrame:
    path = RAW / f"{comp}_league_phase.wiki"
    if refresh or not path.exists():
        try:
            download(comp)
        except Exception as exc:  # noqa: BLE001
            print(f"offline, using cached {comp} page ({exc})")
    df = parse(path.read_text(encoding="utf8"))
    assert len(df) == 144 and len(set(df["home"]) | set(df["away"])) == 36, "unexpected league-phase page"
    return df
