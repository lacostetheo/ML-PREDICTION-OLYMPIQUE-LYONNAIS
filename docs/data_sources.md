# Data Sources

Status legend: ✅ verified and used · ⚠️ caveat / to decide · ❌ excluded / not reachable.
Raw data is **never committed** (see `.gitignore`); it is downloaded into
`data/raw/` by `src/olpred/data/` and cached.

*Last verified: 2026-10-26.*

## 1. football-data.co.uk — match results, statistics, odds (core) ✅

- URL: https://football-data.co.uk/data.php (files: `mmz4281/<season>/<league>.csv`)
- Leagues: `E0` Premier League, `SP1` La Liga, `I1` Serie A, `D1` Bundesliga, `F1` Ligue 1
- Downloaded: seasons 2010/11 → 2026/27 (in progress, last match 2026-10-21), 85 files, **29,239 matches** (OL: 593)
- Variables: date, teams, full-time/half-time score, result, shots, shots on target, corners, fouls, cards, bookmaker odds (B365, Pinnacle, market average/max)
- Frequency: current season refreshed weekly; past seasons are static
- **Terms (quoted from the site):** *"it's all FREE, however its use is intended for private individuals only, NOT commercial or data training products using automated bots/scrapers/AI."*
  - Our use: private, non-commercial academic portfolio project. We download ~85 CSV files once (1 s pause between requests), keep them out of the repository and do not redistribute them.
  - ⚠️ "data training products" is ambiguous for a public ML repo. Mitigation: raw data stays out of git; the repo ships code + aggregated results only; the author may email the site owner to confirm.
- Limitations / biases observed:
  - 2019/20 truncated by COVID (1,725 matches); Ligue 1 has 306 matches/season since 2023/24 (18 teams) vs 380 before.
  - No xG, no player data.
  - Team names inconsistent across seasons (e.g. `Ajaccio` vs `Ajaccio GFCO`) → need a mapping table.
  - Odds coverage: Bet365 complete over the whole period; market average/max only from 2019/20; **Pinnacle missing for 2025/26** (51 % in 2025/26 partial, 0 % in 2026/27) → use Bet365 and, from 2019/20, market average as benchmarks.
  - Date format mixed (`dd/mm/yy`, `dd/mm/yyyy`) → handled with `format="mixed"`.
- Observed class frequencies: Home 44.7 %, Draw 25.2 %, Away 30.0 %.

## 2. ClubElo — Elo ratings ❌ (not reachable yet)

- URL: http://clubelo.com/API
- Status: `api.clubelo.com` resets the connection through the sandbox proxy and `clubelo.com` returns 403 / does not resolve. Terms not read.
- Plan: **compute our own Elo** from the match results (documented, leakage-free by construction). ClubElo remains an optional cross-check if access is fixed.

## 3. Understat — xG ❌ (excluded)

- URL: https://understat.com
- `https://understat.com/robots.txt` reads `User-agent: *` / `Disallow: /` → automated access is disallowed, so it is **not used**.
- Replacement: an **xG proxy** built in-house from football-data shots and shots on target (regression of goals on shots / shots on target, fitted on past seasons only). It is documented as a proxy, not as true xG. Real xG would require a licensed provider.

## 4. Transfermarkt dataset (dcaribou) — market values ✅

- Page: https://www.kaggle.com/datasets/davidcariboo/player-scores (licence declared: **CC0**); downloaded from the public bucket linked in the project README (https://github.com/dcaribou/transfermarkt-datasets).
- Tables used: `player_valuations` (656k rows, 2000 → 2026-06-12) and `games` (only to map clubs by matching results). `game_lineups`, `appearances`, `players`, `clubs` and `transfers` were downloaded but are **not used**.
- Frequency: automated pipeline **stopped in mid-July 2026**: `games` ends 2026-07-06, `player_valuations` 2026-06-12, and 2026/27 squads are not covered.
- Limitations / biases:
  - Values are crowd-sourced estimates, updated irregularly; coverage is dense for 2010–2023 and thinner from 2024 (~9k valuation records/year vs ~14k) → last known value is carried forward.
  - Squad value at date T comes from a monthly snapshot built only from valuations dated strictly before the snapshot day (latest valuation of each player in the previous 12 months, attributed to the club recorded on that valuation). It is **not** rebuilt from line-ups; a player who changed club after his last valuation is still counted for the old club until re-valued.
  - 2026/27: latest snapshot (June 2026) only; transfers after that are not captured.
  - Club names differ from football-data → mapping table required.
  - The data was originally scraped from Transfermarkt, whose own terms were not reviewed; the CC0 label comes from the dataset author. Data stays local and uncommitted.

## 5. Calendars of the matches still to play (display only)

Used **only** to list and date the upcoming matches on the website; they are **not** a model input (the simulator infers the remaining matches from the results: every ordered pair plays once).

- **Premier League, La Liga, Serie A, Bundesliga — openfootball** (https://github.com/openfootball, text files `2026-27/*.txt`, public domain / CC0). Dates and times of the whole season; the scores in these files are older than football-data's, so a match counts as played only if it is in football-data's results. Parsed by `src/olpred/data/openfootball.py` (club names mapped to football-data's by an explicit table, checked against all 78 clubs). Checked against reality: for the 322 matches already played, the calendar date equals the actual date in 98 % of cases (maximum gap 2 days).
- **Ligue 1 — Maxifoot** (https://www.maxifoot.fr/calendrier-ligue1.htm, read once per refresh, cached in `data/raw/calendar/`, not committed). `robots.txt` does not disallow this page; the site's terms of use were not reviewed. The parser is validated (34 matchdays of exactly 18 distinct clubs). One published error is repaired: Rennes–Paris SG appears twice (matchdays 1 and 23); the second is the return leg, Paris SG–Rennes (`repair_duplicates`).
- Dates and kick-off times can change (television, European fixtures). Times are shown only for matches in the next 10 days; a match that is still unplayed after its published date is flagged "postponed".

## 5b. UEFA competitions — Wikipedia (scores and dates only)

- Pages `2026–27 UEFA Champions League league phase` and `2026–27 UEFA Europa League league phase` (wikitext fetched once per refresh through the public API, text under CC BY-SA; only dates, scores and club names, which are facts, are used). Parser: `src/olpred/data/uefa.py`.
- **The pages lag**: on 27 Oct 2026 they listed 29 of the 54 Champions League matches already played (all 54 of the Europa League). The number of missing matches is shown on the site; those matches are simulated.
- Not in the Transfermarkt dataset (strength assumed low): 3 Champions League and 8 Europa League clubs, listed in [`europe_ucl.md`](europe_ucl.md).

## 6. Excluded

- **FBref advanced stats**: provider removed Opta data in January 2026; only basic historical data remains.
- **Injuries/suspensions history, wages**: no clean, free, complete source identified.
- **Direct scraping of Transfermarkt**: not done.
