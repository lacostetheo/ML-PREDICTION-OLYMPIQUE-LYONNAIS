# Data Sources

Status legend: ✅ verified · ⚠️ to verify · ❌ excluded.
**Licence / terms of use must be read on the source page before any data is
downloaded or published.** Raw data is not committed to the repository; it is
downloaded by `src/olpred/data` into `data/raw/` and documented here.

*Last update: Phase 1. The build environment's network policy blocked all four
hosts below (HTTP 403 from proxy), so terms could not yet be read. Statuses
below are therefore ⚠️ until verified.*

## 1. football-data.co.uk — match results and odds (core)

- URL: https://www.football-data.co.uk/data.php
- Files: `mmz4281/<season>/<league>.csv` (`E0` Premier League, `SP1` La Liga, `I1` Serie A, `D1` Bundesliga, `F1` Ligue 1)
- Period: 1993/94 → present (we use 2010/11 →)
- Variables: date, teams, full-time/half-time score, result, shots, shots on target, corners, fouls, cards, bookmaker odds (B365, Pinnacle, market average/max)
- Frequency: current season updated weekly; past seasons static
- Limitations: no xG, no player data, column availability varies by season/league; odds columns have gaps
- Biases: team-name inconsistencies across sources; odds are bookmaker prices (include margin)
- Status: ⚠️ terms of use to read

## 2. ClubElo — Elo ratings

- URL: http://clubelo.com/API
- Variables: dated Elo rating per club
- Limitations: external model; its construction must be checked for look-ahead (values should be "as of date")
- Status: ⚠️ terms and format to verify

## 3. Understat — xG (optional)

- URL: https://understat.com
- Period: 2014/15 → present, Big-5 + others
- Limitations: proprietary xG model; scraping terms not confirmed; **do not redistribute**
- Status: ⚠️ include only if terms allow personal/academic use

## 4. Kaggle "football data from Transfermarkt" (dcaribou) — market values (optional)

- URL: https://www.kaggle.com/datasets/davidcariboo/player-scores
- Content (to confirm): games, player valuations over time, appearances, lineups, transfers
- Limitations: licence to check on dataset page; valuations are crowd-sourced estimates; snapshot dates must precede match dates
- Status: ⚠️ licence to read. Transfermarkt itself will **not** be scraped.

## 5. Excluded

- **FBref advanced stats**: provider removed Opta data in January 2026; only basic historical data remains.
- **Injuries/suspensions history, wages**: no clean, free, complete source identified.
