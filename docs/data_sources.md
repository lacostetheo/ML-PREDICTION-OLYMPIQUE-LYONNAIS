# Data Sources

Status legend: ✅ verified and used · ⚠️ caveat / to decide · ❌ excluded / not reachable.
Raw data is **never committed** (see `.gitignore`); it is downloaded into
`data/raw/` by `src/olpred/data/` and cached.

*Last verified: 2026-10-26.*

## 1. football-data.co.uk — match results, statistics, odds (core) ✅

- URL: https://football-data.co.uk/data.php (files: `mmz4281/<season>/<league>.csv`)
- Leagues: `E0` Premier League, `SP1` La Liga, `I1` Serie A, `D1` Bundesliga, `F1` Ligue 1
- Downloaded: seasons 2010/11 → 2026/27 (in progress), 85 files, **29,239 matches** (OL: 593)
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

## 3. Understat — xG ⚠️ (optional, decision pending)

- URL: https://understat.com (xG for Big-5 leagues since 2014/15)
- Reachable (HTTP 200). **No terms of use or licence found on the site**; xG values are a proprietary model output.
- Decision: not used in the core pipeline. If added in the extension tier, data stays local, uncommitted and credited.

## 4. Kaggle "Football Data from Transfermarkt" (dcaribou) — market values ⚠️ (optional)

- URL: https://www.kaggle.com/datasets/davidcariboo/player-scores
- Licence declared on the Kaggle page: **CC0 (public domain)**. The description says the data is *scraped from Transfermarkt*; Transfermarkt's own terms were not reviewed, so the uploader's CC0 label may not cover everything.
- Decision: extension tier only; use market-value snapshots dated strictly before each match; keep data out of git.

## 5. Excluded

- **FBref advanced stats**: provider removed Opta data in January 2026; only basic historical data remains.
- **Injuries/suspensions history, wages**: no clean, free, complete source identified.
- **Direct scraping of Transfermarkt**: not done.
