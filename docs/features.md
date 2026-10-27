# Feature Engineering (Phase 3)

Output: `data/processed/features.parquet` — **29,239 matches × 124 columns**
(metadata + targets + 106 candidate model columns). Reproduce with
`python -m olpred.features.build`.

Target: `y` ∈ {0 = away win, 1 = draw, 2 = home win}. Bookmaker odds are kept in
the table for the market comparison but are **never** model features.

## Feature families

Each team-level feature `f` appears as `h_f` (home team), `a_f` (away team) and
`d_f = h_f − a_f`.

| Family | Features | Construction |
|---|---|---|
| **Elo** (`elo_home`, `elo_away`, `elo_diff`, `elo_exp_home`) | Home-made Elo, K = 20, home advantage = 65, goal-difference multiplier, 25 % shrink to 1500 between seasons, newcomers start 75 below 1500 | Rating **before** the match; updated after it. Per league. |
| **Form** | Rolling mean of points, goals for/against, xG-proxy for/against, shots on target for/against over the last **3, 5, 10** matches; goal difference (5) | `shift(1)` then rolling; teams without history get the league mean over strictly earlier dates |
| **Venue form** | Points / goals for / goals against over the last 5 *home* (or *away*) matches | Same, grouped by team and venue |
| **League table** | Season-to-date points per game, goal difference per game, normalised position (0 = top) | Standings computed before each date's matches |
| **Schedule / fatigue** | Rest days (capped at 14), matches played in the previous 14 days | Domestic league only — European competitions are not in the data |
| **Squad value** | log total squad value and log mean of top-11 values, relative to the league average of that month; number of valued players | Monthly snapshot from valuations dated before the snapshot; NaN if < 11 valued players |
| **History depth** | `n_prior`, `games_played_season` | Lets models discount early-season noise |

Not built (yet): schedule difficulty of upcoming fixtures, per-position squad
strength, injuries — the required data is not available (see `data_sources.md`).

## How leakage is prevented

1. Every rolling/expanding feature is computed on the series **shifted by one
   match** (`shift(1)`), so a match never sees its own result.
2. Fallbacks for missing history (first match of a team, season start) use the
   league mean over **strictly earlier dates**, never a global mean.
3. Elo stores the rating before the update; season shrinkage uses only
   previous-season ratings.
4. League standings are taken before the matches of the day are applied.
5. Squad values come from a monthly snapshot built from valuations dated
   strictly before the snapshot day (≤ match date).
6. Odds are excluded from the feature matrix.
7. **Automated perturbation test** (`tests/test_leakage.py`): all results and
   statistics from a cutoff date onward are replaced by random values, the whole
   pipeline is re-run, and the features of every match up to and including the
   cutoff day must be bit-for-bit identical. The test was checked against two
   deliberately injected bugs (rolling window without the shift; Elo recorded
   after the update) and fails for both.

### Known, accepted limitations

- **xG proxy parameters** are fitted on seasons ≤ 2019 (two coefficients, fixed
  afterwards). For matches *inside* that window they carry a very slight
  look-ahead; the 2021+ validation/test seasons are unaffected.
- The Elo hyper-parameters (K, home advantage, shrinkage) are set from
  convention, not tuned; they may be tuned on the validation period in Phase 4.
- Squad value for the current season uses the last Transfermarkt snapshot
  (June 2026): summer 2026 transfers are not reflected.

## Data-quality findings

- **The xG proxy is not real xG.** Fitted result:
  `goals ≈ 0.287 × shots on target + 0.000 × shots off target` (the
  non-negativity constraint drives the off-target coefficient to zero). It is
  therefore perfectly collinear with shots on target (out-of-sample correlation
  with goals 0.588, identical to shots on target alone). `xg_*` columns are
  redundant with `sot_*`; the ablation study will treat them as one block.
- **Club mapping to Transfermarkt:** 176 of 189 teams mapped through unambiguous
  score/date matches (min. 24 supporting matches each, no duplicate IDs). 13 teams
  have no mapping (clubs outside the top flight or not covered, e.g. Blackburn,
  Bolton, Coventry, Le Mans, Elversberg) → no squad value.
- **Missing squad values:** season 2010: 26 %, 2011: 11 %, 2012: 1.5 %, 2013–2025: ≈ 0 %,
  2026: 11 % (newly promoted clubs). Overall 490 of 28,000+ modelling rows have no
  squad value; they will be imputed on the training set (with an indicator) or
  handled natively by tree models. Seasons 2010–11 serve as warm-up for Elo and
  rolling features and are not used for fitting.
- **Face validity of Elo:** the home-win rate rises monotonically with the Elo
  expectation (22 % → 35 % → 45 % → 52 % → 70 % across quintiles).

## Splits (draft)

| Period | Seasons | Matches |
|---|---|---|
| Warm-up (features only) | 2010 | 1,826 |
| Train | 2011–2020 | ~18,160 |
| Validation | 2021–2022 | 3,652 |
| Test | 2023 → present | 5,602 |
