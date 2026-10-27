# Project Design (Phase 1 — validated)

## Research question

> Do machine-learning models produce better-calibrated and more accurate
> match-outcome probabilities than simple references (Elo, bookmaker-implied
> probabilities) — and can they keep improving as new results arrive?
> What do they say about Olympique Lyonnais (OL)?

OL is a **case study**; models are trained on the Big-5 leagues
(Premier League, La Liga, Serie A, Bundesliga, Ligue 1), 2010/11 → present
(~17k–28k matches). OL alone has only ~35 league matches per season, which is
far too little to train anything.

## Targets

| Target | Treatment |
|---|---|
| P(Win), P(Draw), P(Loss) | **Main target** (3-class, home perspective) |
| Expected points | Derived: `3·P(W) + P(D)` |
| Season points / rank / Top 4 / Europe / relegation | **Not trained** — Monte Carlo on match probabilities |
| Player contribution | Dropped as a target (causal, too little data) — replaced by the ablation study |

Poisson goal model: extra baseline, also gives scorelines for tie-breaks
(goal difference) in the season simulation.

## Models (scikit-learn only)

1. Baselines: class frequency, league table, Elo, Poisson.
2. Multinomial logistic regression (reference model).
3. Gradient descent variants: `SGDClassifier(loss="log_loss")` (L2/L1).
4. Random Forest, Gradient Boosting (`HistGradientBoostingClassifier`).
5. `MLPClassifier` — simple and deeper; tested as a hypothesis, not assumed to win.
6. LSTM/GRU: **excluded** (rolling features already encode the dynamics; ~20k
   rows; documented as a deliberate decision).

## Online updating ("the weights adjust as matches are played")

Three regimes are compared on the same walk-forward test period:

- **Static**: fit once on the training period, never updated.
- **Periodic refit**: refit on all data available before each matchday (gold standard, cheap at this size).
- **Online**: update weights after each matchday with `partial_fit`
  (SGD logistic regression, MLP) and the Elo update rule.

Question answered: *does continual learning improve forecasts, and by how much
(with a confidence interval)?*

## Validation

- Walk-forward by season/matchday; expanding window.
- Draft split: train 2010–2020, validation 2021–2022, **test 2023/24 → present** (touched once).
- Hyperparameters: `TimeSeriesSplit` inside the training period only.
- 2020/21 (empty stadiums) flagged; sensitivity check.
- Live out-of-sample: predictions logged *before* kick-off for the 2026/27 season.

## Metrics

Primary: **log loss**, Brier score. Also accuracy, F1, ROC-AUC, confusion matrix,
reliability diagram. Significance: paired bootstrap on per-match loss differences.

## Leakage rules

1. Features for a match at date T use only matches strictly before T (`shift(1)` / strictly-earlier windows).
2. No end-of-season aggregates; no same-match statistics.
3. Promoted teams / season start: prior from Elo or previous season.
4. Snapshots (market values, ratings) must be dated before T.
5. Bookmaker odds live in a separate market model, never in the main feature set.
6. Test suite includes a **truncation test**: features recomputed on data cut at T must equal the original.

## Finance extension

Model probabilities vs. margin-adjusted bookmaker probabilities: log loss,
Brier, calibration, and an encompassing regression (does the model add
information beyond the market?). Academic only; no betting advice.

## Scope tiers

- **Core**: results + Elo + rolling features, baselines → ML → MLP, walk-forward, online updating, calibration, market comparison, OL case study, Monte Carlo, README, report.
- **Extension**: xG, market values (licence permitting), full ablation, Streamlit.
- **Bonus**: live pre-match prediction log for 2026/27.
