# Phase 7 — Interpretability

Reproduce: `python -m olpred.run_interpret && python -m olpred.viz.interpret_plots`

Models are fitted on the training seasons (2011–2020) and interpreted on the
**validation** matches (2021–22): never on data used for fitting, never on the test
seasons. Two models are examined: the regularised logistic regression (the reference
model) and the gradient-boosting model (Phase 5 hyper-parameters).

Methods (scikit-learn / numpy; no extra library): **grouped permutation importance**,
**standardised coefficients and exact additive contributions** of the linear model
(for a softmax model, coefficient × standardised value *is* the exact linear-SHAP
contribution to the logit), and **partial dependence**. The `shap` package was not
added: for a linear model the exact values are computed directly, and the model
families here do not need it to answer the question.

## A finding that changed the feature set: clock proxies

A first version of this analysis (kept in `reports/tables/phase7_legacy_*.csv`) used all
106 candidate columns. It showed a non-trivial contribution from "history depth"
variables (`n_prior`, `games_played_season`). These count how many matches a team has
been observed for: **they grow by about 30 per season for every team (mean 18 in 2010
→ 431 in 2026), i.e. they are a calendar clock**. A model can use a clock as a time
trend, and validation values (327–493) lie outside most of the training range (172–417).
`sv_n` (number of players with a market valuation) is a data-coverage artefact.

Removing the three variables costs **+0.0011 log loss on validation (95 % CI
[−0.0008, +0.0031], not significant)** — so they bought nothing measurable and carry a
robustness risk. The **final feature set (`curated_cols`, 100 columns) excludes them**.
Consequently the "all features" numbers in Phases 4–6 (0.9859 for the logistic
regression) are for the legacy set; the curated logistic regression scores 0.9870 on
the same validation seasons. The final comparison (test seasons) uses the curated set
for every model.

## Which variables explain the forecasts?

Shuffling a *family* of columns jointly (20 repeats) and measuring the rise in
validation log loss (a larger rise = the model relies on it more):

| Family | LogReg Δ log loss | Boosting Δ log loss |
|---|---|---|
| **Elo rating** (4 columns) | **+0.0496** | +0.0260 |
| **Squad market value** (6) | **+0.0236** | **+0.0284** |
| Recent form, 3/5/10 matches (30) | +0.0072 | +0.0006 |
| xG proxy (18) | +0.0032 | +0.0045 |
| Shots on target / against (18) | +0.0030 | +0.0019 |
| Fatigue / schedule (6) | +0.0008 | 0.0000 |
| Venue form (9) | +0.0003 | +0.0002 |
| League table (9) | −0.0001 | +0.0002 |

![Family importance](../reports/figures/phase7_family_importance.png)

The ranking is the same in 2021 and in 2022 (`phase7_family_importance_by_season.csv`).
In the linear model the mean absolute contribution to logit(home) − logit(away) is
0.48 for Elo, 0.34 for squad value, and 0.05–0.16 for each other family.

**Answer to "what explains OL's (and every team's) results?"**
1. **Long-run team strength**, measured by the Elo rating and by squad market value,
   explains almost everything the model can use. Squad value is as important as Elo
   for the boosting model, and carries information Elo lacks (talent bought but not yet
   reflected in results).
2. **Home advantage** is carried by the intercept: over the dataset the home side wins
   44.7 % of matches, draws 25.2 %, loses 30.0 %.
3. **Recent form, shots and the xG proxy** add a little *on top of* strength — consistent
   with Phase 4 (all of them together: ≈ −0.010 log loss vs Elo).
4. **Rest days, fatigue, league position and venue-specific form add essentially
   nothing** once strength is known. (Congestion is measured on league games only;
   European matches are not in the data.)
5. Injuries, line-ups and salaries could not be tested (no data, see `data_sources.md`).

![Partial dependence](../reports/figures/phase7_partial_dependence.png)

Partial dependence (all related columns moved together): P(home win) rises roughly
linearly with the Elo difference (0.27 → 0.63 for the logistic regression over ±300
points) and with the squad-value difference (0.36 → 0.55). The boosting model
agrees in the middle of the range but **flattens at extreme Elo gaps** (0.38 → 0.58),
i.e. it refuses to be as confident as the linear model about mismatches, which are
rare in the data. Rest days have no effect (0.449 → 0.455).

## Warning signs that deserve caution

- **The sign of the "form" effect is not interpretable in the linear model.** At fixed
  Elo and squad value, the logistic regression assigns a *negative* net effect to
  better recent form (P(home win) falls 0.50 → 0.40 across ±1.5 points per match),
  while boosting finds none (0.454 → 0.448). Form windows are highly collinear with each
  other, with shots and with Elo (which already updates after every result), so the
  coefficients offset one another. This is a collinearity artefact, not evidence that
  "hot streaks hurt". Coefficients of correlated variables must not be read causally;
  the family-level importance above is the robust summary.
- **Permutation importance of correlated families is shared:** Elo and squad value
  are correlated, so each one's importance includes part of the information the other
  could provide.
- **Squad value is a proxy for club wealth and long-run quality**, not a causal driver
  of results; it is known before the match, so it is not leakage.
- **The xG proxy is not an independent signal** (it equals 0.287 × shots on target), so
  its importance is the importance of shots on target counted twice.
- Importances describe the *models*, estimated on two validation seasons; they say
  what the models use, not what causes results in football.
