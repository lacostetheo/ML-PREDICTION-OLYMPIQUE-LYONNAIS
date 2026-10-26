# Final evaluation — test seasons 2023 → 2026

Reproduce: `python -m olpred.run_final && python -m olpred.run_final_analysis && python -m olpred.viz.final_plots`
(ablation: `python -m olpred.run_ablation`)

## Protocol

- **Walk-forward, one refit per season.** Each test season *S* ∈ {2023, 2024, 2025,
  2026 (partial, to 2026-10-18)} is predicted by models refit on seasons 2011 … *S*−1.
  This is how the system would run in real life: models are updated with new results
  once a year, while Elo and form features are updated after every match.
- Hyper-parameters are **frozen** from Phases 4–6 (selected on training seasons by
  time-ordered CV). The test set was used exactly once, for this evaluation; nothing
  was tuned or selected on it.
- **5,602 matches**, same matches for every model (those without Bet365 odds are
  dropped for all). Final feature set = 100 curated columns (clock proxies removed,
  see `phase7_interpretability.md`).
- Uncertainty: 95 % intervals from a paired **block bootstrap** (calendar weeks resampled).

## 1. Final model comparison

| Model | Accuracy | F1 (macro) | Log loss ↓ [95 % CI] | Brier ↓ | Calibration (ECE) ↓ | ROC-AUC |
|---|---|---|---|---|---|---|
| Baseline: class frequencies | 43.2 % | 0.201 | 1.0754 [1.069, 1.082] | 0.6509 | 0.012 | 0.494 |
| Elo (LogReg) | 52.5 % | 0.391 | 0.9888 [0.979, 0.999] | 0.5900 | 0.017 | 0.658 |
| Poisson goals | 52.0 % | 0.388 | 0.9902 [0.980, 1.001] | 0.5905 | 0.022 | 0.661 |
| **Logistic regression (final)** | 53.3 % | 0.399 | **0.9793** [0.969, 0.990] | 0.5834 | 0.016 | 0.669 |
| Gradient boosting | 53.0 % | 0.395 | 0.9787 [0.968, 0.989] | 0.5826 | 0.020 | 0.672 |
| MLP, deep (5 seeds) | 53.0 % | 0.396 | 0.9796 [0.969, 0.990] | 0.5834 | 0.021 | 0.671 |
| Ensemble LogReg + GB + MLP | 53.0 % | 0.396 | 0.9779 [0.967, 0.988] | 0.5824 | 0.017 | 0.672 |
| *Bookmaker Bet365 (reference)* | *54.0 %* | *0.405* | *0.9658 [0.955, 0.977]* | *0.5740* | *0.022* | *0.686* |

(Full table: `reports/tables/final_test_table.csv`.) The ECE of the bookmaker is
computed on proportionally-normalised probabilities.

![Test log loss](../reports/figures/final_logloss.png)

### Is the improvement real? (paired block bootstrap on log loss)

| Comparison | Δ log loss | 95 % CI | Significant |
|---|---|---|---|
| Logistic regression vs naive | −0.0961 | [−0.106, −0.087] | yes |
| Logistic regression vs Elo | −0.0095 | [−0.0135, −0.0056] | yes |
| Gradient boosting vs logistic regression | −0.0007 | [−0.0032, +0.0018] | no |
| MLP vs logistic regression | +0.0003 | [−0.0016, +0.0022] | no |
| Ensemble vs logistic regression | −0.0014 | [−0.0027, −0.0001] | yes (marginal) |
| Logistic regression vs bookmaker | +0.0135 | [+0.0094, +0.0175] | yes (bookmaker better) |
| Ensemble vs bookmaker | +0.0121 | [+0.0083, +0.0159] | yes (bookmaker better) |

**Reading.** (i) Machine learning on team and squad information is clearly better than
naive rates and than Elo alone. (ii) **Logistic regression, boosting and the neural
network are statistically indistinguishable**; the simplest one is preferred. (iii) Averaging the
three gains 0.0014 — significant but barely, and of no practical weight. (iv) The
**bookmaker beats every model we built** by 0.012–0.014 log loss.

### Robustness

Log loss by test season — the ranking is stable, and the bookmaker is ahead every year
except in the very small 2026 sample:

| | 2023 | 2024 | 2025 | 2026 (346 matches) |
|---|---|---|---|---|
| Logistic regression | 0.9676 | 0.9823 | 0.9904 | 0.9674 |
| Gradient boosting | 0.9679 | 0.9844 | 0.9883 | 0.9551 |
| MLP | 0.9693 | 0.9825 | 0.9906 | 0.9623 |
| Bookmaker | 0.9548 | 0.9647 | 0.9790 | 0.9603 |

By league (logistic regression / bookmaker): Premier League 0.979 / 0.968, La Liga
0.966 / 0.953, Serie A 0.983 / 0.965, Bundesliga 0.977 / 0.963, **Ligue 1 0.995 / 0.983**
— Ligue 1 is the hardest league to forecast for every model, and the gap to the
market is about the same everywhere.

### Calibration, errors and diagnostics

![Diagnostics](../reports/figures/final_diagnostics.png)
![Calibration](../reports/figures/final_reliability.png)

- Probabilities are well calibrated (ECE 1.6–2.2 %): "70 %" really means about 70 %.
- **Draws are almost never the most probable outcome**: only 6 of 5,602 matches were
  predicted as draws (3 correct) among 1,429 actual draws; draw F1 ≈ 0.004, ROC-AUC for the draw class is only 0.56. Home and away wins are
  discriminated moderately (AUC 0.72–0.73). Accuracy (53 %) is therefore a weak
  summary; the probabilistic scores are the right yardstick.
- A model that predicts "home win" for everything has 44 % accuracy; ours is 53 %.

## 2. How much room is there? (irreducible noise)

| Quantity | Value |
|---|---|
| Realised log loss, naive frequencies | 1.0754 |
| Realised log loss, final logistic regression | 0.9793 |
| Realised log loss, bookmaker | 0.9658 |
| Mean entropy of the final model's own probabilities | 0.9791 |
| Mean entropy of the bookmaker's probabilities | 0.9871 |

The model's realised log loss (0.9793) equals the entropy of its own forecasts
(0.9791): it is as uncertain as it claims, i.e. **well calibrated**. A forecaster that knew
the true probabilities could not do much better than the bookmaker's 0.966, so
between the naive 1.075 and that level, our models recover about 87 % of the
achievable improvement (0.096 of 0.110). The remaining ≈ 0.013 is the part that
**line-ups, injuries and news** probably explain — information we do not have. Football
is hard to predict because a match is decided by few goals: the stronger side wins
only about half of the time, and one in four games ends level.
(The bookmaker's realised loss is *below* its own entropy; proportional normalisation of
odds is known to leave probabilities slightly under-confident — consistent with this.)

## 3. Finance extension: do the models add information beyond the market?

The question is purely academic — *can machine-learning probabilities outperform
simple market-implied probabilities?* — and no betting strategy is built or tested.

- Bet365 odds are converted to probabilities by proportional normalisation. The
  average **overround (bookmaker margin) is 5.5 %**.
- The market is better calibrated *and* more accurate than our models (table above).
- **Encompassing test (log-linear pooling).** We combine
  `p ∝ p_market^(1−w) · p_model^w`. The weight *w* is chosen on the validation seasons
  (model fitted ≤ 2020), then applied unchanged to test:

| Model pooled with market | w (from validation) | Pooled log loss | Market log loss | Δ vs market [95 % CI] | Best w on test (in-sample) |
|---|---|---|---|---|---|
| Logistic regression | 0.15 | 0.9663 | 0.9658 | +0.0005 [−0.0001, +0.0011] | **0.00** |
| Ensemble | 0.15 | 0.9664 | 0.9658 | +0.0006 [0.0000, +0.0011] | **0.00** |

The in-sample optimal weight on the model is **zero**: *the models add no information
that the market does not already contain* — the market probabilities encompass ours.
This is consistent with an informationally efficient market for public data. It is
an expected, informative negative result; a 5.5 % margin would in any case make
any naïve use of small differences unprofitable.
- Odds used are Bet365's pre-match prices from football-data.co.uk (not the closing line).

## 4. Ablation study — which data categories add value?

Logistic regression, nested blocks, fitted on 2011–2020, scored on validation 2021–22
(C re-selected for each configuration):

| Model | Columns | Log loss | Δ vs previous [95 % CI] |
|---|---|---|---|
| A: Elo only | 4 | 0.9957 | – |
| B: + results form (3/5/10 matches, venue) | 43 | 0.9931 | −0.0026 [−0.0051, −0.0001] ✔ |
| C: + shots and xG proxy | 79 | 0.9887 | −0.0044 [−0.0072, −0.0017] ✔ |
| D: + league table, fatigue | 94 | 0.9890 | +0.0003 [−0.0013, +0.0019] ✘ |
| E: + squad market value (= final) | 100 | 0.9870 | −0.0020 [−0.0048, +0.0009] ✘ |
| Elo + squad value only | 10 | 0.9916 | – |

Leave-one-family-out from the full model (Δ log loss when removed): Elo **+0.0025**
[+0.0004, +0.0045] ✔, squad value **+0.0020** [−0.0009, +0.0048], recent form
+0.0007, all others ≈ 0 (±0.0004).

**Reading.** Results-based form and shots each add a small, detectable gain on top of
Elo; league position and fatigue add nothing. Squad value brings the final −0.002 but
that step alone is *not* statistically significant on two validation seasons, although
it ranks second in the permutation analysis of Phase 7. The two views differ because
families are **redundant**: when a block is removed and the model refit, the others
compensate (e.g. squad value and Elo both measure long-run strength), whereas
shuffling one without refit breaks it. Take-away: *team-strength information is
essential, the rest is incremental, and the total gain of every non-Elo block together
is only about 0.009 log loss.* Injuries / line-up data (not available) could not be tested.

## 5. OL on the test seasons (109 league matches)

| Model | Log loss [95 % CI] | Accuracy |
|---|---|---|
| Naive frequencies | 1.036 [0.990, 1.081] | 49.5 % |
| Elo | 0.979 [0.904, 1.056] | 51.4 % |
| Logistic regression | 0.982 [0.902, 1.064] | 58.7 % |
| Gradient boosting | 0.971 [0.894, 1.052] | 56.9 % |
| Bookmaker | 0.953 [0.880, 1.029] | 56.9 % |

With 109 matches, intervals are ±0.08 wide: OL performance is consistent with the
whole-sample picture but **no OL-specific conclusion can be drawn** (e.g. the apparently
higher accuracy of the logistic regression on OL is within noise).

## Conclusions

1. Team strength (Elo, squad value) explains almost all the predictable signal;
   everything else is worth about 0.009 log loss.
2. Simple models are enough: logistic regression ≈ boosting ≈ neural network.
3. Models are well calibrated and as uncertain as the data warrant.
4. The market is better; our public-data models carry no incremental information.
