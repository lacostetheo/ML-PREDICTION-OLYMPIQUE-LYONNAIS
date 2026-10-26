"""Phase 7: what drives the predictions?  Run:  python -m olpred.run_interpret

Models are fitted on the training seasons and interpreted on VALIDATION matches
(never on data they were fitted on, never on the test seasons)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier

from olpred.evaluation.splits import split
from olpred.interpret import (families, family_contributions, grouped_permutation_importance,
                              home_minus_away_coefficients, partial_dependence)
from olpred.features.columns import curated_cols
from olpred.models.baselines import LogisticModel

TABLES = Path("reports/tables")
STORE = Path("models_store")
# bundle of related inputs moved together, and the grid of home-minus-away differences
PD_FEATURES = {"elo_diff": (["elo_diff"], np.linspace(-300, 300, 25)),
               "form": (["pts_3", "pts_5", "pts_10"], np.linspace(-1.5, 1.5, 25)),
               "squad_value": (["sv_total_rel", "sv_top11_rel"], np.linspace(-1.5, 1.5, 25)),
               "rest_days": (["rest_days"], np.linspace(-4, 4, 17))}


GB_PARAMS = {"learning_rate": 0.0294, "max_iter": 100, "max_depth": 2, "max_leaf_nodes": 15,
             "min_samples_leaf": 200, "l2_regularization": 1.0}  # tuned in Phase 5


class FrozenGB:
    """Gradient boosting with the hyper-parameters selected in Phase 5."""

    def __init__(self, cols):
        self.cols = list(cols)

    def fit(self, df):
        self.est_ = HistGradientBoostingClassifier(early_stopping=False, random_state=0, **GB_PARAMS
                                                   ).fit(df[self.cols], df["y"])
        return self

    def predict_proba(self, df):
        return self.est_.predict_proba(df[self.cols])


def main() -> None:
    feats = pd.read_parquet("data/processed/features.parquet")
    cols = curated_cols(feats)
    parts = split(feats)
    train, val = parts["train"], parts["val"].reset_index(drop=True)
    y = val["y"].to_numpy()
    lr = LogisticModel(cols).fit(train)
    gb = FrozenGB(cols).fit(train)
    groups = families(cols)

    # 1. grouped permutation importance (both models)
    imp = []
    for name, model in [("LogReg", lr), ("Gradient boosting", gb)]:
        r = grouped_permutation_importance(model.predict_proba, val[cols], y, groups)
        r.insert(0, "model", name)
        imp.append(r)
    imp = pd.concat(imp)
    imp.round(5).to_csv(TABLES / "phase7_family_permutation_importance.csv", index=False)

    # 2. logistic regression: coefficients and additive family contributions
    coefs = home_minus_away_coefficients(lr)
    coefs.round(4).to_csv(TABLES / "phase7_logreg_coefficients.csv", index=False)
    contrib = family_contributions(lr, val)
    summary = pd.DataFrame({"mean_abs_contribution": contrib.abs().mean(),
                            "sd_contribution": contrib.std()}).sort_values("mean_abs_contribution",
                                                                           ascending=False)
    summary.round(4).to_csv(TABLES / "phase7_logreg_family_contribution.csv")

    # 3. partial dependence: linear vs boosted
    pdp = []
    for feat, (bundle, grid) in PD_FEATURES.items():
        for name, model in [("LogReg", lr), ("Gradient boosting", gb)]:
            r = partial_dependence(model.predict_proba, val[cols], bundle, grid)
            r.insert(0, "model", name); r.insert(1, "feature", feat)
            pdp.append(r)
    pd.concat(pdp).round(5).to_csv(TABLES / "phase7_partial_dependence.csv", index=False)

    # 4. stability: does the ranking hold in each validation season?
    by_season = []
    for s, part in val.groupby("season"):
        r = grouped_permutation_importance(lr.predict_proba, part[cols].reset_index(drop=True),
                                           part["y"].to_numpy(), groups, n_repeats=10)
        r.insert(0, "season", s); by_season.append(r)
    pd.concat(by_season).round(5).to_csv(TABLES / "phase7_family_importance_by_season.csv", index=False)

    pd.set_option("display.width", 200)
    print(imp.round(4).to_string(index=False))
    print(summary.round(3)); print(coefs.head(12).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
