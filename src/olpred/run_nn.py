"""Phase 6: neural networks vs the logistic-regression reference, plus a learning curve.

Tuned on the training seasons (time-ordered CV), scored on VALIDATION; test untouched.
Run:  python -m olpred.run_nn
"""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from olpred.evaluation.metrics import paired_block_bootstrap, per_match_log_loss, summarize, week_blocks
from olpred.evaluation.splits import split
from olpred.models.baselines import LogisticModel
from olpred.models.ml import gradient_boosting
from olpred.models.neural import SeedEnsemble, deep_mlp, simple_mlp
from olpred.run_baselines import all_model_cols

PROCESSED = Path("data/processed")
TABLES = Path("reports/tables")
STORE = Path("models_store")
LEARNING_CURVE_SEASONS = [1, 2, 3, 5, 7, 10]  # most recent k training seasons


def learning_curve(train, val, cols, lr_C, gb_params, mlp_ens: SeedEnsemble) -> pd.DataFrame:
    """Validation log loss when training on only the last k seasons before 2021."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    y = val["y"].to_numpy()
    rows = []
    for k in LEARNING_CURVE_SEASONS:
        sub = train[train["season"] > train["season"].max() - k]
        lr = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                           LogisticRegression(C=lr_C, max_iter=1000)).fit(sub[cols], sub["y"])
        gb = HistGradientBoostingClassifier(early_stopping=False, random_state=0, **gb_params
                                            ).fit(sub[cols], sub["y"])
        mlp = SeedEnsemble.__new__(SeedEnsemble)
        mlp.cols, mlp.seeds, mlp.template = mlp_ens.cols, mlp_ens.seeds[:3], mlp_ens.template
        mlp.fit(sub)
        row = {"n_matches": len(sub), "seasons": k}
        for name, p in [("LogReg", lr.predict_proba(val[cols])), ("Gradient Boosting", gb.predict_proba(val[cols])),
                        ("MLP (3-seed avg)", mlp.predict_proba(val))]:
            row[name] = per_match_log_loss(y, p).mean()
        rows.append(row)
        print(row, flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    feats = pd.read_parquet(PROCESSED / "features.parquet")
    cols = all_model_cols(feats)
    parts = split(feats)
    train = parts["train"]
    val = parts["val"].dropna(subset=["b365_h", "b365_d", "b365_a"]).reset_index(drop=True)
    y = val["y"].to_numpy()

    probs = {k: v for k, v in np.load(TABLES / "phase5_val_probs.npz").items()
             if k in ("Class frequency (naive)", "Elo (LogReg)", "LogReg: all features",
                      "Gradient Boosting", "Bookmaker Bet365 (reference)")}
    rows, tuning, fitted = [], [], {}
    for factory in (simple_mlp, deep_mlp):
        tuned = factory(cols)
        print(f"tuning {tuned.name} ...", flush=True)
        tuned.fit(train)
        print("   best:", tuned.best_params_, "cv log loss", round(tuned.cv_log_loss_, 4), flush=True)
        tuning.append(tuned.tuning_table())
        single_name, ens_name = tuned.name + " [1 seed]", tuned.name + " [5-seed avg]"
        probs[single_name] = tuned.predict_proba(val)
        start = time.time()
        ens = SeedEnsemble(tuned).fit(train)
        probs[ens_name] = ens.predict_proba(val)
        fitted[tuned.name] = (tuned, ens)
        for n in (single_name, ens_name):
            rows.append({"model": n, **summarize(y, probs[n]),
                         "cv_log_loss": tuned.cv_log_loss_, "best_params": str(tuned.best_params_)})
        joblib.dump(ens, STORE / f"{tuned.name.split(' (')[0].replace(' ', '_')}_ens.joblib")

    table = pd.DataFrame(rows).set_index("model")
    blocks = week_blocks(val["date"])
    loss = {k: per_match_log_loss(y, p) for k, p in probs.items()}
    nn_names = [n for n in probs if n.startswith("MLP")]
    comp = [{"model": n, "vs": r, **paired_block_bootstrap(loss[n], loss[r], blocks)}
            for n in nn_names for r in ("Elo (LogReg)", "LogReg: all features", "Gradient Boosting",
                                        "Bookmaker Bet365 (reference)")]

    table.round(4).to_csv(TABLES / "phase6_nn_val.csv")
    pd.DataFrame(comp).round(4).to_csv(TABLES / "phase6_nn_val_bootstrap.csv", index=False)
    pd.concat(tuning).to_csv(TABLES / "phase6_tuning.csv", index=False)
    np.savez(TABLES / "phase6_val_probs.npz", **probs)
    val[["match_id", "y"]].to_csv(TABLES / "phase6_val_index.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 120)
    print(table.drop(columns="best_params").round(4).to_string())
    print(pd.DataFrame(comp).round(4).to_string())

    # learning curve: is the dataset large enough for a network?
    lr_C = LogisticModel(cols).fit(train).C_
    gb_params = {"learning_rate": 0.0294, "max_iter": 100, "max_depth": 2, "max_leaf_nodes": 15,
                 "min_samples_leaf": 200, "l2_regularization": 1.0}
    best_nn = min(fitted.values(), key=lambda te: te[0].cv_log_loss_)[1]
    lc = learning_curve(train, val, cols, lr_C, gb_params, best_nn)
    lc.round(4).to_csv(TABLES / "phase6_learning_curve.csv", index=False)
    print(lc.round(4).to_string())


if __name__ == "__main__":
    main()
