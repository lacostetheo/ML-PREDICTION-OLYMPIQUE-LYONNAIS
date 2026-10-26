"""Phase 5: classical ML models, tuned on the training seasons, scored on VALIDATION.

The test seasons are not touched.
Run:  python -m olpred.run_ml
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from olpred.evaluation.metrics import paired_block_bootstrap, per_match_log_loss, summarize, week_blocks
from olpred.evaluation.splits import split
from olpred.models.baselines import BookmakerModel, FrequencyModel, LogisticModel
from olpred.models.ml import Blend, gradient_boosting, random_forest, svm_rbf
from olpred.run_baselines import FORM_DIFFS, all_model_cols

PROCESSED = Path("data/processed")
TABLES = Path("reports/tables")
STORE = Path("models_store")
REFS = ["Class frequency (naive)", "Elo (LogReg)", "LogReg: all features",
        "Bookmaker Bet365 (reference)"]


def build_models(feats: pd.DataFrame) -> dict[str, object]:
    cols = all_model_cols(feats)
    return {
        "Class frequency (naive)": FrequencyModel(),
        "Elo (LogReg)": LogisticModel(["elo_diff"]),
        "LogReg: all features": LogisticModel(cols),
        "Random Forest": random_forest(cols),
        "Gradient Boosting": gradient_boosting(cols),
        "SVM (RBF)": svm_rbf(["elo_diff"] + FORM_DIFFS),
        "Bookmaker Bet365 (reference)": BookmakerModel(),
    }


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    STORE.mkdir(exist_ok=True)
    feats = pd.read_parquet(PROCESSED / "features.parquet")
    parts = split(feats)
    train = parts["train"]
    val = parts["val"].dropna(subset=["b365_h", "b365_d", "b365_a"]).reset_index(drop=True)
    y = val["y"].to_numpy()

    models = build_models(feats)
    probs, rows, tuning = {}, [], []
    for name, model in models.items():
        print(f"fitting {name} ...", flush=True)
        model.fit(train)
        probs[name] = model.predict_proba(val)
        row = {"model": name, **summarize(y, probs[name])}
        if hasattr(model, "fit_seconds_"):
            row["fit_seconds"] = round(model.fit_seconds_)
            row["cv_log_loss"] = model.cv_log_loss_
            tuning.append(model.tuning_table())
            print("   best:", model.best_params_, "cv log loss", round(model.cv_log_loss_, 4), flush=True)
        rows.append(row)
        if name not in ("Class frequency (naive)", "Bookmaker Bet365 (reference)"):
            joblib.dump(model, STORE / f"{name.split(' (')[0].replace(' ', '_').replace(':', '')}.joblib")

    probs["Blend: LogReg + Gradient Boosting"] = Blend(
        [models["LogReg: all features"], models["Gradient Boosting"]]).predict_proba(val)
    rows.append({"model": "Blend: LogReg + Gradient Boosting",
                 **summarize(y, probs["Blend: LogReg + Gradient Boosting"])})

    table = pd.DataFrame(rows).set_index("model")
    blocks = week_blocks(val["date"])
    loss = {k: per_match_log_loss(y, p) for k, p in probs.items()}
    comp = [{"model": n, "vs": r, **paired_block_bootstrap(loss[n], loss[r], blocks)}
            for n in loss for r in REFS if n != r]

    # robustness: log loss by validation season and by league
    by = {}
    for key in ["season", "league"]:
        by[key] = pd.DataFrame({n: pd.Series(l).groupby(val[key].to_numpy()).mean()
                                for n, l in loss.items()}).T

    table.round(4).to_csv(TABLES / "phase5_ml_val.csv")
    pd.DataFrame(comp).round(4).to_csv(TABLES / "phase5_ml_val_bootstrap.csv", index=False)
    pd.concat(tuning).to_csv(TABLES / "phase5_tuning.csv", index=False)
    by["season"].round(4).to_csv(TABLES / "phase5_val_by_season.csv")
    by["league"].round(4).to_csv(TABLES / "phase5_val_by_league.csv")
    np.savez(TABLES / "phase5_val_probs.npz", **probs)
    val[["match_id", "y"]].to_csv(TABLES / "phase5_val_index.csv", index=False)

    pd.set_option("display.width", 220)
    print(table.round(4).to_string())
    print(pd.DataFrame(comp).query("vs == 'LogReg: all features' or vs == 'Bookmaker Bet365 (reference)'")
          .round(4).to_string())
    print(by["season"].round(4).to_string()); print(by["league"].round(4).to_string())
    for m in models.values():
        if hasattr(m, "best_params_"):
            print(m.name, m.best_params_)


if __name__ == "__main__":
    main()
