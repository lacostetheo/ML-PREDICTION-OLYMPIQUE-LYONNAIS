"""Phase 4: fit the baselines on the training seasons, evaluate on VALIDATION only.

The test seasons are deliberately not touched here.
Run:  python -m olpred.run_baselines
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from olpred.evaluation.metrics import paired_block_bootstrap, per_match_log_loss, summarize, week_blocks
from olpred.evaluation.splits import split
from olpred.models.baselines import BookmakerModel, FrequencyModel, LogisticModel, PoissonGoalModel

PROCESSED = Path("data/processed")
TABLES = Path("reports/tables")

FORM_DIFFS = ["d_pts_5", "d_gf_5", "d_ga_5", "d_sot_5", "d_sot_against_5", "d_pts_venue_5"]


def all_model_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("h_", "a_", "d_")) or c.startswith("elo")]


def make_models(df: pd.DataFrame) -> dict[str, object]:
    return {
        "Class frequency (naive)": FrequencyModel(),
        "League table (LogReg)": LogisticModel(["d_pos_norm", "d_ppg_season"]),
        "Elo (LogReg)": LogisticModel(["elo_diff"]),
        "Poisson goals": PoissonGoalModel(["h_gf_10", "a_ga_10", "elo_diff"],
                                          ["a_gf_10", "h_ga_10", "elo_diff"]),
        "LogReg: Elo + form": LogisticModel(["elo_diff"] + FORM_DIFFS),
        "LogReg: all features": LogisticModel(all_model_cols(df)),
        "Bookmaker Bet365 (reference)": BookmakerModel(),
    }


def run() -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, pd.DataFrame]:
    feats = pd.read_parquet(PROCESSED / "features.parquet")
    parts = split(feats)
    train = parts["train"]
    # same matches for every model: drop the (very few) matches without Bet365 odds
    val = parts["val"].dropna(subset=["b365_h", "b365_d", "b365_a"]).reset_index(drop=True)
    y = val["y"].to_numpy()

    probs, rows = {}, []
    for name, model in make_models(feats).items():
        model.fit(train)
        probs[name] = model.predict_proba(val)
        row = {"model": name, **summarize(y, probs[name])}
        if hasattr(model, "C_"):
            row["C"] = model.C_
        rows.append(row)
    table = pd.DataFrame(rows).set_index("model")

    # paired block bootstrap vs the naive baseline, the Elo model and the bookmaker
    blocks = week_blocks(val["date"])
    loss = {k: per_match_log_loss(y, p) for k, p in probs.items()}
    comp = []
    for name in loss:
        for ref in ["Class frequency (naive)", "Elo (LogReg)", "Bookmaker Bet365 (reference)"]:
            if name != ref:
                comp.append({"model": name, "vs": ref,
                             **paired_block_bootstrap(loss[name], loss[ref], blocks)})
    return table, probs, pd.DataFrame(comp), val


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    table, probs, comp, val = run()
    table.round(4).to_csv(TABLES / "phase4_baselines_val.csv")
    comp.round(4).to_csv(TABLES / "phase4_baselines_val_bootstrap.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string())
    print(comp.round(4).to_string())
    np.savez(TABLES / "phase4_val_probs.npz", **probs)
    val[["match_id", "y"]].to_csv(TABLES / "phase4_val_index.csv", index=False)


if __name__ == "__main__":
    main()
