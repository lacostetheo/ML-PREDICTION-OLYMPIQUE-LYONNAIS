"""Final out-of-sample evaluation on the TEST seasons (2023 -> now).

Walk-forward: each test season S is predicted by models refit on all seasons
2011..S-1 (so 2021-22 validation data are used for fitting, never for scoring
here). Hyper-parameters are frozen from Phases 4-6; nothing is tuned on test data.
Run:  python -m olpred.run_final
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from olpred.evaluation.splits import TEST
from olpred.models.registry import ELO, ENSEMBLE, GB, LOGREG, MLP, make_models

TABLES = Path("reports/tables")
FIRST_TRAIN = 2011


def main() -> None:
    feats = pd.read_parquet("data/processed/features.parquet")
    last = int(feats["season"].max())
    pieces = []
    for season in range(TEST[0], last + 1):
        train = feats[(feats["season"] >= FIRST_TRAIN) & (feats["season"] < season)]
        test = feats[feats["season"] == season].dropna(subset=["b365_h", "b365_d", "b365_a"])
        out = test[["match_id", "date", "season", "league", "home", "away", "y", "result",
                    "home_goals", "away_goals"]].reset_index(drop=True)
        for name, model in make_models(feats).items():
            print(f"{season}: fitting {name}", flush=True)
            model.fit(train)
            P = model.predict_proba(test)
            for k, lab in enumerate(["away", "draw", "home"]):
                out[f"{name}|{lab}"] = P[:, k]
        for k, lab in enumerate(["away", "draw", "home"]):
            out[f"{ENSEMBLE}|{lab}"] = np.mean([out[f"{m}|{lab}"] for m in (LOGREG, GB, MLP)], axis=0)
        pieces.append(out)
    pred = pd.concat(pieces, ignore_index=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(TABLES / "final_test_predictions.parquet")
    print(pred.shape, pred["season"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
