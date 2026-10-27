"""Phase 5b: does updating the weights as results arrive improve forecasts?

Three regimes on the same stream of matches, predicted week by week:
  static  - fitted once on the training seasons, never updated
  refit   - fully re-estimated every week on all results seen so far
  online  - weights nudged by gradient descent on each week's results

The learning rate is chosen on a *pseudo-stream* inside the training period
(base model fitted up to 2018, stream = seasons 2019-2020), never on validation.
Run:  python -m olpred.run_online
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from olpred.evaluation.metrics import paired_block_bootstrap, per_match_log_loss, summarize, week_blocks
from olpred.evaluation.splits import split
from olpred.models.baselines import LogisticModel
from olpred.models.online import prequential
from olpred.run_baselines import all_model_cols

PROCESSED = Path("data/processed")
TABLES = Path("reports/tables")
LR_GRID = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03]


def choose_lr(feats: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    base_train = feats[(feats["season"] >= 2011) & (feats["season"] <= 2018)]
    stream = feats[feats["season"].isin([2019, 2020])].sort_values(["date", "match_id"]).reset_index(drop=True)
    base = LogisticModel(cols).fit(base_train)
    y = stream["y"].to_numpy()
    rows = []
    for lr in LR_GRID:
        p = prequential(base, base_train, stream, lr, with_refit=False)
        rows.append({"lr": lr, "static": per_match_log_loss(y, p["static"]).mean(),
                     "online": per_match_log_loss(y, p["online"]).mean()})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    feats = pd.read_parquet(PROCESSED / "features.parquet")
    cols = all_model_cols(feats)

    grid = choose_lr(feats, cols)
    grid.round(5).to_csv(TABLES / "phase5_online_lr_selection.csv", index=False)
    best_lr = float(grid.loc[grid["online"].idxmin(), "lr"])
    print("selected learning rate:", best_lr, flush=True)

    parts = split(feats)
    train = parts["train"]
    val = parts["val"].dropna(subset=["b365_h", "b365_d", "b365_a"]).reset_index(drop=True)
    base = LogisticModel(cols).fit(train)
    probs = prequential(base, train, val, best_lr)

    y = val["y"].to_numpy()
    # prequential() walks week by week, so rows come back in week order = date order
    order = np.argsort(val["date"].to_numpy(), kind="stable")
    assert (order == np.arange(len(val))).all()
    blocks = week_blocks(val["date"])
    loss = {k: per_match_log_loss(y, p) for k, p in probs.items()}
    rows = [{"regime": k, **summarize(y, p)} for k, p in probs.items()]
    table = pd.DataFrame(rows).set_index("regime")
    comp = pd.DataFrame([{"regime": k, "vs": "static",
                          **paired_block_bootstrap(loss[k], loss["static"], blocks)}
                         for k in ("refit", "online")]
                        + [{"regime": "online", "vs": "refit",
                            **paired_block_bootstrap(loss["online"], loss["refit"], blocks)}])
    table.round(4).to_csv(TABLES / "phase5_online_val.csv")
    comp.round(5).to_csv(TABLES / "phase5_online_val_bootstrap.csv", index=False)
    pd.DataFrame({"match_id": val["match_id"], "date": val["date"],
                  **{f"loss_{k}": v for k, v in loss.items()}}).to_csv(
        TABLES / "phase5_online_val_losses.csv", index=False)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string()); print(comp.round(5).to_string())


if __name__ == "__main__":
    main()
