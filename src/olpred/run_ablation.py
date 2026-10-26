"""Ablation study: which categories of data add value? (logistic regression, VALIDATION)

Blocks are added cumulatively (A -> E) and, separately, each family is removed from
the full model (leave-one-family-out). The C of the L2 penalty is re-selected by
time-ordered CV for every configuration. Test seasons are not used.
Run:  python -m olpred.run_ablation
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from olpred.evaluation.metrics import paired_block_bootstrap, per_match_log_loss, summarize, week_blocks
from olpred.evaluation.splits import split
from olpred.features.columns import curated_cols
from olpred.interpret import family
from olpred.models.baselines import LogisticModel

TABLES = Path("reports/tables")
BLOCKS = [("A: Elo only", ["Elo rating"]),
          ("B: + results form (3/5/10 matches, venue)", ["Recent form (3/5/10 matches)", "Venue form (home/away)"]),
          ("C: + shots and xG proxy", ["Shots (on target / against)", "xG proxy"]),
          ("D: + league table, fatigue", ["League table", "Fatigue / schedule"]),
          ("E: + squad market value (= final model)", ["Squad market value"])]


def main() -> None:
    feats = pd.read_parquet("data/processed/features.parquet")
    cols = curated_cols(feats)
    parts = split(feats)
    train = parts["train"]
    val = parts["val"].dropna(subset=["b365_h", "b365_d", "b365_a"]).reset_index(drop=True)
    y, blocks = val["y"].to_numpy(), week_blocks(val["date"])
    fam_of = {c: family(c) for c in cols}

    def run(name, fams):
        sel = [c for c in cols if fam_of[c] in fams]
        m = LogisticModel(sel).fit(train)
        p = m.predict_proba(val)
        print(name, len(sel), round(per_match_log_loss(y, p).mean(), 4), flush=True)
        return {"config": name, "n_columns": len(sel), "C": m.C_, **summarize(y, p)}, per_match_log_loss(y, p)

    rows, losses, used = [], {}, []
    for name, fams in BLOCKS:                      # cumulative blocks
        used += fams
        r, l = run(name, set(used)); rows.append(r); losses[name] = l
    full = BLOCKS[-1][0]
    steps = []
    for (a, _), (b, _) in zip(BLOCKS[:-1], BLOCKS[1:]):
        steps.append({"step": f"{a.split(':')[0]} -> {b.split(':')[0]}", **paired_block_bootstrap(losses[b], losses[a], blocks)})
    loo = []
    all_fams = set(fam_of.values())
    for f in sorted(all_fams):                     # leave one family out
        r, l = run(f"all minus {f}", all_fams - {f})
        loo.append({**r, **{"delta_vs_full": float(l.mean() - losses[full].mean())},
                    **{k: v for k, v in paired_block_bootstrap(l, losses[full], blocks).items() if k.startswith("ci")}})
    # complementary check: Elo + squad value only
    r, l = run("Elo + squad value only", {"Elo rating", "Squad market value"})
    rows.append(r)
    pd.DataFrame(rows).round(4).to_csv(TABLES / "ablation_cumulative.csv", index=False)
    pd.DataFrame(steps).round(4).to_csv(TABLES / "ablation_steps_bootstrap.csv", index=False)
    pd.DataFrame(loo).round(4).to_csv(TABLES / "ablation_leave_one_out.csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows)[["config", "n_columns", "log_loss", "brier", "accuracy", "C"]].round(4).to_string())
    print(pd.DataFrame(steps).round(4).to_string())
    print(pd.DataFrame(loo)[["config", "log_loss", "delta_vs_full", "ci_low", "ci_high"]].round(4).to_string())


if __name__ == "__main__":
    main()
