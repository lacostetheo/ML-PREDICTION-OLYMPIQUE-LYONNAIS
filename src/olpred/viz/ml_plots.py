"""Figures for Phase 5.  Run after run_ml and run_online:  python -m olpred.viz.ml_plots"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from olpred.viz import style
from olpred.viz.baseline_plots import FIGURES, TABLES, logloss_figure


def ml_logloss() -> None:
    probs = dict(np.load(TABLES / "phase5_val_probs.npz"))
    idx = pd.read_csv(TABLES / "phase5_val_index.csv")
    dates = idx.merge(pd.read_parquet("data/processed/features.parquet", columns=["match_id", "date"]),
                      on="match_id")["date"]
    print(logloss_figure(probs, idx["y"].to_numpy(), dates,
                         title="Non-linear models do not beat logistic regression",
                         filename="phase5_logloss.png"))


def online_figure() -> None:
    d = pd.read_csv(TABLES / "phase5_online_val_losses.csv", parse_dates=["date"])
    skip = 300  # the cumulative average is meaningless over the first few weeks
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    ax.axhline(0, color=style.NEUTRAL, lw=1.5)
    for col, color, label in [("loss_refit", style.ORANGE, "Weekly refit"),
                              ("loss_online", style.BLUE, "Online gradient updates")]:
        diff = ((d[col] - d["loss_static"]).cumsum() / np.arange(1, len(d) + 1)).iloc[skip:]
        ax.plot(d["date"].iloc[skip:], diff, color=color, lw=2)
        ax.text(d["date"].iloc[-1] + pd.Timedelta(days=8), diff.iloc[-1], label, color=style.INK_2,
                va="center", fontsize=9)
    ax.text(pd.Timestamp("2022-05-15"), 0.0009, "Grey line = static model (reference)", color=style.INK_2, fontsize=9)
    ax.set_ylabel("Cumulative mean log-loss difference\nvs static model (below 0 = better)")
    ax.set_title("Updating the weights as results arrive: no significant gain")
    ax.set_xlim(d["date"].iloc[skip], d["date"].max() + pd.Timedelta(days=170))
    fig.text(0.01, 0.01, "First 300 matches omitted (cumulative average too noisy). Validation 2021-22.",
             color=style.INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = FIGURES / "phase5_online.png"
    fig.savefig(out); plt.close(fig)
    print(out)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    style.apply()
    ml_logloss()
    if (TABLES / "phase5_online_val_losses.csv").exists():
        online_figure()


if __name__ == "__main__":
    main()
