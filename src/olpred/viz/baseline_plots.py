"""Figures for Phase 4. Run after olpred.run_baselines:  python -m olpred.viz.baseline_plots"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from olpred.evaluation.metrics import (CLASS_NAMES, paired_block_bootstrap, per_match_log_loss,
                                       reliability, week_blocks)
from olpred.viz import style

TABLES = Path("reports/tables")
FIGURES = Path("reports/figures")


def _load():
    probs = dict(np.load(TABLES / "phase4_val_probs.npz"))
    idx = pd.read_csv(TABLES / "phase4_val_index.csv")
    dates = pd.read_parquet("data/processed/features.parquet", columns=["match_id", "date"])
    dates = idx.merge(dates, on="match_id")["date"]
    return probs, idx["y"].to_numpy(), dates


def logloss_figure(probs, y, dates) -> Path:
    blocks = week_blocks(dates)
    rows = []
    for name, p in probs.items():
        loss = per_match_log_loss(y, p)
        ci = paired_block_bootstrap(loss, np.zeros_like(loss), blocks)  # CI of the mean loss
        rows.append((name, ci["diff"], ci["ci_low"], ci["ci_high"]))
    d = pd.DataFrame(rows, columns=["model", "ll", "lo", "hi"]).sort_values("ll", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, r in enumerate(d.itertuples()):
        color = (style.AQUA if r.model.startswith("Bookmaker")
                 else style.NEUTRAL if r.model.startswith("Class") else style.BLUE)
        ax.plot([r.lo, r.hi], [i, i], color=color, lw=2)
        ax.plot(r.ll, i, "o", color=color, ms=8, mec=style.SURFACE, mew=2)
        ax.text(r.hi + 0.002, i, f"{r.ll:.4f}", va="center", color=style.INK_2, fontsize=9)
    ax.set_yticks(range(len(d)), d["model"])
    ax.set_xlabel("Log loss on validation seasons 2021–22 (lower is better); bars = 95 % CI")
    ax.set_title("More information helps, but the bookmaker is still ahead")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(d["lo"].min() - 0.01, d["hi"].max() + 0.012)
    # legend: colour = kind of model
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=c, lw=2, ms=7, label=l) for c, l in
                       [(style.NEUTRAL, "Naive"), (style.BLUE, "Our models"),
                        (style.AQUA, "Bookmaker (reference)")]], loc="upper right")
    out = FIGURES / "phase4_logloss.png"
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out


def reliability_figure(probs, y) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharex=True, sharey=True)
    for k, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], color=style.NEUTRAL, lw=1.2, ls="--")
        for name, color in style.MODEL_COLORS.items():
            r = reliability((y == k).astype(float), probs[name][:, k], bins=8)
            ax.plot(r["forecast"], r["observed"], "-o", color=color, ms=5, mec=style.SURFACE,
                    mew=1.5, label=name.replace(" (reference)", "").replace("Bet365", "Bet365"))
        ax.set_title(CLASS_NAMES[k], fontsize=11)
        ax.set_xlabel("Forecast probability")
        ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.9); ax.set_aspect("equal")
    axes[0].set_ylabel("Observed frequency")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Calibration on validation seasons: forecasts close to the diagonal are honest",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = FIGURES / "phase4_reliability.png"
    fig.savefig(out); plt.close(fig)
    return out


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    style.apply()
    probs, y, dates = _load()
    print(logloss_figure(probs, y, dates), reliability_figure(probs, y))


if __name__ == "__main__":
    main()
