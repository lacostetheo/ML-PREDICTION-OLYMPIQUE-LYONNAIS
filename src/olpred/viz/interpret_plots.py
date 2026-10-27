"""Figures for Phase 7.  Run after olpred.run_interpret:  python -m olpred.viz.interpret_plots"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from olpred.viz import style
from olpred.viz.baseline_plots import FIGURES, TABLES

COLORS = {"LogReg": style.BLUE, "Gradient boosting": style.ORANGE}
LABELS = {"elo_diff": "Elo difference (home − away)",
          "form": "Recent points per match, all windows (home − away)",
          "squad_value": "Squad value, log relative to league (home − away)",
          "rest_days": "Rest days (home − away)"}


def family_importance() -> None:
    d = pd.read_csv(TABLES / "phase7_family_permutation_importance.csv")
    order = (d.groupby("family")["delta_log_loss"].max().sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for model, off in [("LogReg", -0.15), ("Gradient boosting", 0.15)]:
        s = d[d["model"] == model].set_index("family").reindex(order)
        y = np.arange(len(order)) + off
        ax.errorbar(s["delta_log_loss"], y, xerr=1.96 * s["sd"], fmt="o", color=COLORS[model], ms=7,
                    mec=style.SURFACE, mew=1.5, lw=2, label=model)
    ax.axvline(0, color=style.NEUTRAL, lw=1.2)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("Increase in validation log loss when the family is shuffled (bars: ±1.96 sd over 20 shuffles)")
    ax.set_title("Team strength (Elo, squad value) drives the forecasts; the rest adds little")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = FIGURES / "phase7_family_importance.png"
    fig.savefig(out); plt.close(fig)
    print(out)


def partial_dependence() -> None:
    d = pd.read_csv(TABLES / "phase7_partial_dependence.csv")
    feats = list(LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))
    for ax, f in zip(axes.ravel(), feats):
        for model, color in COLORS.items():
            s = d[(d["feature"] == f) & (d["model"] == model)]
            ax.plot(s["value"], s["p_home"], color=color, lw=2, label=model)
        ax.set_xlabel(LABELS[f], fontsize=9)
        ax.set_ylim(0.2, 0.8)
    axes[0, 0].set_ylabel("Mean P(home win)"); axes[1, 0].set_ylabel("Mean P(home win)")
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Partial dependence of P(home win): strength matters most, rest days barely",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGURES / "phase7_partial_dependence.png"
    fig.savefig(out); plt.close(fig)
    print(out)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    style.apply()
    family_importance()
    partial_dependence()


if __name__ == "__main__":
    main()
