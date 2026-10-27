"""Figures for the final test evaluation.  Run: python -m olpred.viz.final_plots"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

from olpred.evaluation.metrics import CLASS_NAMES, reliability
from olpred.models.registry import ENSEMBLE, LOGREG, MARKET, MLP, GB, ELO, NAIVE, POISSON
from olpred.run_final_analysis import probs_of
from olpred.viz import style
from olpred.viz.baseline_plots import FIGURES, TABLES, logloss_figure

ORDER = [NAIVE, ELO, POISSON, LOGREG, GB, MLP, ENSEMBLE, MARKET]


def load():
    pred = pd.read_parquet(TABLES / "final_test_predictions.parquet").sort_values(["date", "match_id"])
    return pred.reset_index(drop=True)


def logloss(pred) -> None:
    probs = {m: probs_of(pred, m) for m in ORDER}
    print(logloss_figure(probs, pred["y"].to_numpy(), pred["date"],
                         title="Three model families are tied; the bookmaker is ahead",
                         filename="final_logloss.png",
                         xlabel="Log loss, walk-forward test 2023 → 2026 (lower is better); bars = 95 % CI"))


def reliability_fig(pred) -> None:
    y = pred["y"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharex=True, sharey=True)
    for k, ax in enumerate(axes):
        ax.plot([0, 1], [0, 1], color=style.NEUTRAL, lw=1.2, ls="--")
        for name, color in [(LOGREG, style.BLUE), (ENSEMBLE, style.ORANGE), (MARKET, style.AQUA)]:
            r = reliability((y == k).astype(float), probs_of(pred, name)[:, k], bins=8)
            ax.plot(r["forecast"], r["observed"], "-o", color=color, ms=5, mec=style.SURFACE, mew=1.5,
                    label=name.replace(" (reference)", ""))
        ax.set_title(CLASS_NAMES[k], fontsize=11); ax.set_xlabel("Forecast probability")
        ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.9); ax.set_aspect("equal")
    axes[0].set_ylabel("Observed frequency"); axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Calibration on the test seasons: probabilities can be taken at face value",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = FIGURES / "final_reliability.png"; fig.savefig(out); plt.close(fig); print(out)


def diagnostics(pred) -> None:
    y = pred["y"].to_numpy(); P = probs_of(pred, LOGREG); M = probs_of(pred, MARKET)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    # confusion matrix (row-normalised)
    cm = confusion_matrix(y, P.argmax(1), labels=[0, 1, 2]); cmn = cm / cm.sum(1, keepdims=True)
    ax = axes[0]; ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1); ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]:.0%}\n(n={cm[i, j]})", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.55 else style.INK, fontsize=9)
    ax.set_xticks(range(3), ["Away", "Draw", "Home"]); ax.set_yticks(range(3), ["Away", "Draw", "Home"])
    ax.set_xlabel("Predicted (most probable outcome)"); ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix, LogReg", fontsize=11)
    for s in ax.spines.values():
        s.set_visible(False)
    # ROC one-vs-rest
    ax = axes[1]; ax.plot([0, 1], [0, 1], color=style.NEUTRAL, ls="--", lw=1.2)
    for k, color in [(2, style.BLUE), (0, style.ORANGE), (1, style.AQUA)]:
        fpr, tpr, _ = roc_curve((y == k).astype(int), P[:, k])
        auc = np.trapezoid(tpr, fpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{CLASS_NAMES[k]} (AUC {auc:.2f})")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (one vs rest), LogReg", fontsize=11); ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    # distribution of predicted P(home win)
    ax = axes[2]; bins = np.linspace(0.05, 0.9, 30)
    ax.hist(P[:, 2], bins=bins, histtype="step", lw=2, color=style.BLUE, label="LogReg (final)")
    ax.hist(M[:, 2], bins=bins, histtype="step", lw=2, color=style.AQUA, label="Bookmaker")
    ax.set_xlabel("Predicted P(home win)"); ax.set_ylabel("Matches")
    ax.set_title("Distribution of predictions", fontsize=11); ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGURES / "final_diagnostics.png"; fig.savefig(out); plt.close(fig); print(out)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    style.apply()
    pred = load()
    logloss(pred); reliability_fig(pred); diagnostics(pred)


if __name__ == "__main__":
    main()
