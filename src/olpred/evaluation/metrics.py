"""Evaluation of 3-class probabilistic forecasts.

Class order everywhere: 0 = away win, 1 = draw, 2 = home win.
Primary metric: log loss (a proper scoring rule — it rewards honest probabilities).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

CLASS_NAMES = ["Away win", "Draw", "Home win"]


def per_match_log_loss(y: np.ndarray, P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 1e-15, 1.0)
    return -np.log(P[np.arange(len(y)), y])


def brier(y: np.ndarray, P: np.ndarray) -> float:
    """Multiclass Brier score: mean over matches of the sum of squared errors."""
    onehot = np.eye(P.shape[1])[y]
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def rps(y: np.ndarray, P: np.ndarray) -> float:
    """Ranked probability score (classes are ordered away < draw < home)."""
    cum_p = np.cumsum(P, axis=1)[:, :-1]
    cum_y = np.cumsum(np.eye(P.shape[1])[y], axis=1)[:, :-1]
    return float(np.mean(np.sum((cum_p - cum_y) ** 2, axis=1) / (P.shape[1] - 1)))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Binary reliability table (quantile bins): mean forecast vs observed frequency."""
    d = pd.DataFrame({"p": p, "o": y})
    d["bin"] = pd.qcut(d["p"], q=bins, duplicates="drop")
    g = d.groupby("bin", observed=True).agg(forecast=("p", "mean"), observed=("o", "mean"),
                                           n=("o", "size"))
    return g.reset_index(drop=True)


def ece(y: np.ndarray, P: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error, one-vs-rest per class then averaged."""
    vals = []
    for k in range(P.shape[1]):
        r = reliability((y == k).astype(float), P[:, k], bins)
        vals.append(np.sum(r["n"] * (r["forecast"] - r["observed"]).abs()) / r["n"].sum())
    return float(np.mean(vals))


def summarize(y: np.ndarray, P: np.ndarray) -> dict[str, float]:
    pred = P.argmax(axis=1)
    return {
        "log_loss": float(per_match_log_loss(y, P).mean()),
        "brier": brier(y, P),
        "rps": rps(y, P),
        "accuracy": float((pred == y).mean()),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "f1_draw": float(f1_score(y, pred, labels=[1], average="macro", zero_division=0)),
        "ece": ece(y, P),
    }


def paired_block_bootstrap(loss_a: np.ndarray, loss_b: np.ndarray, blocks: np.ndarray,
                           n_boot: int = 4000, seed: int = 0) -> dict[str, float]:
    """Mean of (loss_a − loss_b) with a 95 % CI from a block bootstrap.

    Matches are resampled in whole calendar-week blocks because results played
    the same week share context (weather, form period, rule changes) and are not
    independent. Negative mean = model A is better."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    codes, uniq = pd.factorize(blocks)
    sums = np.bincount(codes, weights=d)
    counts = np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "significant": bool(lo > 0 or hi < 0)}


def week_blocks(dates: pd.Series) -> np.ndarray:
    return ((dates - dates.min()).dt.days // 7).to_numpy()
