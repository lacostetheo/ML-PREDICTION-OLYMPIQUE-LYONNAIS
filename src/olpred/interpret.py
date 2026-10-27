"""Model interpretation tools (scikit-learn / numpy only).

- feature *families* (Elo, form, squad value, ...) so that the many correlated
  columns of one idea are evaluated together;
- grouped permutation importance (shuffle a whole family jointly, measure the
  increase in log loss on held-out matches);
- standardised logistic coefficients and exact additive contributions of a linear
  model (for a softmax model, coefficient × standardised value is the exact
  linear-SHAP contribution to each class logit);
- coherent partial dependence curves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from olpred.evaluation.metrics import per_match_log_loss

FAMILY_ORDER = ["Elo rating", "Recent form (3/5/10 matches)", "Shots (on target / against)",
                "xG proxy", "Venue form (home/away)", "League table", "Fatigue / schedule",
                "Squad market value", "History depth"]


def family(col: str) -> str:
    base = col.split("_", 1)[1] if col.startswith(("h_", "a_", "d_")) else col
    if col.startswith("elo"):
        return "Elo rating"
    if base.startswith(("xg", "xga")):
        return "xG proxy"
    if base.startswith("sot") or base.startswith("shots"):
        return "Shots (on target / against)"
    if "venue" in base:
        return "Venue form (home/away)"
    if base.startswith(("ppg_season", "gdpg_season", "pos_norm")):
        return "League table"
    if base.startswith(("rest_days", "matches_14d")):
        return "Fatigue / schedule"
    if base.startswith("sv_"):
        return "Squad market value"
    if base in ("n_prior", "games_played_season"):
        return "History depth"
    return "Recent form (3/5/10 matches)"


def families(cols: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {f: [] for f in FAMILY_ORDER}
    for c in cols:
        out[family(c)].append(c)
    return {k: v for k, v in out.items() if v}


def grouped_permutation_importance(predict_proba, X: pd.DataFrame, y: np.ndarray,
                                   groups: dict[str, list[str]], n_repeats: int = 20,
                                   seed: int = 0) -> pd.DataFrame:
    """Increase in log loss when the columns of a group are shuffled *jointly*
    (same row permutation for all of them), mean and sd over repeats."""
    rng = np.random.default_rng(seed)
    base = per_match_log_loss(y, predict_proba(X)).mean()
    rows = []
    for name, cols in groups.items():
        deltas = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            deltas.append(per_match_log_loss(y, predict_proba(Xp)).mean() - base)
        rows.append({"family": name, "n_columns": len(cols), "delta_log_loss": np.mean(deltas),
                     "sd": np.std(deltas)})
    return pd.DataFrame(rows).sort_values("delta_log_loss", ascending=False).reset_index(drop=True)


def linear_parts(logit_model):
    """(prep, W, b, names) of a fitted ``LogisticModel`` (imputer, scaler, softmax)."""
    prep, clf = logit_model.pipe_[:-1], logit_model.pipe_[-1]
    names = list(prep.get_feature_names_out(logit_model.cols))
    return prep, clf.coef_, clf.intercept_, names


def home_minus_away_coefficients(logit_model) -> pd.DataFrame:
    """Standardised coefficient of each input on logit(home) − logit(away)."""
    _, W, _, names = linear_parts(logit_model)
    df = pd.DataFrame({"feature": names, "coef_home_minus_away": W[2] - W[0],
                       "coef_draw": W[1] - W.mean(axis=0)})
    df["family"] = [family(n) if not n.startswith("missingindicator") else "Squad market value"
                    for n in df["feature"]]
    return df.reindex(df["coef_home_minus_away"].abs().sort_values(ascending=False).index
                      ).reset_index(drop=True)


def family_contributions(logit_model, X: pd.DataFrame) -> pd.DataFrame:
    """Exact additive contribution of each family to logit(home) − logit(away),
    per match (columns = families)."""
    prep, W, _, names = linear_parts(logit_model)
    Z = prep.transform(X[logit_model.cols])
    contrib = Z * (W[2] - W[0])[None, :]
    fam = np.array([family(n) if not n.startswith("missingindicator") else "Squad market value"
                    for n in names])
    return pd.DataFrame({f: contrib[:, fam == f].sum(axis=1) for f in dict.fromkeys(fam)})


def partial_dependence(predict_proba, X: pd.DataFrame, bundle: list[str], grid: np.ndarray) -> pd.DataFrame:
    """Mean predicted class probabilities when a *bundle* of related inputs is moved
    together to a home-minus-away difference ``v`` for each ``v`` in ``grid``.

    Moving one column of a correlated family (e.g. only the 5-match form window)
    would understate the effect, so every column of the bundle is shifted at once:
    ``h_f := a_f + v`` and ``d_f := v``. Bundle ``["elo_diff"]`` is special-cased to
    keep the four Elo columns coherent."""
    rows = []
    for v in grid:
        Xv = X.copy()
        for f in bundle:
            if f == "elo_diff":
                mid = (Xv["elo_home"] + Xv["elo_away"]) / 2
                Xv["elo_home"], Xv["elo_away"], Xv["elo_diff"] = mid + v / 2, mid - v / 2, v
                Xv["elo_exp_home"] = 1 / (1 + 10 ** (-(v + 65.0) / 400))
            else:
                Xv[f"h_{f}"] = Xv[f"a_{f}"] + v
                Xv[f"d_{f}"] = v
        P = predict_proba(Xv).mean(axis=0)
        rows.append({"value": v, "p_away": P[0], "p_draw": P[1], "p_home": P[2]})
    return pd.DataFrame(rows)
