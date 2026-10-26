"""Analysis of the walk-forward TEST predictions (run after olpred.run_final).

Final model comparison, significance tests, robustness (season / league / OL),
irreducible-noise argument, and the market comparison ("finance" extension).
Run:  python -m olpred.run_final_analysis
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score

from olpred.evaluation.metrics import (CLASS_NAMES, paired_block_bootstrap, per_match_log_loss,
                                       summarize, week_blocks)
from olpred.evaluation.splits import split
from olpred.models.registry import ELO, ENSEMBLE, GB, LOGREG, MARKET, MLP, NAIVE, POISSON, make_models

TABLES = Path("reports/tables")
MODELS = [NAIVE, ELO, POISSON, LOGREG, GB, MLP, ENSEMBLE, MARKET]
LABS = ["away", "draw", "home"]


def probs_of(pred: pd.DataFrame, name: str) -> np.ndarray:
    return pred[[f"{name}|{lab}" for lab in LABS]].to_numpy()


def entropy(P: np.ndarray) -> float:
    return float(-(P * np.log(np.clip(P, 1e-15, 1))).sum(axis=1).mean())


def pool(p_market: np.ndarray, p_model: np.ndarray, w: float) -> np.ndarray:
    """Log-linear pooling: p ∝ p_market^(1-w) · p_model^w."""
    z = (1 - w) * np.log(p_market) + w * np.log(p_model)
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def best_weight(p_market, p_model, y, grid=np.linspace(0, 1, 21)) -> float:
    return float(grid[np.argmin([per_match_log_loss(y, pool(p_market, p_model, w)).mean() for w in grid])])


def validation_market_weight(model_name: str) -> float:
    """Pooling weight chosen on VALIDATION (model fitted up to 2020), to be applied on test."""
    feats = pd.read_parquet("data/processed/features.parquet")
    parts = split(feats)
    val = parts["val"].dropna(subset=["b365_h", "b365_d", "b365_a"]).reset_index(drop=True)
    models = make_models(feats)
    model = models[model_name].fit(parts["train"])
    return best_weight(models[MARKET].predict_proba(val), model.predict_proba(val), val["y"].to_numpy())


def main() -> None:
    pred = pd.read_parquet(TABLES / "final_test_predictions.parquet").sort_values(["date", "match_id"])
    pred = pred.reset_index(drop=True)
    y = pred["y"].to_numpy()
    blocks = week_blocks(pred["date"])
    P = {m: probs_of(pred, m) for m in MODELS + [ENSEMBLE]}
    loss = {m: per_match_log_loss(y, P[m]) for m in MODELS}

    # 1. final comparison table
    rows = []
    for m in MODELS:
        s = summarize(y, P[m])
        s["roc_auc"] = float(roc_auc_score(y, P[m], multi_class="ovr", average="macro"))
        ci = paired_block_bootstrap(loss[m], np.zeros_like(loss[m]), blocks)
        s["log_loss_ci_low"], s["log_loss_ci_high"] = ci["ci_low"], ci["ci_high"]
        rows.append({"model": m, **s})
    table = pd.DataFrame(rows).set_index("model")
    table.round(4).to_csv(TABLES / "final_test_table.csv")

    # 2. paired comparisons
    refs = [LOGREG, ELO, MARKET]
    comp = pd.DataFrame([{"model": m, "vs": r, **paired_block_bootstrap(loss[m], loss[r], blocks)}
                         for m in MODELS for r in refs if m != r])
    comp.round(4).to_csv(TABLES / "final_test_bootstrap.csv", index=False)

    # 3. robustness: by season, by league, OL only
    by_season = pd.DataFrame({m: pd.Series(loss[m]).groupby(pred["season"].to_numpy()).mean() for m in MODELS}).T
    by_league = pd.DataFrame({m: pd.Series(loss[m]).groupby(pred["league"].to_numpy()).mean() for m in MODELS}).T
    by_season.round(4).to_csv(TABLES / "final_test_by_season.csv")
    by_league.round(4).to_csv(TABLES / "final_test_by_league.csv")
    ol = ((pred["home"] == "Lyon") | (pred["away"] == "Lyon")).to_numpy()
    ol_rows = []
    for m in [NAIVE, ELO, LOGREG, GB, MLP, ENSEMBLE, MARKET]:
        ci = paired_block_bootstrap(loss[m][ol], np.zeros(ol.sum()), blocks[ol])
        ol_rows.append({"model": m, "n_matches": int(ol.sum()), "log_loss": ci["diff"],
                        "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
                        "accuracy": float((P[m][ol].argmax(1) == y[ol]).mean())})
    pd.DataFrame(ol_rows).round(4).to_csv(TABLES / "final_test_ol_only.csv", index=False)

    # 4. per-class report and confusion matrix of the final logistic regression
    pr, rc, f1, sup = precision_recall_fscore_support(y, P[LOGREG].argmax(1), labels=[0, 1, 2], zero_division=0)
    pd.DataFrame({"class": CLASS_NAMES, "precision": pr, "recall": rc, "f1": f1, "support": sup}
                 ).round(4).to_csv(TABLES / "final_test_class_report.csv", index=False)
    pd.DataFrame(confusion_matrix(y, P[LOGREG].argmax(1), labels=[0, 1, 2]), index=CLASS_NAMES,
                 columns=CLASS_NAMES).to_csv(TABLES / "final_test_confusion_logreg.csv")

    # 5. how much headroom is there? (irreducible noise)
    noise = pd.DataFrame([{"quantity": "Mean entropy of bookmaker probabilities", "value": entropy(P[MARKET])},
                          {"quantity": "Mean entropy of final LogReg probabilities", "value": entropy(P[LOGREG])},
                          {"quantity": "Realised log loss, bookmaker", "value": loss[MARKET].mean()},
                          {"quantity": "Realised log loss, final LogReg", "value": loss[LOGREG].mean()},
                          {"quantity": "Realised log loss, naive frequencies", "value": loss[NAIVE].mean()}])
    noise.round(4).to_csv(TABLES / "final_test_noise_floor.csv", index=False)

    # 6. market comparison: does the model add information beyond the market?
    odds = pd.read_parquet("data/processed/features.parquet", columns=["match_id", "b365_h", "b365_d", "b365_a"])
    od = pred[["match_id"]].merge(odds, on="match_id")
    overround = float((1 / od[["b365_h", "b365_d", "b365_a"]]).sum(axis=1).mean() - 1)
    pool_rows = []
    for name in [LOGREG, ENSEMBLE]:
        w_val = validation_market_weight(name) if name != ENSEMBLE else validation_market_weight(LOGREG)
        pooled = pool(P[MARKET], P[name], w_val)
        lp = per_match_log_loss(y, pooled)
        r = paired_block_bootstrap(lp, loss[MARKET], blocks)
        w_test = best_weight(P[MARKET], P[name], y)
        pool_rows.append({"model": name, "weight_chosen_on_validation": w_val,
                          "pooled_log_loss": lp.mean(), "market_log_loss": loss[MARKET].mean(),
                          "diff_vs_market": r["diff"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                          "significant": r["significant"], "weight_best_in_sample_on_test": w_test})
    pd.DataFrame(pool_rows).round(4).to_csv(TABLES / "final_test_market_pooling.csv", index=False)
    pd.DataFrame([{"mean_overround": overround}]).round(4).to_csv(TABLES / "final_test_overround.csv", index=False)

    pd.set_option("display.width", 220)
    print(table.round(4).to_string()); print(by_season.round(4).to_string()); print(by_league.round(4).to_string())
    print(pd.read_csv(TABLES / "final_test_ol_only.csv").to_string()); print(noise.round(4).to_string())
    print(comp[comp["vs"].isin([LOGREG, MARKET])].round(4).to_string())
    print(pd.read_csv(TABLES / "final_test_market_pooling.csv").to_string()); print("overround", round(overround, 4))


if __name__ == "__main__":
    main()
