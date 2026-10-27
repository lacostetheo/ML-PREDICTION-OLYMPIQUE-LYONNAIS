"""Export everything the static website needs to ``site/data.js``.

The site does no machine learning in the browser: the logistic regression is evaluated
here for every ordered pair of teams in the five leagues (1,752 pairs), together with
the exact decomposition of each forecast by feature family, uncertainty intervals, the
real upcoming fixtures, and a replay of the Monte Carlo simulations.
Run after run_ol / run_season_sim / run_final:  python -m olpred.export_site
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from olpred.data import calendar as cal
from olpred.data.football_data import load_fixtures
from olpred.features.columns import curated_cols
from olpred.interpret import families, family_contributions
from olpred.predict.league import LEAGUES
from olpred.predict.state import fixture_frame
from olpred.predict.uncertainty import match_intervals

PROCESSED, TABLES, SITE = Path("data/processed"), Path("reports/tables"), Path("site")
N_REPLAY = 3000          # simulated seasons shipped for the animated replay (the stats use all 10,000)
FAMILIES_SHORT = {"Elo rating": "Elo", "Squad market value": "Squad value", "Recent form (3/5/10 matches)": "Recent form",
                  "Shots (on target / against)": "Shots", "xG proxy": "xG proxy", "League table": "League table",
                  "Fatigue / schedule": "Schedule", "Venue form (home/away)": "Venue form"}


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def r(x, n=4):
    return [round(float(v), n) for v in x]


def pair_table(league: str, teams: list[str], state, lr, boot, sigma, fam):
    fx = pd.DataFrame([(league, h, a) for h in teams for a in teams if h != a], columns=["league", "home", "away"])
    X = fixture_frame(fx, state, neutral_schedule=True)
    res = match_intervals(boot["models"], X, sigma, n_shocks=10)
    contrib = family_contributions(lr, X)[list(FAMILIES_SHORT)]
    pairs = {}
    for i, row in fx.iterrows():
        pairs[f"{row.home}|{row.away}"] = (
            r(res.loc[i, ["p_away", "p_draw", "p_home"]])
            + r(res.loc[i, ["p_away_total_lo", "p_away_total_hi", "p_draw_total_lo", "p_draw_total_hi",
                            "p_home_total_lo", "p_home_total_hi"]])
            + r(contrib.loc[i], 3) + [round(float(X.loc[i, "elo_diff"]), 1)])
    return pairs


def previous_results(n_recent: int = 150) -> dict:
    """Out-of-sample (walk-forward) forecasts vs what happened, most recent matches first.
    These are re-computed after the fact with only the information available before each match;
    they are not forecasts that were published in advance."""
    d = pd.read_parquet(TABLES / "final_test_predictions.parquet").sort_values(["date", "match_id"])
    M, B = "LogReg (final)", "Bookmaker Bet365 (reference)"
    P = d[[f"{M}|away", f"{M}|draw", f"{M}|home"]].to_numpy()
    Q = d[[f"{B}|away", f"{B}|draw", f"{B}|home"]].to_numpy()
    F = d[[f"Class frequency (naive)|away", f"Class frequency (naive)|draw", f"Class frequency (naive)|home"]].to_numpy()
    y = d["y"].to_numpy()
    ll = lambda A: -np.log(np.clip(A[np.arange(len(y)), y], 1e-15, 1))
    hit = P.argmax(1) == y
    last = d["date"].max()

    def summary(mask):
        m = mask.to_numpy() if hasattr(mask, "to_numpy") else mask
        return {"n": int(m.sum()), "acc": round(float(hit[m].mean()), 4), "ll": round(float(ll(P)[m].mean()), 4),
                "ll_book": round(float(ll(Q)[m].mean()), 4), "ll_naive": round(float(ll(F)[m].mean()), 4)}
    windows = {"Last 7 days": d["date"] > last - pd.Timedelta(days=7), "Last 30 days": d["date"] > last - pd.Timedelta(days=30),
               "Whole test period (2023 → Oct 2026)": d["date"] >= d["date"].min()}
    conf = P.max(1)
    bins = []
    for lo, hi in [(0, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)]:
        m = (conf >= lo) & (conf < hi)
        bins.append({"lo": lo, "hi": min(hi, 1), "n": int(m.sum()), "said": round(float(conf[m].mean()), 3),
                     "came_true": round(float(hit[m].mean()), 3)})
    r = d.tail(n_recent).iloc[::-1]
    rows = [[x.date.strftime("%Y-%m-%d"), x.league, x.home, x.away, int(x.home_goals), int(x.away_goals), int(x.y),
             [round(float(v), 3) for v in P[i]]] for x, i in zip(r.itertuples(), range(len(d) - 1, len(d) - 1 - len(r), -1))]
    return {"last": last.strftime("%Y-%m-%d"), "summary": {k: summary(v) for k, v in windows.items()},
            "calibration": bins, "rows": rows}


def main() -> None:
    SITE.mkdir(exist_ok=True)
    prod = joblib.load(Path("models_store") / "production.joblib")
    boot = joblib.load(Path("models_store") / "bootstrap_logreg.joblib")
    lr, sigma, as_of = prod["logreg"], boot["sigma"], prod["as_of"]
    state = pd.read_parquet(PROCESSED / "team_state.parquet")
    feats = pd.read_parquet(PROCESSED / "features.parquet", columns=["date", "league", "home", "away", "elo_home", "elo_away"])
    fam = families(curated_cols(pd.read_parquet(PROCESSED / "features.parquet").head(1)))

    out = {"as_of": as_of.strftime("%Y-%m-%d"), "sigma": sigma, "families": list(FAMILIES_SHORT.values()),
           "leagues": {}, "next": {}, "calendar_source": cal.URL}
    # Elo timeline (monthly, last value of the month)
    long = pd.concat([feats[["date", "league", "home", "elo_home"]].set_axis(["date", "league", "team", "elo"], axis=1),
                      feats[["date", "league", "away", "elo_away"]].set_axis(["date", "league", "team", "elo"], axis=1)])
    long["month"] = long["date"].dt.to_period("M")
    months = pd.period_range("2010-08", long["month"].max(), freq="M")
    out["elo_months"] = [str(m) for m in months]

    pairs_all = {}
    for code, cfg in LEAGUES.items():
        teams = sorted(state.loc[code].index)
        pairs_all[code] = pair_table(code, teams, state, lr, boot, sigma, fam)
        summ = pd.read_csv(TABLES / f"sim_{code}_summary.csv", index_col=0)
        posp = pd.read_csv(TABLES / f"sim_{code}_position_probs.csv", index_col=0)
        z = np.load(PROCESSED / f"sim_{code}.npz")
        order = [list(z["teams"]).index(t) for t in teams]
        elo_rank = state.loc[code]["elo"].rank(ascending=False, method="min").astype(int)
        tinfo = {}
        for t in teams:
            series = (long[(long.league == code) & (long.team == t)].groupby("month")["elo"].last()
                      .reindex(months))
            tinfo[t] = {"elo": round(float(state.loc[(code, t), "elo"]), 1), "elo_rank": int(elo_rank[t]),
                        "sv": None if pd.isna(state.loc[(code, t), "h_sv_total_rel"]) else round(float(state.loc[(code, t), "h_sv_total_rel"]), 2),
                        "pts10": round(float(state.loc[(code, t), "h_pts_10"]), 2),
                        "gf10": round(float(state.loc[(code, t), "h_gf_10"]), 2), "ga10": round(float(state.loc[(code, t), "h_ga_10"]), 2),
                        "elo_series": [None if pd.isna(v) else int(round(v)) for v in series]}
        out["leagues"][code] = {
            "name": cfg["name"], "teams": teams, "info": tinfo, "slots": {k: cfg[k] for k in ("champions_league", "europe", "relegation")},
            "sim": {"teams": teams, "n_replay": N_REPLAY,
                    "positions": b64(z["positions"][:N_REPLAY][:, order]), "points": b64(z["points"][:N_REPLAY][:, order]),
                    "summary": {t: [round(float(summ.loc[t, c]), 4) for c in
                                    ("exp_points", "pts_p10", "pts_p90", "exp_position", "p_title", "p_champions_league",
                                     "p_top4", "p_top6", "p_europe", "p_relegation", "points_now", "matches_played")] for t in teams},
                    "pos_probs": {t: r(posp.loc[t, :], 4) for t in teams}}}
    out["pairs"] = pairs_all

    # next real fixture of every team (+ bookmaker), with the real rest days of each side
    fx = load_fixtures()
    fx = fx[fx["league"].isin(LEAGUES)].reset_index(drop=True)
    for code in LEAGUES:
        sub = fx[fx["league"] == code].reset_index(drop=True)
        X = fixture_frame(sub, state)
        res = match_intervals(boot["models"], X, sigma, n_shocks=10)
        inv = 1.0 / sub[["B365A", "B365D", "B365H"]].to_numpy()
        mk = inv / inv.sum(axis=1, keepdims=True)
        for i, row in sub.iterrows():
            rec = {"date": row["date"].strftime("%Y-%m-%d"), "time": str(row.get("Time", "")) if pd.notna(row.get("Time", np.nan)) else "",
                   "home": row["home"], "away": row["away"],
                   "p": r(res.loc[i, ["p_away", "p_draw", "p_home"]]),
                   "ci": r(res.loc[i, ["p_away_total_lo", "p_away_total_hi", "p_draw_total_lo", "p_draw_total_hi",
                                       "p_home_total_lo", "p_home_total_hi"]]),
                   "market": r(mk[i]), "confirmed": True}
            out["next"].setdefault(code, {})[row["home"]] = {"n": 1, **rec}
            out["next"][code][row["away"]] = {"n": 1, **rec}

    # matches 2 and 3: published Ligue 1 calendar only (other leagues: user picks the opponent)
    matches = pd.read_parquet(PROCESSED / "matches.parquet")
    cur = matches[(matches["season"] == matches["season"].max()) & (matches["league"] == "F1")]
    teams_f1 = set(cur["home"]) | set(cur["away"])
    path = cal.RAW
    if not path.exists():
        cal.download()
    cdf = cal.parse(path.read_text(errors="ignore"), teams_f1)
    cal.validate(cdf)
    upcoming = {}
    today = pd.Timestamp(as_of.date())
    for t in teams_f1:  # the team's next three matches from today on (a round-8 match already played is skipped)
        mine = cdf[((cdf.home == t) | (cdf.away == t)) & (cdf.date >= today)].sort_values("date").head(3)
        upcoming[t] = [{"date": rr.date.strftime("%Y-%m-%d"), "home": rr.home, "away": rr.away, "matchday": int(rr.matchday)}
                       for rr in mine.itertuples()]
    out["calendar_f1"] = upcoming

    # model arena: cumulative log loss over the test matches, weekly points
    pred = pd.read_parquet(TABLES / "final_test_predictions.parquet").sort_values(["date", "match_id"]).reset_index(drop=True)
    models = ["Class frequency (naive)", "Elo (LogReg)", "LogReg (final)", "Gradient boosting", "MLP deep (5 seeds)",
              "Ensemble LogReg+GB+MLP", "Bookmaker Bet365 (reference)"]
    y = pred["y"].to_numpy()
    week = ((pred["date"] - pred["date"].min()).dt.days // 7).to_numpy()
    last_of_week = np.r_[np.flatnonzero(np.diff(week)), len(pred) - 1]
    arena = {"dates": [pred["date"].iloc[i].strftime("%Y-%m-%d") for i in last_of_week],
             "n": [int(i + 1) for i in last_of_week], "curves": {}}
    for m in models:
        P = pred[[f"{m}|away", f"{m}|draw", f"{m}|home"]].to_numpy()
        loss = -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1))
        arena["curves"][m] = [round(float(np.cumsum(loss)[i] / (i + 1)), 4) for i in last_of_week]
    table = pd.read_csv(TABLES / "final_test_table.csv", index_col=0)
    arena["final"] = {m: {"log_loss": float(table.loc[m, "log_loss"]), "lo": float(table.loc[m, "log_loss_ci_low"]),
                          "hi": float(table.loc[m, "log_loss_ci_high"]), "accuracy": float(table.loc[m, "accuracy"]),
                          "brier": float(table.loc[m, "brier"])} for m in models}
    out["arena"] = arena

    out["prev"] = previous_results()

    ucl = TABLES / "europe_ucl_probabilities.csv"
    if ucl.exists():  # Champions League winner model (python -m olpred.europe_sim)
        u = pd.read_csv(ucl).head(14)
        meta = json.loads((TABLES / "europe_ucl_meta.json").read_text())
        out["ucl"] = {"rows": [{k: (round(float(r[k]), 4) if k != "team" else r[k]) for k in
                                ["team", "top8", "top24", "SF", "Final", "Winner"]} for _, r in u.iterrows()],
                      "played": meta["played"], "total": meta["total"]}

    (SITE / "data.js").write_text("window.OLDATA = " + json.dumps(out, separators=(",", ":")) + ";\n")
    print("site/data.js", round((SITE / "data.js").stat().st_size / 1e6, 2), "MB;", sum(len(v) for v in pairs_all.values()), "pairs")


if __name__ == "__main__":
    main()
