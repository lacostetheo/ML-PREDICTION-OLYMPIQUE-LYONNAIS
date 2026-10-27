"""Export everything the static website needs to ``site/data.js``.

The site does no machine learning in the browser: the logistic regression is evaluated here for every
ordered pair of teams in the five leagues (1,752 pairs), with the exact decomposition of each forecast
by feature family and uncertainty intervals; every fixture still to be played (leagues, Champions League,
Europa League) is looked up in that table; season simulations are summarised.
Run after run_ol / run_season_sim / run_final / europe_sim:  python -m olpred.export_site
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from olpred.data import fixtures as fixtures_mod
from olpred.data.football_data import load_fixtures
from olpred.europe import CLUB_IDS
from olpred.features.columns import curated_cols
from olpred.interpret import families, family_contributions
from olpred.predict.league import LEAGUES
from olpred.predict.state import fixture_frame
from olpred.predict.uncertainty import match_intervals

PROCESSED, TABLES, SITE = Path("data/processed"), Path("reports/tables"), Path("site")
FAMILIES_SHORT = {"Elo rating": "Elo", "Squad market value": "Squad value", "Recent form (3/5/10 matches)": "Recent form",
                  "Shots (on target / against)": "Shots", "xG proxy": "xG proxy", "League table": "League table",
                  "Fatigue / schedule": "Schedule", "Venue form (home/away)": "Venue form"}
# football-data.co.uk spelling -> the name a reader expects (display only, data keep the original key)
DISPLAY = {"Ath Bilbao": "Athletic Club", "Ath Madrid": "Atlético Madrid", "Alaves": "Alavés", "Celta": "Celta Vigo",
           "Espanol": "Espanyol", "La Coruna": "Deportivo", "Malaga": "Málaga", "Santander": "Racing Santander",
           "Sociedad": "Real Sociedad", "Vallecano": "Rayo Vallecano", "Betis": "Real Betis", "Milan": "AC Milan",
           "Ein Frankfurt": "Eintracht Frankfurt", "M'gladbach": "Gladbach", "FC Koln": "Köln", "Leverkusen": "Bayer Leverkusen",
           "Hamburg": "Hamburger SV", "Mainz": "Mainz 05", "Nott'm Forest": "Nottingham Forest", "Hull": "Hull City",
           "Ipswich": "Ipswich Town", "Leeds": "Leeds United", "Coventry": "Coventry City", "Stuttgart": "VfB Stuttgart",
           "Dortmund": "Borussia Dortmund", "Newcastle": "Newcastle United", "Tottenham": "Tottenham Hotspur"}


def r(x, n=4):
    return [round(float(v), n) for v in x]


def pair_table(league: str, teams: list[str], state, lr, boot, sigma):
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


def market_table() -> dict:
    """Bookmaker probabilities (Bet365, margin removed) of the next round, from football-data's fixtures file."""
    fx = load_fixtures()
    fx = fx[fx["league"].isin(LEAGUES)].dropna(subset=["B365H", "B365D", "B365A"])
    inv = 1.0 / fx[["B365A", "B365D", "B365H"]].to_numpy()
    mk = inv / inv.sum(axis=1, keepdims=True)
    return {(l, h, a): r(m, 3) for l, h, a, m in zip(fx["league"], fx["home"], fx["away"], mk)}


def league_fixtures(matches: pd.DataFrame, pairs: dict, as_of: pd.Timestamp) -> dict:
    rem = fixtures_mod.remaining(matches, list(LEAGUES))
    market = market_table()
    today = pd.Timestamp(as_of.date())
    out = {}
    for code in LEAGUES:
        rows = []
        for x in rem[rem["league"] == code].itertuples():
            rec = {"md": int(x.matchday), "d": x.date.strftime("%Y-%m-%d"), "t": x.time, "h": x.home, "a": x.away,
                   "p": pairs[code][f"{x.home}|{x.away}"][:3]}
            if (code, x.home, x.away) in market:
                rec["mk"] = market[(code, x.home, x.away)]
            if x.date < today:
                rec["late"] = 1  # still unplayed although its published date has passed: postponed
            rows.append(rec)
        out[code] = rows
    return out


def europe_block(comp: str, name: str, id_to_fd: dict, as_of: pd.Timestamp) -> dict:
    probs = pd.read_csv(TABLES / f"europe_{comp}_probabilities.csv")
    fx = pd.read_csv(TABLES / f"europe_{comp}_fixtures.csv")
    meta = json.loads((TABLES / f"europe_{comp}_meta.json").read_text())
    ids = CLUB_IDS[comp]

    def key(t):  # clubs of the five leagues keep their football-data name, so one club is one key everywhere
        return id_to_fd.get(ids[t], t) if ids.get(t) else t
    teams = [{"k": key(x.team), "pts": int(x.pts_now), "n": int(x.played), "exp": round(float(x.exp_points), 2),
              "top8": round(float(x.top8), 4), "top24": round(float(x.top24), 4), "r16": round(float(x.R16), 4),
              "qf": round(float(x.QF), 4), "sf": round(float(x.SF), 4), "f": round(float(x.Final), 4),
              "w": round(float(x.Winner), 4), "assumed": bool(x.assumed)} for x in probs.itertuples()]
    today = as_of.strftime("%Y-%m-%d")
    # Wikipedia can lag: a match whose date has passed but that has no score is "played, not recorded yet"
    missing = int((fx["date"] < today).sum())
    fixtures = [{"md": int(x.md), "d": x.date, "t": x.time if isinstance(x.time, str) else "", "h": key(x.home), "a": key(x.away),
                 "p": [round(float(x.p_away), 4), round(float(x.p_draw), 4), round(float(x.p_home), 4)]}
                for x in fx.itertuples() if x.date >= today]
    return {"name": name, "played": meta["played"], "total": meta["total"], "missing": missing, "n_assumed": meta["n_assumed"],
            "teams": teams, "fixtures": fixtures}


def previous_results(n_recent: int = 150) -> dict:
    """Out-of-sample (walk-forward) forecasts vs what happened, most recent matches first.
    These are re-computed after the fact with only the information available before each match;
    they are not forecasts that were published in advance."""
    d = pd.read_parquet(TABLES / "final_test_predictions.parquet").sort_values(["date", "match_id"])
    M, B = "LogReg (final)", "Bookmaker Bet365 (reference)"
    P = d[[f"{M}|away", f"{M}|draw", f"{M}|home"]].to_numpy()
    Q = d[[f"{B}|away", f"{B}|draw", f"{B}|home"]].to_numpy()
    F = d[["Class frequency (naive)|away", "Class frequency (naive)|draw", "Class frequency (naive)|home"]].to_numpy()
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
    rec = d.tail(n_recent).iloc[::-1]
    rows = [[x.date.strftime("%Y-%m-%d"), x.league, x.home, x.away, int(x.home_goals), int(x.away_goals), int(x.y),
             [round(float(v), 3) for v in P[i]]] for x, i in zip(rec.itertuples(), range(len(d) - 1, len(d) - 1 - len(rec), -1))]
    return {"last": last.strftime("%Y-%m-%d"), "from": d["date"].min().strftime("%Y-%m-%d"), "n_total": int(len(d)),
            "summary": {k: summary(v) for k, v in windows.items()}, "calibration": bins, "rows": rows}


def arena() -> dict:
    """Cumulative log loss over the test matches (weekly points) and the final scores."""
    pred = pd.read_parquet(TABLES / "final_test_predictions.parquet").sort_values(["date", "match_id"]).reset_index(drop=True)
    models = ["Class frequency (naive)", "Elo (LogReg)", "LogReg (final)", "Gradient boosting", "MLP deep (5 seeds)",
              "Ensemble LogReg+GB+MLP", "Bookmaker Bet365 (reference)"]
    y = pred["y"].to_numpy()
    week = ((pred["date"] - pred["date"].min()).dt.days // 7).to_numpy()
    last_of_week = np.r_[np.flatnonzero(np.diff(week)), len(pred) - 1]
    out = {"dates": [pred["date"].iloc[i].strftime("%Y-%m-%d") for i in last_of_week],
           "n": [int(i + 1) for i in last_of_week], "curves": {}}
    for m in models:
        P = pred[[f"{m}|away", f"{m}|draw", f"{m}|home"]].to_numpy()
        loss = -np.log(np.clip(P[np.arange(len(y)), y], 1e-15, 1))
        out["curves"][m] = [round(float(np.cumsum(loss)[i] / (i + 1)), 4) for i in last_of_week]
    table = pd.read_csv(TABLES / "final_test_table.csv", index_col=0)
    out["final"] = {m: {"log_loss": float(table.loc[m, "log_loss"]), "lo": float(table.loc[m, "log_loss_ci_low"]),
                        "hi": float(table.loc[m, "log_loss_ci_high"]), "accuracy": float(table.loc[m, "accuracy"]),
                        "brier": float(table.loc[m, "brier"])} for m in models}
    return out


def main() -> None:
    SITE.mkdir(exist_ok=True)
    prod = joblib.load(Path("models_store") / "production.joblib")
    boot = joblib.load(Path("models_store") / "bootstrap_logreg.joblib")
    lr, sigma, as_of = prod["logreg"], boot["sigma"], prod["as_of"]
    state = pd.read_parquet(PROCESSED / "team_state.parquet")
    matches = pd.read_parquet(PROCESSED / "matches.parquet")
    feats = pd.read_parquet(PROCESSED / "features.parquet", columns=["date", "league", "home", "away", "elo_home", "elo_away"])

    out = {"as_of": as_of.strftime("%Y-%m-%d"), "data_through": matches["date"].max().strftime("%Y-%m-%d"), "season": "2026/27",
           "sigma": sigma, "families": list(FAMILIES_SHORT.values()), "display": DISPLAY, "leagues": {}}
    # Elo timeline (monthly, last value of the month)
    long = pd.concat([feats[["date", "league", "home", "elo_home"]].set_axis(["date", "league", "team", "elo"], axis=1),
                      feats[["date", "league", "away", "elo_away"]].set_axis(["date", "league", "team", "elo"], axis=1)])
    long["month"] = long["date"].dt.to_period("M")
    months = pd.period_range("2010-08", long["month"].max(), freq="M")
    out["elo_months"] = [str(m) for m in months]

    pairs_all = {}
    for code, cfg in LEAGUES.items():
        teams = sorted(state.loc[code].index)
        pairs_all[code] = pair_table(code, teams, state, lr, boot, sigma)
        summ = pd.read_csv(TABLES / f"sim_{code}_summary.csv", index_col=0)
        posp = pd.read_csv(TABLES / f"sim_{code}_position_probs.csv", index_col=0)
        elo_rank = state.loc[code]["elo"].rank(ascending=False, method="min").astype(int)
        tinfo = {}
        for t in teams:
            series = long[(long.league == code) & (long.team == t)].groupby("month")["elo"].last().reindex(months)
            series = series.ffill(limit=5)  # no match in June/July (or during the 2020 lockdown): the rating simply stays
            series.iloc[-1] = float(state.loc[(code, t), "elo"])  # the stored value is pre-match; end on today's rating
            tinfo[t] = {"elo": round(float(state.loc[(code, t), "elo"]), 1), "elo_rank": int(elo_rank[t]),
                        "sv": None if pd.isna(state.loc[(code, t), "h_sv_total_rel"]) else round(float(state.loc[(code, t), "h_sv_total_rel"]), 2),
                        "pts10": round(float(state.loc[(code, t), "h_pts_10"]), 2),
                        "gf10": round(float(state.loc[(code, t), "h_gf_10"]), 2), "ga10": round(float(state.loc[(code, t), "h_ga_10"]), 2),
                        "elo_series": [None if pd.isna(v) else int(round(v)) for v in series]}
        played = summ["matches_played"]
        out["leagues"][code] = {
            "name": cfg["name"], "teams": teams, "info": tinfo,
            "slots": {k: cfg[k] for k in ("champions_league", "europe", "relegation")},
            "rounds": {"total": 2 * (len(teams) - 1), "played": int(played.max())},
            "sim": {"summary": {t: [round(float(summ.loc[t, c]), 4) for c in
                                    ("exp_points", "pts_p10", "pts_p90", "exp_position", "p_title", "p_champions_league",
                                     "p_top4", "p_top6", "p_europe", "p_relegation", "points_now", "matches_played")] for t in teams},
                    "pos_probs": {t: r(posp.loc[t, :], 4) for t in teams}}}
    out["pairs"] = pairs_all
    out["fixtures"] = league_fixtures(matches, pairs_all, as_of)

    club_map = pd.read_csv(PROCESSED / "club_map.csv").drop_duplicates("club_id").set_index("club_id")["team"].to_dict()
    out["europe"] = {"ucl": europe_block("ucl", "Champions League", club_map, as_of),
                     "uel": europe_block("uel", "Europa League", club_map, as_of)}
    out["arena"] = arena()
    out["prev"] = previous_results()

    (SITE / "data.js").write_text("window.OLDATA = " + json.dumps(out, separators=(",", ":"), ensure_ascii=False) + ";\n", encoding="utf8")
    n_fx = sum(len(v) for v in out["fixtures"].values())
    print("site/data.js", round((SITE / "data.js").stat().st_size / 1e6, 2), "MB;", sum(len(v) for v in pairs_all.values()),
          "pairs;", n_fx, "league fixtures to play")


if __name__ == "__main__":
    main()
