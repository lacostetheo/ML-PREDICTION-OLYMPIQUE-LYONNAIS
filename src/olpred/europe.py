"""European competitions: strength of a club = squad market value + domestic form.

Domestic leagues are only comparable through the matches clubs play against each other,
so this model is trained on past Champions League / Europa League / Conference League
matches (Transfermarkt dataset, 2012-2025) and tested walk-forward.

Features of a match (home club h, away club a), all known before kick-off:
  d_value = log squad value(h) - log squad value(a)      (monthly snapshot, as for the leagues)
  d_ppg   = domestic points per match (h) - same (a)      (last 20 league matches)
Target: away win / draw / home win (same coding as the league models).

Run:  python -m olpred.europe
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from olpred.data import transfermarkt as tm
from olpred.evaluation.metrics import summarize
from olpred.features.squad_value import monthly_snapshots

TABLES = Path("reports/tables")
PPG_WINDOW = 20
# Transfermarkt club ids of the 2026/27 league-phase clubs (None = not in the dataset)
CLUB_IDS = {
    "ucl": {
        "AEK Athens": None, "Arsenal": 11, "Aston Villa": 405, "Atlético Madrid": 13, "Barcelona": 131,
        "Bayern Munich": 27, "Bodø/Glimt": 501, "Borussia Dortmund": 16, "Club Brugge": 2282,
        "Como": 1047, "Fenerbahçe": 36, "Feyenoord": 234, "Galatasaray": 141, "Inter Milan": 46,
        "LASK": 413, "Lens": 826, "Lille": 1082, "Liverpool": 31, "Manchester City": 281,
        "Manchester United": 985, "Napoli": 6195, "PSV Eindhoven": 383, "Paris Saint-Germain": 583,
        "Porto": 720, "RB Leipzig": 23826, "Real Betis": 150, "Real Madrid": 418, "Roma": 12,
        "Sabah": None, "Shakhtar Donetsk": 660, "Slavia Prague": 62, "Slovan Bratislava": None,
        "Sporting CP": 336, "VfB Stuttgart": 79, "Viking": 239, "Villarreal": 1050},
    "uel": {
        "AZ": 1090, "Anderlecht": 58, "Ararat-Armenia": None, "Bayer Leverkusen": 15, "Benfica": 294,
        "Beşiktaş": 114, "Bournemouth": 989, "Celje": None, "Celta Vigo": 940, "Celtic": 371,
        "Crystal Palace": 873, "Dinamo Zagreb": 419, "Ferencváros": None, "Hapoel Be'er Sheva": None,
        "Jagiellonia Białystok": 2300, "Juventus": 506, "Lech Poznań": 238, "Levski Sofia": None,
        "Lillestrøm": 353, "Lyon": 1041, "Marseille": 244, "Milan": 5, "NEC": 467, "OFI": None,
        "Olympiacos": 683, "Omonia": None, "Real Sociedad": 681, "Red Bull Salzburg": 409,
        "Rennes": 273, "Sparta Prague": 197, "Sturm Graz": 122, "Sunderland": 289, "TSG Hoffenheim": 533,
        "Torreense": None, "Union Saint-Gilloise": 3948, "Viktoria Plzeň": 941}}


def domestic_ppg(games: pd.DataFrame) -> pd.DataFrame:
    """Per (club, domestic match date): points per match over the PPG_WINDOW domestic league
    matches up to and including that date. ``attach_features`` reads it with a strictly-earlier
    as-of join, so a European match only sees domestic matches played before its date."""
    d = games[games["competition_type"] == "domestic_league"].dropna(
        subset=["home_club_goals", "away_club_goals"])
    h = pd.DataFrame({"club_id": d["home_club_id"], "date": d["date"],
                      "pts": np.select([d["home_club_goals"] > d["away_club_goals"],
                                        d["home_club_goals"] == d["away_club_goals"]], [3, 1], 0)})
    a = pd.DataFrame({"club_id": d["away_club_id"], "date": d["date"],
                      "pts": np.select([d["away_club_goals"] > d["home_club_goals"],
                                        d["home_club_goals"] == d["away_club_goals"]], [3, 1], 0)})
    long = pd.concat([h, a]).sort_values(["club_id", "date"]).reset_index(drop=True)
    long["ppg"] = (long.groupby("club_id")["pts"]
                   .transform(lambda s: s.rolling(PPG_WINDOW, min_periods=8).mean()))
    return long[["club_id", "date", "ppg"]].drop_duplicates(["club_id", "date"], keep="last")


def european_matches(games: pd.DataFrame) -> pd.DataFrame:
    e = games[games["competition_id"].isin(["CL", "EL", "UCOL"])].dropna(
        subset=["home_club_goals", "away_club_goals"]).copy()
    e["y"] = np.select([e["home_club_goals"] > e["away_club_goals"],
                        e["home_club_goals"] == e["away_club_goals"]], [2, 1], 0)
    e["month"] = e["date"].dt.to_period("M").dt.to_timestamp()
    return e


def attach_features(e: pd.DataFrame, snaps: pd.DataFrame, ppg: pd.DataFrame) -> pd.DataFrame:
    s = snaps.assign(lv=np.log(snaps["sv_total"].where(snaps["sv_total"] > 0)))
    s = s[s["sv_n"] >= 11][["club_id", "month", "lv"]]
    for side in ("home", "away"):
        e = e.merge(s.rename(columns={"club_id": f"{side}_club_id", "lv": f"lv_{side}"}),
                    on=[f"{side}_club_id", "month"], how="left")
        e = pd.merge_asof(e.sort_values("date"), ppg.sort_values("date").rename(
            columns={"club_id": f"{side}_club_id", "ppg": f"ppg_{side}"}),
            on="date", by=f"{side}_club_id", direction="backward", allow_exact_matches=False,
            tolerance=pd.Timedelta(days=120))
    e["d_value"] = e["lv_home"] - e["lv_away"]
    e["d_ppg"] = e["ppg_home"] - e["ppg_away"]
    return e


class EuropeModel:
    """Multinomial logistic regression on (d_value, d_ppg); d_ppg optional."""

    def __init__(self, cols=("d_value", "d_ppg"), C=1.0):
        self.cols, self.C = list(cols), C

    def fit(self, df):
        self.m_ = LogisticRegression(C=self.C, max_iter=1000).fit(df[self.cols], df["y"])
        return self

    def predict_proba(self, df):
        return self.m_.predict_proba(df[self.cols])


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    tm.download()
    games = tm.load("games")
    games["date"] = pd.to_datetime(games["date"])
    e = european_matches(games)
    clubs = set(e["home_club_id"]) | set(e["away_club_id"]) | {v for ids in CLUB_IDS.values() for v in ids.values() if v}
    snaps = monthly_snapshots(tm.load("player_valuations"), clubs, "2011-07-01", "2026-11-01")
    snaps.to_parquet("data/processed/euro_snaps.parquet")
    ppg = domestic_ppg(games)
    ppg.to_parquet("data/processed/euro_ppg.parquet")
    d = attach_features(e, snaps, ppg).dropna(subset=["d_value", "d_ppg"]).reset_index(drop=True)
    d.to_parquet("data/processed/euro_matches.parquet")
    print(f"European matches with both features: {len(d)} of {len(e)}")

    # walk-forward: refit before each test season 2023..2025 (seasons = start year)
    variants = {"Class frequency": [], "Squad value only": ["d_value"],
                "Domestic form only": ["d_ppg"], "Value + domestic form": ["d_value", "d_ppg"]}
    rows, store = [], {}
    for name, cols in variants.items():
        ps, ys = [], []
        for s in (2023, 2024, 2025):
            tr, te = d[d["season"] < s], d[d["season"] == s]
            if cols:
                p = EuropeModel(cols).fit(tr).predict_proba(te)
            else:
                p = np.tile(np.bincount(tr["y"], minlength=3) / len(tr), (len(te), 1))
            ps.append(p); ys.append(te["y"].to_numpy())
        p, y = np.vstack(ps), np.concatenate(ys)
        store[name] = (p, y)
        rows.append({"model": name, "n": len(y), **summarize(y, p)})
    res = pd.DataFrame(rows).set_index("model")
    res.round(4).to_csv(TABLES / "europe_walk_forward.csv")
    print(res.round(4).to_string())

    # paired block bootstrap (weeks) of the gain from domestic form
    from olpred.evaluation.metrics import paired_block_bootstrap, per_match_log_loss, week_blocks
    te = pd.concat([d[d["season"] == s] for s in (2023, 2024, 2025)])
    blocks = week_blocks(te["date"])
    out = {}
    for a, b in [("Value + domestic form", "Squad value only"), ("Squad value only", "Class frequency"),
                 ("Domestic form only", "Class frequency")]:
        la, lb = per_match_log_loss(store[a][1], store[a][0]), per_match_log_loss(store[b][1], store[b][0])
        out[f"{a} vs {b}"] = {k: round(float(v), 4) for k, v in paired_block_bootstrap(la, lb, blocks).items()}
    print(json.dumps(out, indent=1))
    json.dump(out, open(TABLES / "europe_bootstrap.json", "w"), indent=1)

    final = EuropeModel().fit(d)
    print("final coefficients (classes A,D,H):", final.m_.coef_.round(3).tolist(),
          final.m_.intercept_.round(3).tolist())


if __name__ == "__main__":
    main()
