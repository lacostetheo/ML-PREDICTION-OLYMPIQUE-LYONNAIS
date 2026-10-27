"""Monte Carlo of the 2026/27 Champions League (league phase -> knockout -> final).

Inputs: the real fixture list and results so far (``data/raw/calendar/ucl_2026_27.csv``,
copied from Wikipedia), the European strength model of ``olpred.europe`` and the clubs'
current squad values / domestic form.  Run:  python -m olpred.europe_sim

Assumptions (all flagged in docs/europe_ucl.md): bracket structure follows the 2024/25
pattern but pairings inside a tier are drawn at random; knockout ties are decided over two legs
(final: one neutral match); extra time / penalties are close to a coin flip; each club's strength
gets a random shock (sd 0.15 in log squad value) to reflect uncertainty.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from olpred.europe import CLUB_IDS, EuropeModel, PPG_WINDOW, domestic_ppg

TABLES = Path("reports/tables")
CAL = Path("data/raw/calendar/ucl_2026_27.csv")
SIGMA = 0.15
N_SIMS = 20_000
PEN_SHRINK = 0.5   # extra time / penalties: win prob moves only half-way from 50 % towards the stronger side
# winners of the knockout play-offs, indexed by the seeds they meet in the round of 16
PLAYOFF = {"P1": ((9, 10), (23, 24)), "P2": ((11, 12), (21, 22)),
           "P3": ((13, 14), (19, 20)), "P4": ((15, 16), (17, 18))}
R16_SEED_PATH = {1: "P4", 2: "P3", 3: "P1", 4: "P2", 5: "P1", 6: "P2", 7: "P3", 8: "P4"}


def team_inputs(teams: list[str]) -> pd.DataFrame:
    """log squad value (snapshot of 1 Nov 2026 = valuations to June 2026) and domestic ppg."""
    snaps = pd.read_parquet("data/processed/euro_snaps.parquet")
    snap = snaps[snaps["month"] == snaps["month"].max()].set_index("club_id")
    games = pd.read_csv("data/raw/transfermarkt/games.csv.gz", parse_dates=["date"])
    ppg_tm = domestic_ppg(games).sort_values("date").groupby("club_id").tail(1).set_index("club_id")["ppg"]
    # Big-5 clubs: current-season results are in the football-data tables (newer than Transfermarkt)
    m = pd.read_parquet("data/processed/matches.parquet")
    cmap = pd.read_csv("data/processed/club_map.csv").set_index(["league", "team"])["club_id"]
    rows = []
    for _, r in m.iterrows():
        for side, pts in (("home", r["pts_home"]), ("away", r["pts_away"])):
            cid = cmap.get((r["league"], r[side]))
            if cid is not None and not pd.isna(cid):
                rows.append((int(cid), r["date"], pts))
    long = pd.DataFrame(rows, columns=["club_id", "date", "pts"]).sort_values(["club_id", "date"])
    ppg_big5 = long.groupby("club_id")["pts"].apply(lambda s: s.tail(PPG_WINDOW).mean())

    out = []
    for t in teams:
        cid = CLUB_IDS[t]
        lv = np.log(snap.loc[cid, "sv_total"]) if cid in snap.index else np.nan
        ppg = ppg_big5.get(cid, ppg_tm.get(cid, np.nan)) if cid else np.nan
        src = "football-data (2026/27)" if cid in ppg_big5.index else ("Transfermarkt (2025/26)" if cid else "assumed")
        out.append({"team": t, "lv": lv, "ppg": ppg, "ppg_source": src})
    d = pd.DataFrame(out)
    d["assumed"] = d["lv"].isna()
    # clubs absent from the dataset (AEK, Sabah, Slovan Bratislava): weakest observed value, median ppg
    d["lv"] = d["lv"].fillna(d["lv"].min())
    d["ppg"] = d["ppg"].fillna(d["ppg"].median())
    return d


def prob_tensor(model: EuropeModel, lv: np.ndarray, ppg: np.ndarray) -> np.ndarray:
    n = len(lv)
    h, a = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    X = pd.DataFrame({"d_value": (lv[h] - lv[a]).ravel(), "d_ppg": (ppg[h] - ppg[a]).ravel()})
    return model.predict_proba(X).reshape(n, n, 3)  # [home, away, (away win, draw, home win)]


def sample_scores(rng, outcome: np.ndarray, pools: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Goals (home, away) drawn from the empirical pool of European matches with this outcome."""
    hg = np.zeros(outcome.shape, int); ag = np.zeros(outcome.shape, int)
    for k, pool in pools.items():
        idx = np.where(outcome == k)
        j = rng.integers(0, len(pool), len(idx[0]))
        hg[idx], ag[idx] = pool[j, 0], pool[j, 1]
    return hg, ag


def simulate(teams, lv0, ppg, model, fixtures, pools, n_sims=N_SIMS, sigma=SIGMA, seed=1):
    rng = np.random.default_rng(seed)
    n = len(teams)
    ix = {t: i for i, t in enumerate(teams)}
    hi = fixtures["home"].map(ix).to_numpy(); ai = fixtures["away"].map(ix).to_numpy()
    played = fixtures["hg"].notna().to_numpy()
    base_pts = np.zeros(n); base_gd = np.zeros(n); base_gf = np.zeros(n)
    for r in fixtures[played].itertuples():
        h, a, hg, ag = ix[r.home], ix[r.away], int(r.hg), int(r.ag)
        base_pts[h] += 3 * (hg > ag) + (hg == ag); base_pts[a] += 3 * (ag > hg) + (hg == ag)
        base_gd[h] += hg - ag; base_gd[a] += ag - hg; base_gf[h] += hg; base_gf[a] += ag
    todo = np.where(~played)[0]

    stage = {k: np.zeros(n) for k in ["top8", "top24", "R16", "QF", "SF", "Final", "Winner"]}
    pos_hist = np.zeros((n, n))
    exp_pts = np.zeros(n)
    for s in range(n_sims):
        lv = lv0 + rng.normal(0, sigma, n)
        P = prob_tensor(model, lv, ppg)
        # --- league phase: remaining matches
        p = P[hi[todo], ai[todo]]
        u = rng.random(len(todo))
        out = (u > p[:, 0]).astype(int) + (u > p[:, 0] + p[:, 1]).astype(int)  # 0 away, 1 draw, 2 home
        hg, ag = sample_scores(rng, out, pools)
        pts = base_pts.copy(); gd = base_gd.copy(); gf = base_gf.copy()
        np.add.at(pts, hi[todo], 3 * (out == 2) + (out == 1)); np.add.at(pts, ai[todo], 3 * (out == 0) + (out == 1))
        np.add.at(gd, hi[todo], hg - ag); np.add.at(gd, ai[todo], ag - hg)
        np.add.at(gf, hi[todo], hg); np.add.at(gf, ai[todo], ag)
        order = np.lexsort((rng.random(n), -gf, -gd, -pts))  # points, goal diff, goals for, then random
        rank = np.empty(n, int); rank[order] = np.arange(1, n + 1)
        pos_hist[np.arange(n), rank - 1] += 1
        exp_pts += pts
        stage["top8"] += rank <= 8; stage["top24"] += rank <= 24
        seat = {int(rank[i]): i for i in range(n)}  # seed -> team index

        def tie(a, b, home_second_leg=None):
            """Two legs (a hosts the first), returns winner index."""
            p1 = P[a, b]; p2 = P[b, a]
            r1 = rng.random(); o1 = 0 if r1 < p1[0] else (1 if r1 < p1[0] + p1[1] else 2)
            r2 = rng.random(); o2 = 0 if r2 < p2[0] else (1 if r2 < p2[0] + p2[1] else 2)
            g1 = sample_scores(rng, np.array([o1]), pools); g2 = sample_scores(rng, np.array([o2]), pools)
            a_goals = g1[0][0] + g2[1][0]; b_goals = g1[1][0] + g2[0][0]
            if a_goals != b_goals:
                return a if a_goals > b_goals else b
            pa = (P[a, b, 2] + P[b, a, 0]) / 2; pb = (P[a, b, 0] + P[b, a, 2]) / 2  # neutral-ish strength
            q = pa / (pa + pb)
            q = 0.5 + PEN_SHRINK * (q - 0.5)
            return a if rng.random() < q else b

        # --- play-offs: seeded (9-16) host the second leg
        po = {}
        for k, ((s1, s2), (u1, u2)) in PLAYOFF.items():
            po[k] = []
            for sd, ud in ((s1, u2), (s2, u1)):
                po[k].append(tie(seat[ud], seat[sd]))
        # --- round of 16: seeds 1-8 versus the playoff winners (drawn within their pair of paths)
        # explicit mapping: every playoff produces two winners; two feed the top-4 seeds, two feed 5-8
        w = {k: list(rng.permutation(po[k])) for k in po}
        pairing = {1: w["P4"][0], 2: w["P3"][0], 3: w["P1"][0], 4: w["P2"][0],
                   5: w["P1"][1], 6: w["P2"][1], 7: w["P3"][1], 8: w["P4"][1]}
        r16_winners = []
        for sd, opp in pairing.items():
            stage["R16"][seat[sd]] += 1; stage["R16"][opp] += 1
            r16_winners.append(tie(opp, seat[sd]))  # seeded team hosts the second leg
        # --- QF / SF: random pairing of the winners (draw), higher seed hosts leg 2
        def knockout(ts, key):
            ts = list(rng.permutation(ts)); nxt = []
            for i in range(0, len(ts), 2):
                a, b = ts[i], ts[i + 1]
                if rank[a] < rank[b]:
                    a, b = b, a
                nxt.append(tie(a, b))
            for t in ts:
                stage[key][t] += 1
            return nxt
        qf = knockout(r16_winners, "QF")
        sf = knockout(qf, "SF")
        a, b = sf
        stage["Final"][a] += 1; stage["Final"][b] += 1
        pa = (P[a, b, 2] + P[b, a, 0]) / 2; pb = (P[a, b, 0] + P[b, a, 2]) / 2
        win = a if rng.random() < pa / (pa + pb) else b  # neutral venue, 90 min + ET/pens folded in
        stage["Winner"][win] += 1
    res = pd.DataFrame({"team": teams, "exp_points": exp_pts / n_sims, **{k: v / n_sims for k, v in stage.items()}})
    return res.sort_values("Winner", ascending=False).reset_index(drop=True), pos_hist / n_sims


def main() -> None:
    fx = pd.read_csv(CAL)
    teams = sorted(set(fx["home"]) | set(fx["away"]))
    inp = team_inputs(teams)
    d = pd.read_parquet("data/processed/euro_matches.parquet")
    model = EuropeModel().fit(d)
    pools = {k: d.loc[d["y"] == k, ["home_club_goals", "away_club_goals"]].to_numpy(int) for k in (0, 1, 2)}
    res, _ = simulate(teams, inp["lv"].to_numpy(), inp["ppg"].to_numpy(), model, fx, pools)
    res = res.merge(inp[["team", "ppg", "ppg_source", "assumed"]], on="team")
    res.round(4).to_csv(TABLES / "europe_ucl_probabilities.csv", index=False)
    # sensitivity to the strength-shock size
    sens = {}
    for sg in (0.0, 0.3):
        r, _ = simulate(teams, inp["lv"].to_numpy(), inp["ppg"].to_numpy(), model, fx, pools, n_sims=5000, sigma=sg, seed=2)
        sens[str(sg)] = r.set_index("team")["Winner"].round(4).head(8).to_dict()
    json.dump({"played": int(fx["hg"].notna().sum()), "total": len(fx), "sensitivity_title": sens},
              open(TABLES / "europe_ucl_meta.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(res.head(16).round(3).to_string()); print(sens)


if __name__ == "__main__":
    main()
