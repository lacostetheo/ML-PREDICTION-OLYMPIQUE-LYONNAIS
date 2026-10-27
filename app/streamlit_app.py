"""Streamlit dashboard — match predictor and season simulation.

Run the pipeline first (see README / Makefile), then:  streamlit run app/streamlit_app.py
Everything shown comes from the project's own models and pre-computed simulations;
nothing here is betting advice.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402

from olpred.interpret import family_contributions  # noqa: E402
from olpred.predict.league import LEAGUES  # noqa: E402
from olpred.predict.simulate import LinearSim  # noqa: E402
from olpred.predict.state import fixture_frame  # noqa: E402
from olpred.viz import style  # noqa: E402

PROCESSED, TABLES, STORE = ROOT / "data/processed", ROOT / "reports/tables", ROOT / "models_store"
OL = "Lyon"
st.set_page_config(page_title="Football match predictor", page_icon="⚽", layout="wide")
style.apply()


@st.cache_resource(show_spinner="Loading models…")
def load_models():
    prod = joblib.load(STORE / "production.joblib")
    state = pd.read_parquet(PROCESSED / "team_state.parquet")
    boot = joblib.load(STORE / "bootstrap_logreg.joblib") if (STORE / "bootstrap_logreg.joblib").exists() else None
    return prod, state, boot


@st.cache_data
def load_sim(league: str):
    z = np.load(PROCESSED / f"sim_{league}.npz")
    return z["positions"], z["points"], list(z["teams"])


@st.cache_data
def load_fixtures_market() -> pd.DataFrame | None:
    path = ROOT / "data/raw/football_data/fixtures.csv"
    if not path.exists():
        return None
    from olpred.data.football_data import load_fixtures
    f = load_fixtures(path.parent)
    inv = 1.0 / f[["B365A", "B365D", "B365H"]]
    f[["mkt_away", "mkt_draw", "mkt_home"]] = inv.div(inv.sum(axis=1), axis=0)
    return f


def need_pipeline() -> bool:
    needed = [STORE / "production.joblib", PROCESSED / "team_state.parquet"]
    if all(p.exists() for p in needed):
        return False
    st.error("Model files not found. Run the pipeline first: `make pipeline` "
             "(see README → Reproducibility).")
    return True


def probability_bars(p_home, p_draw, p_away, home, away) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 1.9))
    left = 0
    for v, color, lab in [(p_home, style.BLUE, f"{home} win"), (p_draw, style.NEUTRAL, "Draw"),
                          (p_away, style.ORANGE, f"{away} win")]:
        ax.barh(0, v, left=left, color=color, height=0.6, edgecolor=style.SURFACE, linewidth=2)
        text = f"{lab}\n{v:.0%}" if v >= 0.16 else f"{v:.0%}"  # short label when the segment is narrow
        ax.text(left + v / 2, 0, text, ha="center", va="center", color="white",
                fontsize=9 if v >= 0.16 else 8, fontweight="bold")
        left += v
    ax.set_xlim(0, 1); ax.axis("off")
    fig.tight_layout()
    return fig


def predictor_tab(prod, state, boot) -> None:
    lr = prod["logreg"]
    st.subheader("Match predictor")
    st.caption(f"Team strengths are a snapshot as of **{prod['as_of']:%d %b %Y}** "
               "(Elo, recent form, squad market value, shots). To refresh: "
               "`python -m olpred.run_ol YYYY-MM-DD`.")
    c1, c2, c3 = st.columns([1.2, 1.2, 1.0])
    league = c1.selectbox("League", list(LEAGUES), format_func=lambda k: LEAGUES[k]["name"],
                          index=list(LEAGUES).index("F1"))
    teams = sorted(state.loc[league].index)
    team = c1.selectbox("Team", teams, index=teams.index(OL) if OL in teams else 0)
    venue = c2.radio("Venue", ["Home", "Away"], horizontal=True)
    opponent = c2.selectbox("Opponent", [t for t in teams if t != team],
                            index=0)
    rest_t = c3.slider(f"Days of rest — {team}", 2, 14, 7)
    rest_o = c3.slider(f"Days of rest — {opponent}", 2, 14, 7)
    home, away = (team, opponent) if venue == "Home" else (opponent, team)
    rest_h, rest_a = (rest_t, rest_o) if venue == "Home" else (rest_o, rest_t)

    fx = pd.DataFrame({"league": [league], "home": [home], "away": [away]})
    X = fixture_frame(fx, state)
    X["h_rest_days"], X["a_rest_days"] = float(rest_h), float(rest_a)
    X["d_rest_days"] = X["h_rest_days"] - X["a_rest_days"]
    p = lr.predict_proba(X)[0]
    p_away, p_draw, p_home = p
    st.pyplot(probability_bars(p_home, p_draw, p_away, home, away), width="content")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{home} expected points", f"{3 * p_home + p_draw:.2f}")
    m2.metric(f"{away} expected points", f"{3 * p_away + p_draw:.2f}")
    m3.metric("Elo difference (home − away)", f"{X['elo_diff'].iloc[0]:+.0f}")

    if boot is not None:
        sim = [LinearSim(m, X) for m in boot["models"]]
        zero = np.zeros((1, 1))
        model_only = np.array([s.probs(zero, zero)[0, 0] for s in sim])
        rng = np.random.default_rng(0)
        total = np.concatenate([s.probs(rng.normal(0, boot["sigma"], (20, 1)),
                                        rng.normal(0, boot["sigma"], (20, 1)))[:, 0] for s in sim])
        rows = []
        for name, k in [(f"{home} win", 2), ("Draw", 1), (f"{away} win", 0)]:
            rows.append({"Outcome": name, "Probability": f"{p[k]:.1%}",
                         "90 % interval (model)": "{:.1%} – {:.1%}".format(*np.percentile(model_only[:, k], [5, 95])),
                         "90 % interval (model + team ratings)":
                             "{:.1%} – {:.1%}".format(*np.percentile(total[:, k], [5, 95]))})
        st.markdown("**Uncertainty on the probabilities**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Model: 100 refits on resampled weeks of history. Ratings: each team's Elo shaken by "
                   f"±{boot['sigma']:.0f} points. The match result itself stays random — these intervals "
                   "describe how sure the model is of its own probabilities.")

    fix = load_fixtures_market()
    if fix is not None:
        hit = fix[(fix["league"] == league) & (fix["home"] == home) & (fix["away"] == away)]
        if len(hit):
            h = hit.iloc[0]
            st.info(f"Bookmaker (Bet365, margin removed) for this real fixture on {h['date']:%d %b}: "
                    f"{home} {h['mkt_home']:.0%} · draw {h['mkt_draw']:.0%} · {away} {h['mkt_away']:.0%}. "
                    "In testing the bookmaker beat our model, and adding the model to it did not help.")

    st.markdown("**Key factors** — contribution to *home advantage in the win-vs-loss odds*, "
                "relative to an average Big-5 match (positive = favours the home side)")
    contrib = family_contributions(lr, X).iloc[0].sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    colors = [style.BLUE if v >= 0 else style.ORANGE for v in contrib]
    ax.barh(contrib.index, contrib.values, color=colors, height=0.6)
    ax.axvline(0, color=style.NEUTRAL, lw=1)
    ax.set_xlabel(f"← favours {away}      log-odds contribution      favours {home} →")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    st.pyplot(fig, width="content")
    st.caption("Exact additive decomposition of the logistic regression. Correlated inputs share credit, "
               "so read families, not single coefficients.")


def simulation_tab(prod, state) -> None:
    st.subheader("Season simulation (10,000 Monte Carlo seasons)")
    c1, c2 = st.columns(2)
    league = c1.selectbox("League ", list(LEAGUES), format_func=lambda k: LEAGUES[k]["name"],
                          index=list(LEAGUES).index("F1"), key="sim_league")
    if not (PROCESSED / f"sim_{league}.npz").exists():
        st.warning("Simulation files not found — run `python -m olpred.run_season_sim`.")
        return
    positions, points, teams = load_sim(league)
    team = c2.selectbox("Team ", teams, index=teams.index(OL) if OL in teams else 0, key="sim_team")
    i, n = teams.index(team), len(teams)
    cfg = LEAGUES[league]
    pos, pts = positions[:, i], points[:, i]
    cols = st.columns(6)
    for col, (label, v) in zip(cols, [("Title", (pos == 1).mean()),
                                      (f"Champions League (top {cfg['champions_league']})",
                                       (pos <= cfg["champions_league"]).mean()),
                                      ("Top 4", (pos <= 4).mean()), ("Top 6", (pos <= 6).mean()),
                                      (f"Europe (top {cfg['europe']})", (pos <= cfg["europe"]).mean()),
                                      ("Relegation", (pos > n - cfg["relegation"]).mean())]):
        col.metric(label, f"{v:.1%}")
    st.caption("European slots and relegation places are assumptions (configs/leagues.json), not verified facts.")

    left, right = st.columns(2)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    freq = np.bincount(pos, minlength=n + 1)[1:] / len(pos)
    ax.bar(range(1, n + 1), freq, color=style.BLUE, width=0.75)
    ax.set_xlabel("Final position"); ax.set_ylabel("Probability"); ax.set_xticks(range(1, n + 1, 2 if n > 18 else 1))
    ax.set_title(f"{team}: final position", fontsize=11); ax.grid(axis="x", visible=False)
    fig.tight_layout(); left.pyplot(fig, width="content")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    vals, cnt = np.unique(pts, return_counts=True)
    ax.bar(vals, cnt / len(pts), color=style.BLUE, width=0.85)
    ax.axvline(pts.mean(), color=style.INK, lw=1.4)
    ax.set_xlabel("Final points"); ax.set_ylabel("Probability")
    ax.set_title(f"{team}: mean {pts.mean():.1f}, 80 % in [{np.percentile(pts, 10):.0f}, {np.percentile(pts, 90):.0f}]",
                 fontsize=11)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); right.pyplot(fig, width="content")

    summary = pd.read_csv(TABLES / f"sim_{league}_summary.csv", index_col=0)
    show = summary[["points_now", "matches_played", "exp_points", "pts_p10", "pts_p90", "p_title",
                    "p_champions_league", "p_top4", "p_top6", "p_relegation"]].copy()
    for c in show.columns[5:]:
        show[c] = (show[c] * 100).round(1)
    show.columns = ["Points now", "Played", "Exp. points", "P10", "P90", "Title %", "CL %", "Top 4 %",
                    "Top 6 %", "Relegation %"]
    st.markdown("**Whole league**")
    st.dataframe(show.round(1), width="stretch")


def about_tab() -> None:
    st.subheader("About these forecasts")
    st.markdown("""
**Model.** Multinomial logistic regression on 100 features (Elo rating, recent form, shots,
squad market value, schedule), trained on ~29,000 matches from the five major European
leagues (2011 → 2026) and refit as results arrive. In walk-forward testing on 2023-2026 it
reached a log loss of **0.979** (naive: 1.075; bookmaker: 0.966); boosting and neural networks
were statistically no better.

**What it can and cannot do**
- Probabilities are well calibrated, but football outcomes are very noisy: even a 65 % favourite
  fails to win one match in three. Draws are essentially unpredictable.
- The bookmaker's prices were better than the model, and mixing the model into them added
  nothing: this model contains no information that the market lacks.
- Not modelled: injuries, suspensions, line-ups, coaching changes, European fixtures.
- Season-simulation intervals were slightly too narrow in the backtest (72 % coverage for
  nominal 80 %).

**Not betting advice.** This project is an academic exercise in probabilistic forecasting.
""")


def main() -> None:
    st.title("⚽ Football match & season predictor")
    st.caption("Olympique Lyonnais case study — Big-5 leagues, walk-forward validated, uncertainty included.")
    if need_pipeline():
        st.stop()
    prod, state, boot = load_models()
    t1, t2, t3 = st.tabs(["Match predictor", "OL season simulation", "About"])
    with t1:
        predictor_tab(prod, state, boot)
    with t2:
        simulation_tab(prod, state)
    with t3:
        about_tab()


main()
