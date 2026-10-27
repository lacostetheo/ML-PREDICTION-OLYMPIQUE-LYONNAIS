"""Figures for Phases 8-9 (OL case study and season simulation).
Run after run_ol and run_season_sim:  python -m olpred.viz.ol_plots"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from olpred.evaluation.metrics import reliability
from olpred.viz import style
from olpred.viz.baseline_plots import FIGURES, TABLES

OL = "Lyon"
RIVALS = [("Paris SG", style.ORANGE), ("Marseille", style.AQUA)]


def _ol_rows(feats: pd.DataFrame) -> pd.DataFrame:
    home = feats[feats["home"] == OL].assign(side="h")
    away = feats[feats["away"] == OL].assign(side="a")
    rows = pd.concat([home, away]).sort_values("date")
    p = rows["side"]
    out = pd.DataFrame({"date": rows["date"]})
    out["elo"] = np.where(p == "h", rows["elo_home"], rows["elo_away"])
    for f in ("pts_10", "xg_10", "xga_10", "gf_10", "ga_10"):
        out[f] = np.where(p == "h", rows[f"h_{f}"], rows[f"a_{f}"])
    return out


def elo_evolution() -> None:
    f = pd.read_parquet("data/processed/features.parquet", columns=["date", "league", "home", "away",
                                                                     "elo_home", "elo_away"])
    f = f[f["league"] == "F1"]
    long = pd.concat([f[["date", "home", "elo_home"]].set_axis(["date", "team", "elo"], axis=1),
                      f[["date", "away", "elo_away"]].set_axis(["date", "team", "elo"], axis=1)])
    def with_gaps(g: pd.DataFrame) -> pd.DataFrame:
        """Insert a break wherever the club was absent from the league (> 200 days)."""
        g = g.sort_values("date").reset_index(drop=True)
        gap = g["date"].diff().dt.days > 200
        if gap.any():
            blanks = g[gap].assign(elo=np.nan, date=lambda d: d["date"] - pd.Timedelta(days=1))
            g = pd.concat([g, blanks]).sort_values("date", kind="stable")
        return g

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for team, g in long.groupby("team"):
        if team not in (OL, *[r for r, _ in RIVALS]):
            g = with_gaps(g)
            ax.plot(g["date"], g["elo"], color=style.GRID, lw=1)
    for team, color in [(OL, style.BLUE), *RIVALS]:
        g = with_gaps(long[long["team"] == team])
        ax.plot(g["date"], g["elo"], color=color, lw=2)
        ax.text(g["date"].iloc[-1] + pd.Timedelta(days=60), g["elo"].iloc[-1], team, color=style.INK_2,
                va="center", fontsize=9)
    ax.set_ylabel("Elo rating (Ligue 1)"); ax.set_xlim(long["date"].min(), long["date"].max() + pd.Timedelta(days=500))
    ax.set_xticks(pd.to_datetime([f"{y}-01-01" for y in range(2011, 2027, 2)]))
    ax.set_xticklabels(range(2011, 2027, 2))
    ax.set_title("OL rating: usually just behind Paris SG, with a deep trough in 2023-24")
    fig.text(0.01, 0.01, "Grey: other Ligue 1 clubs. Elo computed from results of the league only.",
             color=style.INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); out = FIGURES / "ol_elo_evolution.png"
    fig.savefig(out); plt.close(fig); print(out)


def form_figure() -> None:
    f = pd.read_parquet("data/processed/features.parquet")
    d = _ol_rows(f); d = d[d["date"] >= "2022-08-01"]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.6), sharex=True)
    axes[0].plot(d["date"], d["pts_10"], color=style.BLUE, lw=2)
    axes[0].axhline(1.37, color=style.NEUTRAL, lw=1, ls="--")
    axes[0].text(d["date"].iloc[0], 1.42, "league average ≈ 1.37", color=style.INK_2, fontsize=8)
    axes[0].set_ylabel("Points per match\n(last 10)")
    axes[1].plot(d["date"], d["xg_10"] - d["xga_10"], color=style.BLUE, lw=2)
    axes[1].axhline(0, color=style.NEUTRAL, lw=1)
    axes[1].set_ylabel("Shot-based xG proxy\nfor − against (last 10)")
    axes[0].set_title("OL rolling performance, before each match")
    fig.text(0.01, 0.01, "xG proxy = 0.287 × shots on target (not real xG). Rolling 10 matches, computed before the match.",
             color=style.INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); out = FIGURES / "ol_form.png"
    fig.savefig(out); plt.close(fig); print(out)


def strength_profile() -> None:
    d = pd.read_csv(TABLES / "ol_strength_profile.csv")
    d = d.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for i, r in d.iterrows():
        ax.plot([0, r["percentile"]], [i, i], color=style.GRID, lw=2)
        ax.plot(r["percentile"], i, "o", color=style.BLUE, ms=9, mec=style.SURFACE, mew=2)
        ax.text(104, i, f"{int(r['rank'])}/{int(r['of'])}", va="center", color=style.INK_2, fontsize=9)
    ax.set_yticks(range(len(d)), d["dimension"], fontsize=9)
    ax.set_xlim(0, 112); ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Percentile among Ligue 1 clubs (100 = best); right: rank")
    ax.set_title("OL today: strong squad value and results, middling shot volume")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); out = FIGURES / "ol_strength_profile.png"
    fig.savefig(out); plt.close(fig); print(out)


def next_match() -> None:
    d = pd.read_csv(TABLES / "ol_next_round.csv")
    r = d[d["involves_OL"]].iloc[0]
    home = r["home"] == OL
    w, dr, l = (r["p_home"], r["p_draw"], r["p_away"]) if home else (r["p_away"], r["p_draw"], r["p_home"])
    mw, md, ml = (r["mkt_home"], r["mkt_draw"], r["mkt_away"]) if home else (r["mkt_away"], r["mkt_draw"], r["mkt_home"])
    key = "home" if home else "away"
    fig, ax = plt.subplots(figsize=(9, 2.9))
    for y, vals, name in [(1, (w, dr, l), "Model"), (0, (mw, md, ml), "Bookmaker (Bet365)")]:
        left = 0
        for v, color, lab in zip(vals, (style.BLUE, style.NEUTRAL, style.ORANGE), ("OL win", "Draw", "OL loss")):
            ax.barh(y, v, left=left, color=color, height=0.55, edgecolor=style.SURFACE, linewidth=2)
            ax.text(left + v / 2, y, f"{v:.0%}", ha="center", va="center", color="white", fontsize=10,
                    fontweight="bold")
            left += v
    ax.set_yticks([0, 1], ["Bookmaker (Bet365)", "Model"]); ax.set_xlim(0, 1); ax.set_xticks([])
    ax.legend(handles=[Patch(color=c, label=l_) for c, l_ in
                       [(style.BLUE, "OL win"), (style.NEUTRAL, "Draw"), (style.ORANGE, "OL loss")]],
              ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.02))
    ax.grid(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    ep = r[f"exp_pts_{key}"]; lo, hi = r[f"exp_pts_{key}_total_lo"], r[f"exp_pts_{key}_total_hi"]
    ax.set_title(f"{r['home']} – {r['away']} ({pd.Timestamp(r['date']):%d %b %Y}): expected points for OL "
                 f"{ep:.2f} [90 % interval {lo:.2f}–{hi:.2f}]", fontsize=10.5)
    fig.tight_layout(); out = FIGURES / "ol_next_match.png"
    fig.savefig(out); plt.close(fig); print(out)


def remaining_fixtures() -> None:
    d = pd.read_csv(TABLES / "ol_remaining_fixtures.csv").sort_values("ol_exp_pts").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 8))
    for i, r in d.iterrows():
        c = style.BLUE if r["venue"] == "home" else style.ORANGE
        ax.plot([r["ol_exp_pts_total_lo"], r["ol_exp_pts_total_hi"]], [i, i], color=c, lw=1.5, alpha=0.45)
        ax.plot([r["ol_exp_pts_model_lo"], r["ol_exp_pts_model_hi"]], [i, i], color=c, lw=3.5)
        ax.plot(r["ol_exp_pts"], i, "o", color=c, ms=7, mec=style.SURFACE, mew=1.5)
    ax.set_yticks(range(len(d)), [f"{o} ({v[0].upper()})" for o, v in zip(d["opponent"], d["venue"])], fontsize=8)
    ax.set_xlabel("Expected points for OL (thick: model uncertainty; thin: + team-rating uncertainty; 90 %)")
    ax.set_title("OL remaining Ligue 1 fixtures, expected points by opponent and venue")
    ax.legend(handles=[Line2D([], [], marker="o", color=style.BLUE, lw=2, label="Home (H)"),
                       Line2D([], [], marker="o", color=style.ORANGE, lw=2, label="Away (A)")], loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); out = FIGURES / "ol_remaining_fixtures.png"
    fig.savefig(out); plt.close(fig); print(out)


def sim_position() -> None:
    p = pd.read_csv(TABLES / "sim_F1_position_probs.csv", index_col=0).loc[OL]
    s = pd.read_csv(TABLES / "sim_F1_summary.csv", index_col=0).loc[OL]
    pos = p.index.astype(int); n = len(pos)
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.axvspan(0.5, 3.5, color="#e8f0fb", lw=0); ax.axvspan(3.5, 6.5, color="#f1f1ef", lw=0)
    ax.axvspan(n - 1.5, n + 0.5, color="#fbeae4", lw=0)
    ax.bar(pos, p.to_numpy(), color=style.BLUE, width=0.72)
    ax.set_ylim(0, p.max() * 1.32)
    for k, v in zip(pos, p.to_numpy()):
        if v >= 0.02:
            ax.text(k, v + 0.004, f"{v:.0%}", ha="center", fontsize=8.5, color=style.INK_2)
    ax.set_xticks(pos); ax.set_xlabel("Final league position"); ax.set_ylabel("Probability")
    ax.text(2, ax.get_ylim()[1] * 0.97, "Champions\nLeague*", ha="center", va="top", fontsize=8, color=style.INK_2)
    ax.text(5, ax.get_ylim()[1] * 0.97, "Other\nEurope*", ha="center", va="top", fontsize=8, color=style.INK_2)
    ax.text(n - 0.5, ax.get_ylim()[1] * 0.97, "Relegation", ha="center", va="top", fontsize=8, color=style.INK_2)
    ax.set_title(f"OL final position, 10,000 simulations: title {s['p_title']:.0%}, top 4 {s['p_top4']:.0%}, "
                 f"top 6 {s['p_top6']:.0%}", fontsize=11)
    ax.grid(axis="x", visible=False)
    fig.text(0.01, 0.01, "*Assumed UEFA slots (configs/leagues.json). Expected position "
             f"{s['exp_position']:.1f}.", color=style.INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); out = FIGURES / "sim_ol_position.png"
    fig.savefig(out); plt.close(fig); print(out)


def sim_points() -> None:
    d = pd.read_csv(TABLES / "sim_ol_points_distribution.csv")
    s = pd.read_csv(TABLES / "sim_F1_summary.csv", index_col=0).loc[OL]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(d["final_points"], d["probability"], color=style.BLUE, width=0.85)
    ax.axvspan(s["pts_p10"], s["pts_p90"], color=style.BLUE, alpha=0.10, lw=0)
    ax.axvline(s["exp_points"], color=style.INK, lw=1.5)
    ax.text(s["exp_points"] + 0.5, ax.get_ylim()[1] * 0.95, f"mean {s['exp_points']:.1f}", fontsize=9,
            color=style.INK_2, va="top")
    ax.set_xlabel("OL final points"); ax.set_ylabel("Probability")
    ax.set_title(f"OL final points: 80 % of simulations between {s['pts_p10']:.0f} and {s['pts_p90']:.0f} "
                 f"(14 points after 7 matches)", fontsize=11)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); out = FIGURES / "sim_ol_points.png"
    fig.savefig(out); plt.close(fig); print(out)


def sim_league_heatmap() -> None:
    p = pd.read_csv(TABLES / "sim_F1_position_probs.csv", index_col=0)
    order = (p * np.arange(1, p.shape[1] + 1)).sum(axis=1).sort_values().index
    p = p.loc[order]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    im = ax.imshow(p.to_numpy(), cmap="Blues", vmin=0, vmax=0.6, aspect="auto"); ax.grid(False)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.iloc[i, j]
            if v >= 0.08:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.3 else style.INK)
    ax.set_yticks(range(len(p)), p.index, fontsize=9); ax.set_xticks(range(p.shape[1]), range(1, p.shape[1] + 1), fontsize=8)
    ax.set_xlabel("Final league position")
    ax.set_title("Ligue 1 2026/27: probability of each final position (10,000 simulations)")
    for s_ in ax.spines.values():
        s_.set_visible(False)
    for t, lab in enumerate(p.index):
        if lab == OL:
            ax.add_patch(plt.Rectangle((-0.5, t - 0.5), p.shape[1], 1, fill=False, ec=style.ORANGE, lw=2))
    fig.tight_layout(); out = FIGURES / "sim_league_heatmap.png"
    fig.savefig(out); plt.close(fig); print(out)


def sim_backtest_calibration() -> None:
    ev = pd.read_parquet(TABLES / "backtest_eval_teams.parquet")
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.plot([0, 1], [0, 1], color=style.NEUTRAL, ls="--", lw=1.2)
    for name, col, color in [("Title", "title", style.BLUE), ("Top 4", "top4", style.ORANGE),
                             ("Relegation", "relegation", style.AQUA)]:
        r = reliability(ev[f"o_{col}"].astype(float).to_numpy(), ev[f"p_{col}"].to_numpy(), bins=6)
        ax.plot(r["forecast"], r["observed"], "-o", color=color, ms=6, mec=style.SURFACE, mew=1.5, label=name)
    ax.set_xlabel("Simulated probability"); ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.legend(loc="upper left")
    ax.set_title("Backtest 2023-25: simulated probabilities\nvs what happened (5 leagues)", fontsize=11)
    fig.tight_layout(); out = FIGURES / "sim_backtest_calibration.png"
    fig.savefig(out); plt.close(fig); print(out)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    style.apply()
    for fn in (elo_evolution, form_figure, strength_profile, next_match, remaining_fixtures, sim_position,
               sim_points, sim_league_heatmap, sim_backtest_calibration):
        fn()


if __name__ == "__main__":
    main()
