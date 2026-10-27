"""Build the research-paper style report (LaTeX -> PDF).   Run:  python -m olpred.report_tex

Needs pdflatex (texlive-latex-recommended, -extra, -fonts-recommended).  Numbers that depend on
the data come from reports/tables/ and site/data.js, so the text cannot drift from the results."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pandas as pd

REPORTS = Path("reports")
TAB = REPORTS / "tables"
OUT = REPORTS / "OL_ML_project.tex"
LEAGUES = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
REPO = "github.com/lacostetheo/ML-PREDICTION-OLYMPIQUE-LYONNAIS"


def esc(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("$", r"\$")]:
        s = s.replace(a, b)
    return s


def dmy(iso: str) -> str:
    """2026-10-21 -> 21 Oct 2026"""
    return pd.Timestamp(iso).strftime("%-d %b %Y")


def pct(x, d=0) -> str:
    return esc(f"{100 * x:.{d}f}%")


PENDING: list[str] = []


def cap(text) -> str:
    """Caption for the next table (it is then a floating, numbered table)."""
    PENDING.append(text)
    return ""


def tab(rows, spec, size=r"\small", bold_rows=()) -> str:
    """booktabs table; the first row is the header; cells are already LaTeX."""
    caption = PENDING.pop() if PENDING else None
    out = [r"\begin{table}[H]", rf"\caption{{{caption}}}"] if caption else []
    out += [r"\begin{center}", size, rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    for i, r in enumerate(rows):
        cells = [rf"\textbf{{{c}}}" if i in bold_rows else str(c) for c in r]
        if i == 0:
            cells = [rf"\textbf{{{c}}}" for c in r]
        out.append(" & ".join(cells) + r" \\")
        if i == 0:
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{center}"]
    if caption:
        out.append(r"\end{table}")
    return "\n".join(out)


def fig(name, width, caption, label=None) -> str:
    lab = rf"\label{{{label}}}" if label else ""
    return (rf"\begin{{figure}}[H]\centering\includegraphics[width={width}\textwidth]{{figures/{name}}}"
            rf"\caption{{{caption}}}{lab}\end{{figure}}")


# ------------------------------------------------------------------ data
t = Path("site/data.js").read_text()
D = json.loads(t[len("window.OLDATA = "):].rstrip().rstrip(";"))
final = pd.read_csv(TAB / "final_test_table.csv", index_col=0)
ucl = pd.read_csv(TAB / "europe_ucl_probabilities.csv")
euro = pd.read_csv(TAB / "europe_walk_forward.csv", index_col=0)
prev = D["prev"]
sim = D["leagues"]["F1"]["sim"]
ol = sim["summary"]["Lyon"]  # exp_pts,p10,p90,exp_pos,title,cl,top4,top6,europe,rel,pts_now,played
ol_pos = sim["pos_probs"]["Lyon"]
uel = pd.read_csv(TAB / "europe_uel_probabilities.csv")
euro_meta = {c: json.loads((TAB / f"europe_{c}_meta.json").read_text()) for c in ("ucl", "uel")}
eb = json.loads((TAB / "europe_bootstrap.json").read_text())
boot = pd.read_csv(TAB / "final_test_bootstrap.csv").set_index(["model", "vs"])
pool = pd.read_csv(TAB / "final_test_market_pooling.csv").set_index("model")
profile = pd.read_csv(TAB / "ol_strength_profile.csv").set_index("dimension")
matches = pd.read_parquet("data/processed/matches.parquet")
preds = pd.read_parquet(TAB / "final_test_predictions.parquet")
AS_OF, THROUGH = pd.Timestamp(D["as_of"]), pd.Timestamp(D["data_through"])
ROUNDS = D["leagues"]["F1"]["rounds"]["played"]
th = lambda n: f"{n:,}".replace(",", "{,}")                                   # 5,650 -> 5{,}650 (LaTeX)
fd = lambda ts: ts.strftime("%-d %B %Y")
WORDS = ["", "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh", "twelfth",
         "thirteenth", "fourteenth", "fifteenth", "sixteenth", "seventeenth", "eighteenth"]
NUM = {6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}
LR, BK, NV = "LogReg (final)", "Bookmaker Bet365 (reference)", "Class frequency (naive)"
ll = lambda k: final.loc[k, "log_loss"]

# ------------------------------------------------------------------ preamble
PRE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[a4paper,left=2.6cm,right=2.6cm,top=2.8cm,bottom=2.8cm]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{booktabs,graphicx,microtype,xcolor,fancyhdr,caption,enumitem,float}
\usepackage[hidelinks]{hyperref}
\captionsetup{font=small,labelfont=bf}
\captionsetup[table]{position=above,skip=4pt}
\setlist{itemsep=2pt,topsep=4pt}
\theoremstyle{definition}
\newtheorem{definition}{Definition}[section]
\newtheorem{result}{Result}[section]
\theoremstyle{remark}
\newtheorem{remark}{Remark}[section]
\pagestyle{fancy}\fancyhf{}
\fancyfoot[L]{\footnotesize Théo Lacoste $\cdot$ \texttt{""" + REPO + r"""}}
\fancyfoot[R]{\footnotesize\thepage}
\renewcommand{\headrulewidth}{0pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries}{\thesection}{1em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{1em}{}
\titlespacing*{\section}{0pt}{16pt}{6pt}
\titlespacing*{\subsection}{0pt}{10pt}{4pt}
\newcommand{\E}{\mathbb{E}}
\begin{document}
"""

# ------------------------------------------------------------------ title page
TITLE = r"""
\thispagestyle{fancy}
\begin{center}
{\Large\bfseries Machine Learning Project: Predicting the Victories\\[2pt] of Olympique Lyonnais}\\[10pt]
{\large\bfseries Walk-Forward Validation, Probabilistic Forecasts,\\[1pt] Monte Carlo Season Simulation, and a Test Against the Betting Market}\\[16pt]
{\large Théo Lacoste}\\[2pt]
{\small EDHEC Business School}\\[2pt]
{\small \texttt{""" + REPO + r"""}}\\[10pt]
{\normalsize """ + AS_OF.strftime("%B %-d, %Y") + r"""}
\end{center}
\vspace{2pt}
\begin{center}\begin{minipage}{0.88\textwidth}\small\itshape
\textbf{Disclaimer.} I hope the ML does not predict well, because what the f***, OL will be 5th this season.
(The simulation puts the average final position of OL at """ + f"{ol[3]:.1f}" + r""". So much for my hopes.)\ \ Joke aside, let us begin.
\end{minipage}\end{center}
\vspace{6pt}
\begin{center}\textbf{Abstract}\end{center}
\vspace{-4pt}
\begin{center}\begin{minipage}{0.88\textwidth}\small
This project was carried out in October 2026, two months into the 2026/27 season, and every ``current'' number refers to the state of @ASOF@ (@ROUNDSN@ Ligue 1 matches played).
We ask whether machine learning can forecast the results of Olympique Lyonnais (OL) better than an Elo rating or the betting market.
On @N_ALL@ matches of the five major European leagues (2010/11 to October 2026) we build features strictly from information available before kick-off, checked by an automated perturbation test,
and compare Elo, Poisson, logistic regression, random forest, gradient boosting, SVM and neural networks on a walk-forward test of @N_TEST@ matches.
(i)~The \emph{chosen model} is a multinomial logistic regression on 100 variables, refitted every season.
(ii)~Boosting and neural networks are statistically indistinguishable from it (log loss """ + f"{ll(LR):.3f}" + r"""); all beat Elo (""" + f"{ll('Elo (LogReg)'):.3f}" + r""") but none beats the bookmaker (""" + f"{ll(BK):.3f}" + r""").
(iii)~Pooling our probabilities with the market's adds nothing: the optimal weight on the model is zero.
(iv)~Elo and squad market value explain almost everything that is predictable.
A Monte Carlo simulator, backtested on 2023--2025, projects OL at """ + f"{ol[0]:.1f}" + r""" points (80\% interval """ + f"{ol[1]:.0f}--{ol[2]:.0f}" + r"""), an average final position of """ + f"{ol[3]:.1f}" + r""" and a """ + pct(ol[6]) + r""" chance of the top four.
The Champions League and the Europa League, which OL plays this season, have no clear favourite: the best-placed club wins only """ + pct(ucl["Winner"].max()) + r""" and """ + pct(uel["Winner"].max()) + r""" of the simulations respectively.
\end{minipage}\end{center}
"""

BODY = []
add = BODY.append

# ------------------------------------------------------------------ 1 Introduction
add(r"\section{Introduction}")
add(r"""\textbf{When this was done.} This project was carried out in \textbf{October 2026}, two months into the 2026/27 season. @ROUNDSW@ rounds of Ligue 1 had been played; the data end with the matches of @THROUGH@,
and the snapshot used for all forecasts is dated @ASOF@. Anything described as ``current'', ``next'' or ``projected'' refers to that date and will change as the season goes on.""")
add(r"""Football is the sport where the gap between a simple forecast and the best one is smallest: a match is decided by a handful of goals,
the stronger team wins only about one time in two, and one game in four ends in a draw. This makes it a good laboratory for a question that also matters in finance:
\emph{how much can a model extract from public data, how reliably can that be measured, and does it survive comparison with a liquid market?}""")
add(r"""OL plays about 35 league matches a year, far too few to train a model. I therefore train on all five major leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1)
and use OL as a case study. The object to forecast is the following.""")
add(r"""\begin{definition}[Match forecast]
A \emph{match forecast} is a probability vector $p=(p_A,p_D,p_H)$ over the outcomes away win, draw and home win, computed from information available before kick-off.
The \emph{expected points} of the home side are $3p_H+p_D$.
\end{definition}""")
nxt = next(m for m in D["fixtures"]["F1"] if "Lyon" in (m["h"], m["a"]))
nh, na = esc(D["display"].get(nxt["h"], nxt["h"])), esc(D["display"].get(nxt["a"], nxt["a"]))
nxt_date = pd.Timestamp(nxt["d"]).strftime("%A %-d %B %Y")
p_away, p_draw, p_home = nxt["p"]
ol_win, ol_loss = (p_home, p_away) if nxt["h"] == "Lyon" else (p_away, p_home)
add(rf"""For instance, for OL's next Ligue 1 match, {nh}--{na} on {nxt_date}, the model gives the forecast of Table~\ref{{tab:next}}.""")
cap(rf"Forecast for {nh}--{na} (Ligue 1, {nxt_date}).\label{{tab:next}}")
add(tab([["", "OL win", "Draw", "OL loss", "OL expected points"],
         ["Model", pct(ol_win), pct(p_draw), pct(ol_loss), f"{3 * ol_win + p_draw:.2f}"]], "lcccc"))
add(rf"""\textbf{{The model I chose.}} A \emph{{multinomial logistic regression}} on 100 input variables (Elo rating, squad market value, recent form, shots, schedule),
refitted once a year (Section~\ref{{sec:model}}). I also tested random forest, gradient boosting, a support-vector machine and neural networks, all with scikit-learn.
None is significantly better, so I kept the simplest and most explainable one.""")
add(r"""\textbf{Three rules.} (1)~\emph{No leakage}: a variable used to predict a match played at date $T$ contains nothing dated $T$ or later.
(2)~\emph{Temporal validation}: train on the past, judge on the future, never on a random split.
(3)~\emph{Honest evidence}: every claimed improvement comes with a confidence interval, and negative results are reported as negative.""")

# ------------------------------------------------------------------ 2 Data
add(r"\section{Data}")
cap(r"Data sources.")
add(tab([["Source", "Content", "Use"],
         ["football-data.co.uk", "Results, shots, corners, cards, bookmaker odds; 5 leagues, 2010/11--2026", "Core data"],
         ["Transfermarkt dataset (CC0)", "656{,}000 player market valuations", "Squad value"],
         ["Self-computed", "Elo ratings, shot-based ``xG proxy''", "Variables"],
         ["Wikipedia", "Champions League and Europa League 2026/27 fixtures and scores", "Section~\\ref{sec:ucl}"],
         ["openfootball, Maxifoot", "Calendars of the matches still to play (dates only)", "Display"]], "p{3.6cm}p{8.2cm}l"))
add(r"""The sample has @N_ALL@ matches (@N_OL@ for OL) and @N_CLUBS@ clubs: training 2011--2020 (about 18{,}200 matches), validation 2021--22 (3{,}651) and test October 2023 to October 2026 (@N_TEST@).
\textbf{Rejected sources.} FBref advanced statistics were removed by their provider in January 2026; Understat forbids automated access in its \texttt{robots.txt}; ClubElo was unreachable.
No free historical source exists for injuries or line-ups, so they could not be tested.
\textbf{Terms of use.} football-data.co.uk allows private, non-commercial use only. This is a private academic study: raw data are downloaded, kept out of the repository and not redistributed.""")

# ------------------------------------------------------------------ 3 Features
add(r"\section{Variables and leakage control}")
add(r"""Every variable is computed for the home team, the away team and their difference, from matches strictly before the one predicted.""")
add(r"""\begin{definition}[Leakage-free variable]
A variable attached to a match played at date $T$ is \emph{leakage-free} if it is a function of results dated strictly before $T$ only.
\end{definition}""")
cap(r"Families of variables.")
add(tab([["Family", "Construction"],
         ["Elo rating", "Strength score updated after every match (below), one scale per league"],
         ["Recent form", "Mean over the last 3, 5, 10 matches of points, goals for and against, shots on target"],
         ["Venue form", "Last 5 home (or away) matches"],
         ["League table", "Points and goal difference per match, position before the match"],
         ["Schedule", "Rest days, matches in the previous 14 days"],
         ["Squad value", "$\\ln V$ of the team minus the league average of that month (Transfermarkt)"]], "p{3.0cm}p{9.5cm}"))
add(r"""\textbf{Elo} \cite{elo1978}\textbf{.} Before a match between $i$ (home) and $j$ (away), the expected score of the home team is
\begin{equation}E_{ij}=\frac{1}{1+10^{-(R_i-R_j+65)/400}},\end{equation}
where 65 points is the home advantage. With $S=1$, $\tfrac12$, $0$ for a win, draw, loss and $g$ the goal difference, the ratings are updated after the match by
\begin{equation}R_i' = R_i + K\,\bigl(1+\ln(1+|g|)\bigr)\,(S-E_{ij}),\qquad R_j' = R_j - (R_i'-R_i),\qquad K=20.\end{equation}
The rating stored for the match is the one \emph{before} this update.""")
add(r"""\textbf{Squad value} is measured relative to the league so that market inflation is not mistaken for strength:
\begin{equation}sv_i=\ln V_i-\frac{1}{|L|}\sum_{k\in L}\ln V_k.\end{equation}""")
add(r"""\textbf{How leakage is prevented.} Rolling variables are shifted by one match; Elo is stored before its update; valuations used for the squad value are dated before the monthly snapshot;
bookmaker odds are never a variable. An automated test replaces every result after a cutoff date by random values, reruns the whole pipeline and requires all earlier variables to be identical.
It fails, as it should, on two bugs I injected on purpose (a rolling window without the shift, Elo stored after its update).""")
add(r"""\textbf{Two lessons from the data.} The ``xG proxy'' (a regression of goals on shots) collapses to $0.287\times$ shots on target, so it is not a real xG.
Variables counting how many matches a team has been observed for are just a calendar clock; I removed them (cost $+0.0011$ log loss on validation, not significant).""")

# ------------------------------------------------------------------ 4 Model
add(r"\section{The model I chose: multinomial logistic regression}\label{sec:model}")
add(r"""\begin{definition}[Multinomial logistic regression]
Let $x\in\mathbb{R}^{100}$ be the (standardised) variables of a match. For each outcome $k\in\{A,D,H\}$ the model computes a linear score $z_k=w_k^\top x+b_k$ and returns
\begin{equation}P(\text{outcome}=k\mid x)=\frac{\exp(w_k^\top x+b_k)}{\sum_{j\in\{A,D,H\}}\exp(w_j^\top x+b_j)}.\end{equation}
\end{definition}""")
add(r"""\textbf{Estimation.} The weights maximise the log-likelihood of the training matches, with an $L^2$ penalty that keeps them small:
\begin{equation}\min_{W,b}\; -\frac1N\sum_{i=1}^{N}\ln P(y_i\mid x_i)\;+\;\frac{1}{2CN}\,\|W\|^2 .\end{equation}
The strength $1/C$ of the penalty is chosen by \emph{time-ordered} cross-validation inside the training seasons (each fold trains on the past and scores the next block).
Variables are standardised and missing values replaced by the median, both fitted on training data only. The model is refitted once a year.""")
add(r"""\begin{remark}[Why this model]
(a) It is as accurate as the alternatives (Section~\ref{sec:results}). (b) It is explainable: the linear scores split exactly into the contribution of each family of variables,
$z_k=b_k+\sum_f c_{k,f}$, which the website displays. (c) Its probabilities are well calibrated. (d) It is fast enough to simulate 10{,}000 seasons per league.
\end{remark}""")
cap(r"Models compared.")
add(tab([["Family", "Models compared"],
         ["Baselines", "Class frequencies; league table; Elo calibrated to 3 classes; independent Poisson goals \\cite{dixon1997}"],
         ["Linear (chosen)", "Multinomial logistic regression with $L^2$ penalty"],
         ["Trees", "Random forest; gradient boosting"],
         ["Kernel", "RBF support-vector machine (under-tuned)"],
         ["Neural", "MLP with 1 and 2--3 hidden layers ($L^2$, few epochs, mean of 5 seeds)"],
         ["Online", "Logistic regression whose weights are updated by gradient descent after each week of results"],
         ["Reference", "Bet365 prices turned into probabilities (margin removed, average margin 5.5\\%)"]], "p{3.0cm}p{9.5cm}"))
add(r"""Only scikit-learn \cite{pedregosa2011} and numpy are used. Dropout and Batch Normalisation do not exist in scikit-learn; I replaced them by an $L^2$ penalty, few epochs and seed averaging.
LSTM and GRU networks were not built: the rolling variables already summarise each team's history.""")

# ------------------------------------------------------------------ 5 Metrics
add(r"\section{How quality is measured}")
add(r"""Accuracy is a weak summary here: a draw is almost never the most likely outcome, so a useful model can rarely name one. The main metric is the log loss, a strictly proper scoring rule \cite{gneiting2007}.""")
add(r"""\begin{definition}[Log loss]
For $N$ matches, with $p_i$ the probability given to the outcome that actually occurred in match $i$,
\begin{equation}\mathrm{LL}=-\frac1N\sum_{i=1}^{N}\ln p_i .\end{equation}
\end{definition}""")
add(r"""\emph{Example.} If the model gave 20\% to what happened, the match costs $\ln 5=1.61$; if it gave 60\%, it costs $0.51$.
A model that always says $(30.0\%,25.2\%,44.7\%)$ (the average frequencies) scores $1.075$; a model that knew the result would score $0$.""")
add(r"""I also report the Brier score \cite{brier1950} and the expected calibration error:
\begin{equation}\mathrm{Brier}=\frac1N\sum_{i=1}^{N}\sum_{k}(p_{ik}-o_{ik})^2,\qquad \mathrm{ECE}=\sum_{b}\frac{n_b}{N}\,\bigl|\bar p_b-\bar o_b\bigr|,\end{equation}
where $o_{ik}=1$ if outcome $k$ occurred in match $i$ and $0$ otherwise, and $b$ indexes bins of forecast probabilities.""")
add(r"""\textbf{Significance.} To compare models $A$ and $B$, I take the paired difference $\Delta=\overline{\ell_A-\ell_B}$ of their per-match log losses and build a 95\% confidence interval by a \emph{block bootstrap}
that resamples whole calendar weeks (matches of the same week are not independent), with 4{,}000 draws. With 5{,}600 test matches, a difference below about 0.003 cannot be detected.""")

# ------------------------------------------------------------------ 6 Validation
add(r"\section{Validation design}")
add(r"""Football results form a time series: a random split would let the model learn from matches played after those it is tested on.
Training covers 2011--2020, validation 2021--22 (used to compare models during development) and test October 2023 to October 2026 (used once, with frozen hyper-parameters).
The final evaluation is \emph{walk-forward}: each test season $S$ is predicted by models fitted on seasons $2011,\dots,S-1$, as it would run in real life.""")

# ------------------------------------------------------------------ 7 Results
add(r"\section{Results}\label{sec:results}")
rows = [["Model", "Accuracy", "Log loss [95\\% CI]", "Brier", "ECE"]]
for name, label in [(NV, "Baseline (frequencies)"), ("Elo (LogReg)", "Elo"), ("Poisson goals", "Poisson goals"),
                    (LR, "Logistic regression (chosen)"), ("Gradient boosting", "Gradient boosting"),
                    ("MLP deep (5 seeds)", "Neural network (MLP)"), ("Ensemble LogReg+GB+MLP", "Ensemble of the three"),
                    (BK, "Bookmaker (Bet365)")]:
    r = final.loc[name]
    rows.append([label, pct(r["accuracy"], 1), f"{r['log_loss']:.4f} [{r['log_loss_ci_low']:.3f}, {r['log_loss_ci_high']:.3f}]", f"{r['brier']:.4f}", f"{r['ece']:.3f}"])
cap(r"Walk-forward test, October 2023 to October 2026 (@N_TEST@ matches).")
add(tab(rows, "lcccc", bold_rows=(4,)))
add(fig("final_logloss.png", 0.8, "Log loss on the test seasons with 95\\% confidence intervals: three model families are tied, the bookmaker is ahead.", "fig:ll"))
d_elo, d_gb, d_mlp, d_bk = boot.loc[(LR, "Elo (LogReg)")], boot.loc[("Gradient boosting", LR)], boot.loc[("MLP deep (5 seeds)", LR)], boot.loc[(LR, BK)]
add(rf"""\begin{{result}}
The logistic regression beats Elo ($\Delta={d_elo['diff']:+.4f}$, 95\% CI $[{d_elo['ci_low']:+.4f},{d_elo['ci_high']:+.4f}]$). Gradient boosting (${d_gb['diff']:+.4f}$) and the neural network (${d_mlp['diff']:+.4f}$) do \emph{{not}} beat it: their intervals contain 0.
The bookmaker beats every model ($\Delta={d_bk['diff']:+.4f}$ for the logistic regression, CI $[{d_bk['ci_low']:+.4f},{d_bk['ci_high']:+.4f}]$). Complexity did not pay.
\end{{result}}""")
add(r"""\subsection{Calibration}""")
add(fig("final_reliability.png", 0.95, "Reliability diagrams on the test seasons: the closer to the diagonal, the more honest the forecast.", "fig:rel"))
ece_v = final.loc[[LR, "Gradient boosting", "MLP deep (5 seeds)", "Ensemble LogReg+GB+MLP", BK], "ece"]
n_draw = int((preds[[f"{LR}|away", f"{LR}|draw", f"{LR}|home"]].to_numpy().argmax(1) == 1).sum())
add(rf"""The probabilities are well calibrated (ECE between {100 * ece_v.min():.1f}\% and {100 * ece_v.max():.1f}\%). The model almost never names a draw as the most likely outcome ({n_draw} matches out of @N_TEST@): draws are essentially unpredictable.""")
add(r"""\subsection{Ablation: which data add value?}""")
cap(r"Ablation: nested blocks of variables (validation seasons).")
add(tab([["Logistic regression, validation", "Log loss", "Change vs previous [95\\% CI]"],
         ["A: Elo only", "0.9957", "--"],
         ["B: + results form", "0.9931", "$-0.0026$ $[-0.0051,-0.0001]$"],
         ["C: + shots and xG proxy", "0.9887", "$-0.0044$ $[-0.0072,-0.0017]$"],
         ["D: + league table, fatigue", "0.9890", "$+0.0003$ $[-0.0013,+0.0019]$ (nothing)"],
         ["E: + squad value (final)", "0.9870", "$-0.0020$ $[-0.0048,+0.0009]$"]], "lcc"))
add(r"""Form and shots each add a small but detectable gain on top of Elo; league position and fatigue add nothing. Squad value is not significant on two validation seasons although it ranks second in the importance analysis below:
it overlaps with Elo, both measuring long-run strength.""")

# ------------------------------------------------------------------ 8 Interpretation
add(r"\section{What drives the predictions?}")
add(fig("phase7_family_importance.png", 0.78, "Increase of the validation log loss when a family of variables is shuffled (permutation importance).", "fig:imp"))
add(r"""Elo ($+0.050$) and squad value ($+0.024$) dominate. Recent form adds $+0.007$ and shots $+0.003$; rest days, league table and venue form are close to zero.
Home sides win 44.7\% of matches, draw 25.2\% and lose 30.0\%, which sits in the intercept.""")
add(r"""\begin{remark}
In the linear model, form has a slightly negative effect at fixed Elo and squad value. This is a collinearity artefact (overlapping windows), not evidence that a good run hurts.
Importance describes what the \emph{model} uses, not what causes results.
\end{remark}""")

# ------------------------------------------------------------------ 9 Market
add(r"\section{Model versus bookmaker}")
add(r"""This is an academic question only; no betting strategy is built. To test whether the model contains information the market lacks, I pool the two by log-linear pooling,
\begin{equation}p\;\propto\;p_{\text{market}}^{\,1-w}\;p_{\text{model}}^{\,w},\end{equation}
and choose the weight $w$ on validation. On the test seasons the best weight on the model is $w=@WBEST@$: the market already contains everything the model knows.
The pooled log loss (@POOLED@) is not better than the market alone (@MARKET@). The market takes a margin of 5.5\% on average and is still sharper than the model; the likely reason is information I do not have (line-ups, injuries, news).""")

# ------------------------------------------------------------------ 10 Previous results
add(r"\section{Previous results: was the model right?}")
add(r"""\begin{remark}
These forecasts are recomputed \emph{after the fact} with the walk-forward method (models fitted only on earlier seasons, variables built only from earlier matches).
They were \textbf{not published in advance}. ``Right'' means that the most likely outcome occurred: a 45\% pick is wrong more often than right, by design, so log loss and calibration are the fair scores.
\end{remark}""")
rows = [["Period", "Matches", "Correct", "LL model", "LL bookmaker", "LL no-skill"]]
last_d = pd.Timestamp(prev["last"])
when = {"Last 7 days": f"{(last_d - pd.Timedelta(days=6)).strftime('%-d %b')}--{last_d.strftime('%-d %b %Y')}", "Last 30 days": f"{(last_d - pd.Timedelta(days=29)).strftime('%-d %b')}--{last_d.strftime('%-d %b %Y')}"}
for k, v in prev["summary"].items():
    rows.append([esc(k + (f" ({when[k]})" if k in when else "")), v["n"], pct(v["acc"]), f"{v['ll']:.3f}", f"{v['ll_book']:.3f}", f"{v['ll_naive']:.3f}"])
cap(r"Accuracy and log loss (LL) over recent periods.")
add(tab(rows, "lccccc"))
rows = [["Top pick given", "Matches", "Mean forecast", "Came true"]]
for b in prev["calibration"]:
    rows.append([f"{int(b['lo'] * 100)}--{int(b['hi'] * 100)}\\%", b["n"], pct(b["said"]), pct(b["came_true"])])
cap(r"When the model said $x$\%, how often did it happen? (all @N_TEST@ test matches)")
add(tab(rows, "cccc"))
W30 = prev["summary"]["Last 30 days"]
sharp = "slightly less sharp than" if W30["ll"] > W30["ll_book"] else "slightly sharper than"
add(rf"""When the top pick was given 60--70\%, it happened {pct(prev["calibration"][3]["came_true"])} of the time. Over the last 30 days of the data ({(last_d - pd.Timedelta(days=29)).strftime("%-d %B")} to {fd(last_d)}) the model is right {pct(W30["acc"])} of the time and is {sharp} the bookmaker.""")
names = ["Away win", "Draw", "Home win"]
rows = [["Date", "Match", "Score", "H / D / A", "Top pick", "Result"]]
for d, lg, h, a, hg, ag, y, p in prev["rows"][:14]:
    best = max(range(3), key=lambda i: p[i])
    rows.append([pd.Timestamp(d).strftime("%-d %b"), esc(f"{h} -- {a}").replace("--", "--"), f"{hg}--{ag}", f"{100 * p[2]:.0f} / {100 * p[1]:.0f} / {100 * p[0]:.0f}",
                 f"{names[best]} {100 * p[best]:.0f}\\%", "Right" if best == y else "Wrong"])
cap(r"The 14 most recent matches before the snapshot, 2026 season (probabilities in \%).")
add(tab(rows, "llcccl", size=r"\footnotesize"))

# ------------------------------------------------------------------ 11 OL
add(r"\section{Case study: Olympique Lyonnais}")
add(fig("ol_strength_profile.png", 0.8, "OL's profile in Ligue 1 on @ASOF@ (percentile among clubs; right: rank).", "fig:prof"))
elo_rank = int(profile.loc["Elo rating", "rank"])
leader = max(D["leagues"]["F1"]["info"].items(), key=lambda kv: kv[1]["elo"])
rk = lambda dim: WORDS[int(profile.loc[dim, "rank"])]
v_nxt = D["pairs"]["F1"][f"{nxt['h']}|{nxt['a']}"]
lo_hi = (v_nxt[7], v_nxt[8]) if nxt["h"] == "Lyon" else (v_nxt[3], v_nxt[4])
add(rf"""OL is {WORDS[elo_rank]} in Ligue 1 by Elo ({th(round(profile.loc["Elo rating", "value"]))}; {esc(D["display"].get(leader[0], leader[0]))} {th(round(leader[1]["elo"]))}), {rk("Recent form: points/match, last 10")} for recent points and {rk("Recent attack: goals scored, last 10")} for goals scored,
and {rk("Squad value: total")} by squad value (EUR {profile.loc["Squad value: total", "value"]:.0f} m; the defence is {rk("Squad value: defenders")}). Its shot volume (shots on target) ranks {rk("Recent attack: shots on target, last 10")}.
For {nh}--{na}, the 90\% interval on OL's win probability is {100 * lo_hi[0]:.0f}--{100 * lo_hi[1]:.0f}\% once model uncertainty (100 bootstrap refits) and uncertainty on the true strength of both teams (Elo shaken by 60 points) are added.""")

# ------------------------------------------------------------------ 12 Simulation
add(r"\section{Season simulation}")
add(r"""\begin{definition}[Simulated season]
Given the current state, a \emph{simulated season} (i)~shakes the Elo rating of every team by $\varepsilon\sim\mathcal{N}(0,60^2)$, (ii)~draws the outcome of each remaining match from the model's probabilities,
(iii)~draws a scoreline from a Poisson goal model consistent with that outcome (for goal-difference tie-breaks) and (iv)~ranks the table on points, goal difference and goals scored.
The expected points of a team are its current points plus $\sum_{m}(3p^{(m)}_{\mathrm{win}}+p^{(m)}_{\mathrm{draw}})$ over its remaining matches.
\end{definition}""")
add(r"""The shock size, 60 Elo points, was chosen by a \emph{backtest}: each past season was simulated from 31 October using only what was known then, and compared with the real final table.""")
cap(r"Backtest on 2023--2025 (15 league-seasons, 288 team-seasons).")
add(tab([["", "Simulator", "Naive reference"],
         ["Mean absolute error of final points", "6.4", "8.9"],
         ["Brier score, title", "0.022", "0.049"],
         ["Brier score, top 4", "0.064", "0.165"],
         ["Brier score, relegation", "0.055", "0.117"],
         ["Coverage of the 80\\% interval", "72\\%", "80\\%"]], "lcc"))
add(r"""The simulator beats the naive reference, but its ``80\%'' intervals cover only 72\%: they should be read as lower bounds.""")
rows = [["League", "1st", "2nd", "3rd"]]
for L, name in LEAGUES.items():
    s = D["leagues"][L]["sim"]["summary"]
    top = sorted(s.items(), key=lambda kv: kv[1][3])[:3]
    rows.append([name] + [esc(f"{tm} ({v[0]:.0f} pts, title {100 * v[4]:.0f}%)") for tm, v in top])
cap(r"Projected top three of each league (by expected final position, 10{,}000 simulated seasons).")
add(tab(rows, "llll", size=r"\footnotesize"))
rows = [["Team", "Exp. points [80\\%]", "Title", "Champions Lg.", "Top 4", "Relegation"]]
for tm in ["Paris SG", "Monaco", "Lille", "Lyon"]:
    v = sim["summary"][tm]
    rows.append([esc(tm), f"{v[0]:.1f} [{v[1]:.0f}--{v[2]:.0f}]", pct(v[4], 1), pct(v[5], 1), pct(v[6], 1), pct(v[9], 1)])
cap(r"Ligue 1 2026/27 (state of @ASOF@, after @ROUNDSN@ rounds).")
add(tab(rows, "lccccc", bold_rows=(4,)))
add(fig("sim_ol_position.png", 0.8, "Final position of OL over 10{,}000 simulated seasons.", "fig:pos"))
top2 = sorted(range(len(ol_pos)), key=lambda i: -ol_pos[i])[:2]
pts_sorted = sorted((v[10] for v in sim["summary"].values()), reverse=True)
add(rf"""The average final position of OL is \textbf{{{ol[3]:.1f}}}; the most likely finishes are {WORDS[top2[0] + 1]} and {WORDS[top2[1] + 1]} ({pct(ol_pos[top2[0]])} and {pct(ol_pos[top2[1]])}), but the distribution is wide:
after {ROUNDS} matches the clubs from 2nd to 7th place are within {int(pts_sorted[1] - pts_sorted[6])} points of each other, so the forecast rests on team strength rather than on the table.""")

# ------------------------------------------------------------------ 13 Champions League
add(r"\section{Champions League and Europa League}\label{sec:ucl}")
eu, el = euro_meta["ucl"], euro_meta["uel"]
add(rf"""The aim is to use the real fixtures and to combine each club's performance in its own championship; OL plays the Europa League this season. The calendars (144 league-phase matches and 36 clubs in each competition) come from Wikipedia;
at the snapshot date {eu['played']} Champions League matches and {el['played']} Europa League matches had been recorded there, and the others are simulated.
Domestic leagues are only comparable through matches between their clubs, so I fitted a logistic regression on 3{{,}}085 past Champions League, Europa League and Conference League matches (2012--2025):""")
add(r"""\begin{equation}d^{\text{value}}=\ln V_{\text{home}}-\ln V_{\text{away}},\qquad d^{\text{ppg}}=\mathrm{ppg}_{\text{home}}-\mathrm{ppg}_{\text{away}},\end{equation}
where $\mathrm{ppg}$ is the number of domestic league points per match over the last 20 league matches.""")
rows = [["Walk-forward 2023--2025 (943 matches)", "Log loss"]]
for k in ["Class frequency", "Squad value only", "Domestic form only", "Value + domestic form"]:
    rows.append([k, f"{euro.loc[k, 'log_loss']:.4f}"])
cap(r"European strength model.")
add(tab(rows, "lc"))
e_val, e_form = eb["Squad value only vs Class frequency"], eb["Value + domestic form vs Squad value only"]
add(rf"""\begin{{result}}
Squad value carries the signal (${e_val['diff']:+.3f}$ vs the class frequencies, significant). Domestic form adds nothing measurable once value is known (${e_form['diff']:+.4f}$, 95\% CI $[{e_form['ci_low']:+.4f},{e_form['ci_high']:+.4f}]$).
I keep it because the question asked to combine both, but its weight is small.
\end{{result}}""")
add(r"""I then simulated 20{,}000 times the rest of the league phase, the play-offs (ranks 9--24), the round of 16, quarter-finals, semi-finals and the final (two legs; extra time and penalties close to a coin flip; the final on neutral ground).""")
rows = [["Club", "Top 8", "Semi-final", "Final", "Winner"]]
for r in ucl.head(10).itertuples():
    rows.append([esc(r.team), pct(r.top8), pct(r.SF), pct(r.Final), pct(r.Winner, 1)])
cap(r"Champions League 2026/27: probabilities from 20{,}000 simulations.")
add(tab(rows, "lcccc"))
rows = [["Club", "Top 8", "Semi-final", "Final", "Winner"]]
shown = list(uel.head(10).itertuples())
if not any(r.team == "Lyon" for r in shown):
    shown.append(next(r for r in uel.itertuples() if r.team == "Lyon"))
for r in shown:
    rows.append([esc(r.team), pct(r.top8), pct(r.SF), pct(r.Final), pct(r.Winner, 1)])
cap(r"Europa League 2026/27: probabilities from 20{,}000 simulations (OL included).")
add(tab(rows, "lcccc", bold_rows=tuple(i + 1 for i, r in enumerate(shown) if r.team == "Lyon")))
close = lambda df: int((df["Winner"] >= df["Winner"].max() - 0.01).sum())
names = lambda df, k: ", ".join(esc(x) for x in df["team"].head(k))
lyon_uel = next(r for r in uel.itertuples() if r.team == "Lyon")
add(rf"""There is \textbf{{no clear favourite}} in either competition. In the Champions League {close(ucl)} clubs are within one point of the leader ({names(ucl, close(ucl))}), the best at {pct(ucl["Winner"].max(), 1)};
in the Europa League the leader, {esc(uel.iloc[0].team)}, has {pct(uel.iloc[0].Winner, 1)}, followed by {esc(uel.iloc[1].team)} ({pct(uel.iloc[1].Winner, 1)}) and {esc(uel.iloc[2].team)} ({pct(uel.iloc[2].Winner, 1)}), and OL has {pct(lyon_uel.Winner, 1)}.
These models are validated match by match, not on whole tournaments; the knockout bracket is approximate; {eu['n_assumed']} Champions League clubs and {el['n_assumed']} Europa League clubs are not in the squad-value data and are assumed weak.""")

# ------------------------------------------------------------------ 14 Limits, 15 Conclusion
add(r"\section{Limitations}")
add(r"""\begin{itemize}
\item No injuries, suspensions, line-ups or coaching news (the likely reason the bookmaker is better); no real xG.
\item Squad values are a crowd-sourced snapshot (June 2026); 13 clubs have no value.
\item The bookmaker benchmark uses Bet365 pre-match prices with a simple margin removal.
\item Few validation windows; the SVM is under-tuned; online updating was tested for the linear model only.
\item Simulation: intervals are too narrow in the backtest, UEFA places are assumptions, and strength is constant apart from a random shock.
\item Wikipedia lags: @UCLMISS@ Champions League matches played before the snapshot are not recorded there yet, and are simulated.
\end{itemize}""")
add(r"\section{Conclusion}")
add(rf"""On public data, football results can be forecast with \emph{{well-calibrated probabilities that are only modestly sharper than team strength alone}}. Elo and squad value explain most of what is predictable.
The chosen model, a regularised multinomial logistic regression, is as good as boosting and neural networks: complexity did not pay. The models cannot beat the betting market and add no information to it,
which is consistent with an efficient market. For OL, the model sees the {WORDS[int(profile.loc["Squad value: total", "rank"])]} most valuable squad of Ligue 1 and a middling start: about {ol[0]:.1f} points, an average finish of {ol[3]:.1f} and a {pct(ol[5])} chance of a Champions League place.
So the disclaimer at the beginning stands: I really hope the ML is wrong.""")
add(r"""\textbf{Reproducibility.} Code, tests (including the leakage test) and the interactive website are in the repository; \texttt{make weekly} refreshes the data, the variables, the OL predictions, the simulations and the website. All numbers come from \texttt{reports/tables/}.""")

REFS = r"""
\begin{thebibliography}{9}\small
\bibitem{brier1950} G.~W. Brier. Verification of forecasts expressed in terms of probability. \emph{Monthly Weather Review}, 78(1):1--3, 1950.
\bibitem{gneiting2007} T.~Gneiting and A.~E. Raftery. Strictly proper scoring rules, prediction, and estimation. \emph{Journal of the American Statistical Association}, 102(477):359--378, 2007.
\bibitem{pedregosa2011} F.~Pedregosa et al. Scikit-learn: machine learning in Python. \emph{Journal of Machine Learning Research}, 12:2825--2830, 2011.
\bibitem{elo1978} A.~E. Elo. \emph{The Rating of Chessplayers, Past and Present}. Arco, 1978.
\bibitem{dixon1997} M.~J. Dixon and S.~G. Coles. Modelling association football scores and inefficiencies in the football betting market. \emph{Journal of the Royal Statistical Society C}, 46(2):265--280, 1997.
\end{thebibliography}
"""


def main() -> None:
    tex = PRE + TITLE + "\n\n".join(BODY) + REFS + "\n\\end{document}\n"
    tokens = {"@N_ALL@": th(len(matches)), "@N_OL@": str(int(((matches["home"] == "Lyon") | (matches["away"] == "Lyon")).sum())),
              "@N_CLUBS@": str(len(set(matches["home"]) | set(matches["away"]))), "@N_TEST@": th(prev["n_total"]),
              "@ASOF@": fd(AS_OF), "@THROUGH@": fd(THROUGH), "@ROUNDSN@": str(ROUNDS), "@ROUNDSW@": NUM.get(ROUNDS, str(ROUNDS)),
              "@WBEST@": f"{pool.loc[LR, 'weight_best_in_sample_on_test']:.2f}", "@POOLED@": f"{pool.loc[LR, 'pooled_log_loss']:.4f}",
              "@MARKET@": f"{pool.loc[LR, 'market_log_loss']:.4f}", "@UCLMISS@": str(D["europe"]["ucl"]["missing"])}
    for k, v in tokens.items():
        tex = tex.replace(k, v)
    assert "@" not in re.sub(r"\\texttt\{[^}]*\}", "", tex).replace("\\bibitem", ""), "unreplaced token"
    tex = tex.replace(r"\section{Introduction}", r"\section{Introduction}", 1)
    OUT.write_text(tex, encoding="utf8")
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", OUT.name], cwd=REPORTS, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-3000:])
            raise SystemExit("pdflatex failed")
    for ext in ("aux", "log", "out"):
        (REPORTS / f"OL_ML_project.{ext}").unlink(missing_ok=True)
    print(OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
