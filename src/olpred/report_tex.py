"""Build the research-paper style report (LaTeX -> PDF).   Run:  python -m olpred.report_tex

Needs pdflatex (texlive-latex-recommended, -extra, -fonts-recommended).  Numbers that depend on
the data come from reports/tables/ and site/data.js, so the text cannot drift from the results."""
from __future__ import annotations

import json
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
euro_meta = json.loads((TAB / "europe_ucl_meta.json").read_text())
prev = D["prev"]
sim = D["leagues"]["F1"]["sim"]
ol = sim["summary"]["Lyon"]  # exp_pts,p10,p90,exp_pos,title,cl,top4,top6,europe,rel,pts_now,played
ol_pos = sim["pos_probs"]["Lyon"]
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
{\normalsize October 27, 2026}
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
We ask whether machine learning can forecast the results of Olympique Lyonnais (OL) better than an Elo rating or the betting market.
On 29{,}239 matches of the five major European leagues (2010/11 to October 2026) we build features strictly from information available before kick-off, checked by an automated perturbation test,
and compare Elo, Poisson, logistic regression, random forest, gradient boosting, SVM and neural networks on a walk-forward test of 5{,}602 matches.
(i)~The \emph{chosen model} is a multinomial logistic regression on 100 variables, refitted every season.
(ii)~Boosting and neural networks are statistically indistinguishable from it (log loss """ + f"{ll(LR):.3f}" + r"""); all beat Elo (""" + f"{ll('Elo (LogReg)'):.3f}" + r""") but none beats the bookmaker (""" + f"{ll(BK):.3f}" + r""").
(iii)~Pooling our probabilities with the market's adds nothing: the optimal weight on the model is zero.
(iv)~Elo and squad market value explain almost everything that is predictable.
A Monte Carlo simulator, backtested on 2023--2025, projects OL at """ + f"{ol[0]:.1f}" + r""" points (80\% interval """ + f"{ol[1]:.0f}--{ol[2]:.0f}" + r"""), an average final position of """ + f"{ol[3]:.1f}" + r""" and a """ + pct(ol[6]) + r""" chance of the top four.
A Champions League extension finds no clear favourite: five clubs sit around 13\%.
\end{minipage}\end{center}
"""

BODY = []
add = BODY.append

# ------------------------------------------------------------------ 1 Introduction
add(r"\section{Introduction}")
add(r"""Football is the sport where the gap between a simple forecast and the best one is smallest: a match is decided by a handful of goals,
the stronger team wins only about one time in two, and one game in four ends in a draw. This makes it a good laboratory for a question that also matters in finance:
\emph{how much can a model extract from public data, how reliably can that be measured, and does it survive comparison with a liquid market?}""")
add(r"""OL plays about 35 league matches a year, far too few to train a model. I therefore train on all five major leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1)
and use OL as a case study. The object to forecast is the following.""")
add(r"""\begin{definition}[Match forecast]
A \emph{match forecast} is a probability vector $p=(p_A,p_D,p_H)$ over the outcomes away win, draw and home win, computed from information available before kick-off.
The \emph{expected points} of the home side are $3p_H+p_D$.
\end{definition}""")
add(r"""For instance, for Paris SG--Lyon on 26 October 2026, the model gives the forecast of Table~\ref{tab:psg}, close to the bookmaker's.""")
cap(r"Forecast for Paris SG--Lyon, 26 October 2026.\label{tab:psg}")
add(tab([["", "OL win", "Draw", "OL loss", "OL expected points"],
         ["Model", "15\\%", "21\\%", "64\\%", "0.67"],
         ["Bookmaker (margin removed)", "15\\%", "20\\%", "65\\%", "--"]], "lcccc"))
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
         ["Wikipedia", "Champions League 2026/27 fixtures and scores", "Section~\\ref{sec:ucl}"]], "p{3.6cm}p{8.2cm}l"))
add(r"""The sample has 29{,}239 matches (593 for OL) and 189 clubs: training 2011--2020 (about 18{,}200 matches), validation 2021--22 (3{,}651) and test October 2023 to October 2026 (5{,}602).
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
cap(r"Walk-forward test, October 2023 to October 2026 (5{,}602 matches).")
add(tab(rows, "lcccc", bold_rows=(4,)))
add(fig("final_logloss.png", 0.8, "Log loss on the test seasons with 95\\% confidence intervals: three model families are tied, the bookmaker is ahead.", "fig:ll"))
add(r"""\begin{result}
The logistic regression beats Elo ($\Delta=-0.0095$, 95\% CI $[-0.0135,-0.0056]$). Gradient boosting ($-0.0007$) and the neural network ($+0.0003$) do \emph{not} beat it: their intervals contain 0.
The bookmaker beats every model ($\Delta=+0.0135$ for the logistic regression, CI $[+0.0094,+0.0175]$). Complexity did not pay.
\end{result}""")
add(r"""\subsection{Calibration}""")
add(fig("final_reliability.png", 0.95, "Reliability diagrams on the test seasons: the closer to the diagonal, the more honest the forecast.", "fig:rel"))
add(r"""The probabilities are well calibrated (ECE between 1.6\% and 2.2\%). The model almost never names a draw as the most likely outcome (6 matches out of 5{,}602): draws are essentially unpredictable.""")
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
and choose the weight $w$ on validation. On the test seasons the best weight on the model is $w=0.00$: the market already contains everything the model knows.
The pooled log loss (0.9663) is not better than the market alone (0.9658). The market takes a margin of 5.5\% on average and is still sharper than the model; the likely reason is information I do not have (line-ups, injuries, news).""")

# ------------------------------------------------------------------ 10 Previous results
add(r"\section{Previous results: was the model right?}")
add(r"""\begin{remark}
These forecasts are recomputed \emph{after the fact} with the walk-forward method (models fitted only on earlier seasons, variables built only from earlier matches).
They were \textbf{not published in advance}. ``Right'' means that the most likely outcome occurred: a 45\% pick is wrong more often than right, by design, so log loss and calibration are the fair scores.
\end{remark}""")
rows = [["Period", "Matches", "Correct", "LL model", "LL bookmaker", "LL no-skill"]]
for k, v in prev["summary"].items():
    rows.append([esc(k), v["n"], pct(v["acc"]), f"{v['ll']:.3f}", f"{v['ll_book']:.3f}", f"{v['ll_naive']:.3f}"])
cap(r"Accuracy and log loss (LL) over recent periods.")
add(tab(rows, "lccccc"))
rows = [["Top pick given", "Matches", "Mean forecast", "Came true"]]
for b in prev["calibration"]:
    rows.append([f"{int(b['lo'] * 100)}--{int(b['hi'] * 100)}\\%", b["n"], pct(b["said"]), pct(b["came_true"])])
cap(r"When the model said $x$\%, how often did it happen? (all 5{,}602 test matches)")
add(tab(rows, "cccc"))
add(r"""When the top pick was given 60--70\%, it happened 63\% of the time. Over the last 30 days the model is right 56\% of the time and is slightly less sharp than the bookmaker.""")
names = ["Away win", "Draw", "Home win"]
rows = [["Date", "Match", "Score", "H / D / A", "Top pick", "Result"]]
for d, lg, h, a, hg, ag, y, p in prev["rows"][:14]:
    best = max(range(3), key=lambda i: p[i])
    rows.append([d[5:], esc(f"{h} -- {a}").replace("--", "--"), f"{hg}--{ag}", f"{100 * p[2]:.0f} / {100 * p[1]:.0f} / {100 * p[0]:.0f}",
                 f"{names[best]} {100 * p[best]:.0f}\\%", "Right" if best == y else "Wrong"])
cap(r"The 14 most recent matches (probabilities in \%).")
add(tab(rows, "llcccl", size=r"\footnotesize"))

# ------------------------------------------------------------------ 11 OL
add(r"\section{Case study: Olympique Lyonnais}")
add(fig("ol_strength_profile.png", 0.8, "OL's profile in Ligue 1 on 26 October 2026 (percentile among clubs; right: rank).", "fig:prof"))
add(r"""OL is third in Ligue 1 by Elo (1{,}615; Paris SG 1{,}707), fifth for recent points and goals scored, and fifth by squad value (EUR 305 m; the defence is only tenth).
Results and rating are better than the shot volume (tenth for shots on target) would suggest.
The 90\% interval on OL's win probability at Paris SG is 14--17\% with model uncertainty only (100 bootstrap refits) and 12--21\% once uncertainty on the true strength of both teams is added (Elo shaken by 60 points).""")

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
cap(r"Ligue 1 2026/27 (state of 26 October 2026, 7 matches played).")
add(tab(rows, "lccccc", bold_rows=(4,)))
add(fig("sim_ol_position.png", 0.8, "Final position of OL over 10{,}000 simulated seasons.", "fig:pos"))
add(rf"""The average final position of OL is \textbf{{{ol[3]:.1f}}}; the most likely finishes are 2nd and 3rd ({pct(ol_pos[1])} and {pct(ol_pos[2])}), but the distribution is wide:
after 7 matches six clubs share 13--14 points, so the forecast rests on team strength rather than on the table.""")

# ------------------------------------------------------------------ 13 Champions League
add(r"\section{Champions League 2026/27}\label{sec:ucl}")
add(rf"""The aim is to use the real fixtures and to combine each club's performance in its own championship. The calendar (144 league-phase matches, 36 clubs) comes from Wikipedia;
only {euro_meta['played']} matches were recorded when I collected it, the others are simulated.
Domestic leagues are only comparable through matches between their clubs, so I fitted a logistic regression on 3{{,}}085 past Champions League, Europa League and Conference League matches (2012--2025):""")
add(r"""\begin{equation}d^{\text{value}}=\ln V_{\text{home}}-\ln V_{\text{away}},\qquad d^{\text{ppg}}=\mathrm{ppg}_{\text{home}}-\mathrm{ppg}_{\text{away}},\end{equation}
where $\mathrm{ppg}$ is the number of domestic league points per match over the last 20 league matches.""")
rows = [["Walk-forward 2023--2025 (943 matches)", "Log loss"]]
for k in ["Class frequency", "Squad value only", "Domestic form only", "Value + domestic form"]:
    rows.append([k, f"{euro.loc[k, 'log_loss']:.4f}"])
cap(r"European strength model.")
add(tab(rows, "lc"))
add(r"""\begin{result}
Squad value carries the signal ($-0.051$ vs the class frequencies, significant). Domestic form adds nothing measurable once value is known ($-0.0005$, 95\% CI $[-0.0048,+0.0037]$).
I keep it because the question asked to combine both, but its weight is small.
\end{result}""")
add(r"""I then simulated 20{,}000 times the rest of the league phase, the play-offs (ranks 9--24), the round of 16, quarter-finals, semi-finals and the final (two legs; extra time and penalties close to a coin flip; the final on neutral ground).""")
rows = [["Club", "Top 8", "Semi-final", "Final", "Winner"]]
for r in ucl.head(10).itertuples():
    rows.append([esc(r.team), pct(r.top8), pct(r.SF), pct(r.Final), pct(r.Winner, 1)])
cap(r"Champions League 2026/27: probabilities from 20{,}000 simulations.")
add(tab(rows, "lcccc"))
add(r"""There is \textbf{no clear favourite}: Arsenal, Manchester City, Barcelona, Real Madrid and Paris SG are each around 13\%, Bayern 10\%, Liverpool 6\%; the differences between the first five are smaller than the uncertainty.
This model is validated match by match, not on whole tournaments; the knockout bracket is approximate; AEK Athens, Sabah and Slovan Bratislava are not in the data and are assumed weak. No Europa League winner is predicted yet.""")

# ------------------------------------------------------------------ 14 Limits, 15 Conclusion
add(r"\section{Limitations}")
add(r"""\begin{itemize}
\item No injuries, suspensions, line-ups or coaching news (the likely reason the bookmaker is better); no real xG.
\item Squad values are a crowd-sourced snapshot (June 2026); 13 clubs have no value.
\item The bookmaker benchmark uses Bet365 pre-match prices with a simple margin removal.
\item Few validation windows; the SVM is under-tuned; online updating was tested for the linear model only.
\item Simulation: intervals are too narrow in the backtest, UEFA places are assumptions, and strength is constant apart from a random shock.
\item Team state excludes the results of 23--25 October 2026, not yet in the data source.
\end{itemize}""")
add(r"\section{Conclusion}")
add(rf"""On public data, football results can be forecast with \emph{{well-calibrated probabilities that are only modestly sharper than team strength alone}}. Elo and squad value explain most of what is predictable.
The chosen model, a regularised multinomial logistic regression, is as good as boosting and neural networks: complexity did not pay. The models cannot beat the betting market and add no information to it,
which is consistent with an efficient market. For OL, the model sees a top-five squad with results slightly ahead of its shot volume: about {ol[0]:.1f} points, an average finish of {ol[3]:.1f} and a coin-flip chance of the top four.
So the disclaimer at the beginning stands: I really hope the ML is wrong.""")
add(r"""\textbf{Reproducibility.} Code, tests (including the leakage test) and the interactive website are in the repository; \texttt{make pipeline} rebuilds data, variables, OL predictions and simulations. All numbers come from \texttt{reports/tables/}.""")

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
    tex = PRE + TITLE + "\n".join(BODY) + REFS + "\n\\end{document}\n"
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
