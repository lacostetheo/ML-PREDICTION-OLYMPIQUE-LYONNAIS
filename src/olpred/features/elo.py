"""Home-made Elo ratings (online, leakage-free by construction).

The rating attached to a match is the one *before* it is played; the update
uses the result and is only visible to later matches.
"""
from __future__ import annotations

import math

import pandas as pd


def compute_elo(
    matches: pd.DataFrame,
    k: float = 20.0,
    home_adv: float = 65.0,
    season_carry: float = 0.75,
    entrant_offset: float = -75.0,
    base: float = 1500.0,
) -> pd.DataFrame:
    """Return ``match_id, elo_home, elo_away, elo_exp_home`` (pre-match values).

    - Goal-difference multiplier ``1 + ln(1 + |gd|)`` on the update size.
    - Between seasons ratings shrink toward ``base`` (``season_carry`` = share kept).
    - A team that was not in the league the previous season (promoted / first
      appearance) starts at ``base + entrant_offset``; every team of the very
      first season starts at ``base``.
    Ratings are kept separately per league (no cross-league games in the data).
    """
    ratings: dict[tuple[str, str], float] = {}
    last_season: dict[tuple[str, str], int] = {}
    first_season = int(matches["season"].min())
    current: dict[str, int] = {}
    rows = []

    for m in matches.sort_values(["date", "match_id"]).itertuples(index=False):
        lg = m.league
        if current.get(lg) != m.season:  # first match of a new season in this league
            current[lg] = m.season
            for key in [kk for kk in ratings if kk[0] == lg]:
                if last_season[key] == m.season - 1:
                    ratings[key] = base + season_carry * (ratings[key] - base)
        for team in (m.home, m.away):
            key = (lg, team)
            if key not in ratings or last_season[key] < m.season - 1:
                ratings[key] = base if m.season == first_season else base + entrant_offset
            last_season[key] = m.season

        rh, ra = ratings[(lg, m.home)], ratings[(lg, m.away)]
        exp_h = 1.0 / (1.0 + 10 ** (-(rh - ra + home_adv) / 400.0))
        rows.append((m.match_id, rh, ra, exp_h))

        score_h = 1.0 if m.home_goals > m.away_goals else 0.5 if m.home_goals == m.away_goals else 0.0
        mult = 1.0 + math.log1p(abs(m.home_goals - m.away_goals))
        delta = k * mult * (score_h - exp_h)
        ratings[(lg, m.home)] = rh + delta
        ratings[(lg, m.away)] = ra - delta

    return pd.DataFrame(rows, columns=["match_id", "elo_home", "elo_away", "elo_exp_home"])
