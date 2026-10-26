"""Leakage tests: features of a match must not depend on that match or later ones.

Method (the "perturbation test"): rewrite every result/statistic from a cutoff
date onwards with random values, rebuild all features, and check that the
features of every match on or before the cutoff are *identical*. Rows on the
cutoff date test independence from the match's own result.
"""
import numpy as np
import pandas as pd
import pytest

from olpred.data.clean import RESULT_TO_Y
from olpred.features.build import build_features
from olpred.features.elo import compute_elo
from olpred.features.squad_value import MIN_VALUED_PLAYERS, monthly_snapshots, squad_features

TARGET_COLS = {"home_goals", "away_goals", "result", "y", "pts_home", "pts_away"}


def synthetic_matches(seed: int = 0, n_teams: int = 8, seasons=(2010, 2011, 2012)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    rows = []
    for season in seasons:
        day = pd.Timestamp(f"{season}-08-10")
        order = list(teams)
        rounds = []
        for _ in range(n_teams - 1):  # circle method: each team plays once per round
            rounds.append([(order[i], order[n_teams - 1 - i]) for i in range(n_teams // 2)])
            order = [order[0]] + [order[-1]] + order[1:-1]
        rounds += [[(a, h) for h, a in r] for r in rounds]  # second leg, venues swapped
        for r in rounds:
            rows += [(day, season, "XX", h, a) for h, a in r]
            day += pd.Timedelta(days=7)
    m = pd.DataFrame(rows, columns=["date", "season", "league", "home", "away"])
    return randomise_results(m, rng)


def randomise_results(m: pd.DataFrame, rng) -> pd.DataFrame:
    m = m.copy()
    n = len(m)
    m["home_goals"] = rng.poisson(1.5, n)
    m["away_goals"] = rng.poisson(1.1, n)
    m["hs"] = m.home_goals * 4 + rng.integers(3, 10, n)
    m["as_"] = m.away_goals * 4 + rng.integers(3, 10, n)
    m["hst"] = np.minimum(m.hs, m.home_goals + rng.integers(0, 5, n))
    m["ast"] = np.minimum(m.as_, m.away_goals + rng.integers(0, 5, n))
    for c in ["hc", "ac", "hf", "af", "hy", "ay", "hr", "ar", "b365_h", "b365_d", "b365_a",
              "avg_h", "avg_d", "avg_a"]:
        m[c] = np.nan
    m["result"] = np.select([m.home_goals > m.away_goals, m.home_goals < m.away_goals], ["H", "A"], "D")
    m["y"] = m["result"].map(RESULT_TO_Y)
    m["pts_home"] = m["y"].map({0: 0, 1: 1, 2: 3})
    m["pts_away"] = m["y"].map({0: 3, 1: 1, 2: 0})
    m = m.drop(columns=["match_id"], errors="ignore")
    m = m.sort_values(["date", "home"], kind="stable").reset_index(drop=True)
    m.insert(0, "match_id", np.arange(len(m)))
    return m


def perturb_from(m: pd.DataFrame, cutoff: pd.Timestamp, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    future = m["date"] >= cutoff
    fresh = randomise_results(m, rng)
    out = m.copy()
    for c in TARGET_COLS | {"hs", "as_", "hst", "ast"}:
        out.loc[future, c] = fresh.loc[future, c]
    return out


@pytest.mark.parametrize("day_index", [16, 25, 40])
def test_features_do_not_depend_on_cutoff_or_future(day_index):
    m = synthetic_matches()
    # cutoff is a real match date, so matches ON the cutoff day check independence from their own result
    cutoff = sorted(m["date"].unique())[day_index]
    assert (m["date"] == cutoff).any()
    assert (m["date"] >= cutoff).sum() >= 4 and (m["date"] < cutoff).sum() > 10

    # xG proxy is fitted on the first season only, so perturbing later data cannot touch it
    original = build_features(m, xg_train_end=2010)
    altered = build_features(perturb_from(m, cutoff), xg_train_end=2010)

    feature_cols = [c for c in original.columns if c not in TARGET_COLS | {"match_id"}]
    past = original["date"] <= cutoff
    pd.testing.assert_frame_equal(
        original.loc[past, feature_cols].reset_index(drop=True),
        altered.loc[altered["date"] <= cutoff, feature_cols].reset_index(drop=True),
        check_exact=True,
    )
    # sanity: the test can fail — features after the cutoff really do change
    assert not original.loc[~past, feature_cols].equals(altered.loc[~past, feature_cols])


def test_first_matches_use_no_history():
    m = synthetic_matches()
    feats = build_features(m, xg_train_end=2010)
    first_day = feats[feats["date"] == feats["date"].min()]
    assert (first_day["h_n_prior"] == 0).all() and (first_day["a_n_prior"] == 0).all()
    assert (first_day["h_games_played_season"] == 0).all()


def test_elo_is_pre_match_and_zero_sum():
    m = synthetic_matches()
    elo = compute_elo(m)
    first = m["date"] == m["date"].min()
    assert (elo.loc[first, "elo_home"] == 1500).all()  # initial rating, no result seen yet
    # changing a single result only affects later matches of the two teams involved
    m2 = m.copy()
    m2.loc[0, ["home_goals", "away_goals"]] = [5, 0]
    elo2 = compute_elo(m2)
    assert elo.loc[0, ["elo_home", "elo_away"]].equals(elo2.loc[0, ["elo_home", "elo_away"]])
    assert (elo["elo_home"] != elo2["elo_home"]).any()


def test_squad_value_ignores_valuations_on_or_after_snapshot():
    rows = [(p, pd.Timestamp("2015-03-01"), 1_000_000.0 * (p + 1), 7)
            for p in range(MIN_VALUED_PLAYERS)]
    leak = [(99, pd.Timestamp("2015-06-01"), 500_000_000.0, 7)]  # dated on the snapshot day
    v = pd.DataFrame(rows + leak, columns=["player_id", "date", "market_value_in_eur", "current_club_id"])
    snaps = monthly_snapshots(v, {7}, start="2015-06-01", end="2015-06-01")
    assert snaps.loc[0, "sv_n"] == MIN_VALUED_PLAYERS
    assert snaps.loc[0, "sv_total"] == sum(1_000_000.0 * (p + 1) for p in range(MIN_VALUED_PLAYERS))
    # a club with too few valued players gets no squad value
    club_map = pd.DataFrame({"league": ["XX"], "team": ["T0"], "club_id": [7]})
    thin = snaps.assign(sv_n=3)
    matches = pd.DataFrame({"league": ["XX"], "home": ["T0"], "away": ["T0"],
                            "date": [pd.Timestamp("2015-06-20")]})
    assert squad_features(matches, club_map, thin)["sv_total_rel"].isna().all()


def test_future_fixture_features_equal_training_features():
    """The phantom-match 'team state' must reproduce, for any past date, exactly the
    features the training table holds for the matches played that day."""
    from pathlib import Path

    if not (Path("data/processed/features.parquet").exists() and Path("data/raw/transfermarkt").exists()):
        pytest.skip("processed data not available")
    from olpred.features.build import squad_resources
    from olpred.features.columns import curated_cols
    from olpred.predict.state import fixture_frame, team_state

    f = pd.read_parquet("data/processed/features.parquet")
    m = pd.read_parquet("data/processed/matches.parquet")
    club_map, snaps = squad_resources(m, end=(f["date"].max() + pd.offsets.MonthBegin(1)).strftime("%Y-%m-01"))
    as_of = pd.Timestamp("2025-11-08")
    day = f[f["date"] == as_of].reset_index(drop=True)
    assert len(day) > 5
    ours = fixture_frame(day[["league", "home", "away"]], team_state(m, as_of, club_map, snaps))
    cols = [c for c in curated_cols(f) if "rest_days" not in c]  # rest days differ only by schedule rounding
    pd.testing.assert_frame_equal(ours[cols].astype(float), day[cols].astype(float), check_exact=False,
                                  rtol=1e-9, atol=1e-9)
