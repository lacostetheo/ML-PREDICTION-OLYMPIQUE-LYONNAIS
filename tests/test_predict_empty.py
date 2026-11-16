import pandas as pd

from olpred.predict.uncertainty import match_intervals


def test_match_intervals_without_fixtures():
    """The weekly refresh must survive an international break: no fixture, no crash, same columns."""
    out = match_intervals([], pd.DataFrame(), 60.0)
    assert out.empty
    for c in ("p_home", "p_draw", "p_away", "exp_pts_home", "p_home_total_lo", "exp_pts_away_model_hi"):
        assert c in out.columns
