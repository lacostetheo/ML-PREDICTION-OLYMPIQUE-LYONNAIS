"""Consistency checks of the exported website data (skipped if site/data.js is absent)."""
import json
from pathlib import Path

import pytest

DATA = Path("site/data.js")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run `python -m olpred.export_site` first")


@pytest.fixture(scope="module")
def d():
    t = DATA.read_text()
    return json.loads(t[t.index("{"): t.rindex("}") + 1])


def test_pair_probabilities_are_valid(d):
    n = 0
    for pairs in d["pairs"].values():
        for key, v in pairs.items():
            assert abs(sum(v[:3]) - 1) < 2e-3 and min(v[:3]) > 0, key
            assert v[3] <= v[0] <= v[4] or v[3] <= v[0] + 1e-3, key  # model-mean inside/near its interval
            assert len(v) == 18
            n += 1
    assert n == 1752  # 3 x 380 + 2 x 306 ordered pairs


def test_every_team_has_a_next_fixture_and_ligue1_has_three_upcoming(d):
    for code, lg in d["leagues"].items():
        assert set(d["next"][code]) == set(lg["teams"])
    for team, ms in d["calendar_f1"].items():
        assert len(ms) == 3 and all(m["date"] >= d["as_of"] for m in ms) and all(team in (m["home"], m["away"]) for m in ms)
        assert [m["date"] for m in ms] == sorted(m["date"] for m in ms)


def test_olympique_lyonnais_upcoming_matches_match_the_published_calendar(d):
    got = [(m["home"], m["away"]) for m in d["calendar_f1"]["Lyon"]]
    assert got == [("Paris SG", "Lyon"), ("Lyon", "Angers"), ("Brest", "Lyon")]  # Lyon Capitale / Maxifoot, 26 Oct 2026


def test_arena_and_simulation_blocks(d):
    assert abs(d["arena"]["final"]["LogReg (final)"]["log_loss"] - 0.9793) < 1e-3
    s = d["leagues"]["F1"]["sim"]
    assert abs(s["summary"]["Lyon"][0] - 57.6) < 0.3
    assert all(abs(sum(p) - 1) < 1e-2 for p in s["pos_probs"].values())
