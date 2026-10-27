"""Consistency checks of the exported website data (skipped if site/data.js is absent)."""
import json
from pathlib import Path

import pytest

DATA = Path("site/data.js")
pytestmark = pytest.mark.skipif(not DATA.exists(), reason="run `python -m olpred.export_site` first")


@pytest.fixture(scope="module")
def d():
    t = DATA.read_text(encoding="utf8")
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


def test_every_remaining_league_fixture_is_forecast_once(d):
    for code, lg in d["leagues"].items():
        n = len(lg["teams"])
        played = lg["sim"]["summary"]
        left = d["fixtures"][code]
        assert len(left) == n * (n - 1) - sum(v[11] for v in played.values()) / 2, code
        seen = set()
        for m in left:
            assert (m["h"], m["a"]) not in seen and m["h"] != m["a"]
            seen.add((m["h"], m["a"]))
            assert f"{m['h']}|{m['a']}" in d["pairs"][code]
            assert abs(sum(m["p"]) - 1) < 2e-3 and min(m["p"]) > 0
            assert m["d"] >= d["as_of"] or m.get("late") == 1  # a past date means "postponed", and is flagged


def test_european_competitions(d):
    for comp, name in (("ucl", "Champions League"), ("uel", "Europa League")):
        e = d["europe"][comp]
        assert e["name"] == name and len(e["teams"]) == 36
        assert abs(sum(t["w"] for t in e["teams"]) - 1) < 1e-2          # exactly one winner per simulation
        assert abs(sum(t["f"] for t in e["teams"]) - 2) < 1e-2          # two finalists
        assert abs(sum(t["top8"] for t in e["teams"]) - 8) < 1e-2 and abs(sum(t["top24"] for t in e["teams"]) - 24) < 1e-2
        for m in e["fixtures"]:
            assert abs(sum(m["p"]) - 1) < 2e-3 and m["d"] >= d["as_of"]
        assert e["played"] + len(e["fixtures"]) + e["missing"] == e["total"]
    assert any(t["k"] == "Lyon" for t in d["europe"]["uel"]["teams"])   # OL plays the Europa League in 2026/27


def test_simulation_and_arena_blocks(d):
    for lg in d["leagues"].values():
        s = lg["sim"]
        assert abs(sum(v[4] for v in s["summary"].values()) - 1) < 1e-2          # one champion per simulated season
        assert all(abs(sum(p) - 1) < 1e-2 for p in s["pos_probs"].values())
        assert lg["rounds"]["total"] == 2 * (len(lg["teams"]) - 1)
    assert 0.97 < d["arena"]["final"]["LogReg (final)"]["log_loss"] < 0.99
    assert d["arena"]["final"]["Bookmaker Bet365 (reference)"]["log_loss"] < d["arena"]["final"]["LogReg (final)"]["log_loss"]


def test_previous_results_block(d):
    p = d["prev"]
    assert p["last"] <= d["data_through"] and len(p["rows"]) == 150
    assert p["rows"][0][0] >= p["rows"][-1][0]                                   # most recent first
    assert all(abs(sum(r[7]) - 1) < 2e-3 for r in p["rows"])
    assert sum(b["n"] for b in p["calibration"]) == p["n_total"]
