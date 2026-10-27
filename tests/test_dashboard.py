"""Smoke test of the Streamlit app (skipped when the pipeline outputs are absent)."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
NEEDED = [Path("models_store/production.joblib"), Path("data/processed/team_state.parquet"),
          Path("data/processed/sim_F1.npz")]


@pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="run `make pipeline` first")
def test_app_runs_and_reacts_to_inputs():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception
    selects = {s.label: s for s in at.selectbox}
    selects["Opponent"].select("Paris SG").run()
    assert not at.exception
    home_pts = float(at.metric[0].value)
    at.radio[0].set_value("Away").run()  # same teams, venue swapped -> different expected points
    assert not at.exception and float(at.metric[0].value) != home_pts
    {s.label: s for s in at.selectbox}["League "].select("E0").run()
    assert not at.exception
