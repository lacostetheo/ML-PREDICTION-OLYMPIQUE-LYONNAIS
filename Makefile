# Reproduce the whole analysis:  make all      (needs internet for the first data download)
PY ?= python
export PYTHONPATH := src

.PHONY: weekly report europe site all install data features baselines ml nn interpret final ablation backtest ol simulate figures pipeline dashboard test clean

all: features baselines ml nn interpret final ablation backtest ol simulate figures

install:
	$(PY) -m pip install -r requirements.txt

data features:           ## download results + Transfermarkt values, build the leakage-free feature table
	$(PY) -m olpred.features.build

baselines:               ## Phase 4
	$(PY) -m olpred.run_baselines && $(PY) -m olpred.viz.baseline_plots

ml:                      ## Phase 5 (random search with time-ordered CV; ~10 min)
	$(PY) -m olpred.run_ml && $(PY) -m olpred.run_online

nn:                      ## Phase 6 (~10 min)
	$(PY) -m olpred.run_nn

interpret:               ## Phase 7
	$(PY) -m olpred.run_interpret && $(PY) -m olpred.viz.interpret_plots

final:                   ## walk-forward evaluation on the test seasons + diagnostics
	$(PY) -m olpred.run_final && $(PY) -m olpred.run_final_analysis && $(PY) -m olpred.viz.final_plots

ablation:
	$(PY) -m olpred.run_ablation

backtest:                ## simulator backtest + choice of the rating-uncertainty sigma
	$(PY) -m olpred.run_backtest

ol:                      ## Phase 8: OL strength and match predictions (state = today)
	$(PY) -m olpred.run_ol

simulate:                ## Phase 9: Monte Carlo seasons for the five leagues
	$(PY) -m olpred.run_season_sim

figures:
	$(PY) -m olpred.viz.ml_plots && $(PY) -m olpred.viz.ol_plots

# Minimal pipeline needed by the dashboard (skips the model comparison)
pipeline: features backtest ol simulate

europe:                  ## Champions League and Europa League winners (reads the Wikipedia league-phase pages)
	$(PY) -m olpred.europe && $(PY) -m olpred.europe_sim

# Refresh everything the website shows (~40 min, needs internet): results, team state, simulations, Europe, site
weekly: features final ol simulate europe site

report:                  ## paper rebuilt from the latest data: reports/OL_ML_project_updated.pdf (needs pdflatex)
	$(PY) -m olpred.report_tex

site:                    ## export data for the website and build a single-file version
	$(PY) -m olpred.export_site && $(PY) -m olpred.build_standalone

dashboard:
	streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest -q

clean:
	rm -rf data/interim/* data/processed/* models_store
