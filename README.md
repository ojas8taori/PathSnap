# PathSnap

PathSnap maps noisy, time-ordered GPS observations to a plausible sequence of road arcs from each trip's supplied graph. The final pipeline uses a tuned HistGradientBoosting classifier to score candidate arcs, followed by an HMM/Viterbi decoder to select a route that fits those scores and the directed road connectivity.

## Project state and result

The eight-model comparison, HGB Bayesian tuning, decoder evaluation, reserved-evaluator run, FastAPI service and evaluator UI are implemented. The consolidated report draft is `reports/PathSnap_Technical_Report.docx`. The run data and report figures are stored under `results/` and `visualizations/`.

On the 10 reserved evaluator trips, mean trip Top-1 accuracy was 0.8736 with HMM/Viterbi and 0.8081 with classifier-only ranking. Mean route Macro F1 was 0.7280 versus 0.6195, continuity was 0.9429 versus 0.8873, and candidate miss rate was 0.0202 for both. These are final evaluation measurements; they did not inform model or decoder selection. The per-trip values, artifact and archive hashes, settings, and runtime are in `results/final_evaluator.json`.

## Data partitions and contract

The student archive contains 80 training trips and 10 student-test trips. The separate evaluator archive contains 10 reserved trips. Keep the archives outside Git and configure their paths locally. Use training trips for fitting and grouped cross-validation, student-test trips for development comparisons and decoder setting selection, and the reserved evaluator trips only for the final frozen run.

Each trip has whitespace-separated, headerless files: `.track` contains longitude, latitude and timestamp seconds; `.nodes` contains longitude and latitude with row number as node ID; `.arcs` contains integer from/to node IDs with row number as arc ID; and `.route` contains the ordered reference arc IDs. Route and arc IDs are trip-local. A GPS row does not map directly to a node or arc row. See `docs/data_contract.md` for validation and labeling details.

## Install

Use Python 3.11 and the project requirements. Scikit-learn is pinned to 1.6.1 to match the serialized final model.

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Set the student archive path in `configs/default.json` before regenerating data audits, EDA, comparison, tuning or decoder results. The original development path is `C:/OJ/sem5/ML/hackathon/Student folder.zip`.

## Reproduce the experiments

```powershell
.venv\Scripts\python.exe scripts\audit_dataset.py "C:\OJ\sem5\ML\hackathon\Student folder.zip" --output results\dataset_audit.json
.venv\Scripts\python.exe scripts\run_eda.py --config configs/default.json
.venv\Scripts\python.exe scripts\run_experiments.py --config configs/default.json
.venv\Scripts\python.exe scripts\consolidate_cloud_runs.py
.venv\Scripts\python.exe scripts\tune_model.py --config configs/default.json --model hist_gradient_boosting --trials 12
.venv\Scripts\python.exe scripts\evaluate_decoder.py --config configs/default.json
```

The eight selected approaches are Logistic Regression, distance-weighted KNN, Random Forest, Extra Trees, HistGradientBoosting, RBF SVM, shallow MLP and deep MLP. The reasoned scope and omitted alternatives are in `docs/model_scope.md`. All use the same candidate generation and twelve inference-time features. Validation uses five folds grouped by trip.

## Final evaluator run

After the model and decoder settings are fixed, run the evaluator-only script. It scores the frozen artifact on the reserved evaluator package and does not fit or retune anything.

```powershell
.venv\Scripts\python.exe scripts\evaluate_final.py "C:\Users\ojas9\Downloads\evaluator_test.zip" --artifact artifacts\final_model.joblib --partition evaluator_test --output results\final_evaluator.json
```

The API and this script build route-free inputs for prediction. Reference routes stay server-side and are read separately for scoring.

## Representative route map

Generate an interactive Folium map and static report figure for a student-test trip:

```powershell
.venv\Scripts\python.exe scripts\create_representative_route_map.py --archive "C:\OJ\sem5\ML\hackathon\Student folder.zip" --trip-id 00000007
```

The HTML contains separate layers for GPS observations, nearby road network, supplied reference route, and HGB plus HMM/Viterbi prediction. It is saved at `visualizations/representative_trip.html`; the static image used in the report is `visualizations/representative_trip.png`.

To rebuild the report plots and editable DOCX after updating experiment outputs:

```powershell
.venv\Scripts\python.exe scripts\build_report_assets.py
.venv\Scripts\python.exe scripts\build_report_docx.py
```

## Run the API and UI

Keep the API and UI in one repository. For local evaluation, use two PowerShell windows from the repository root.

API window:

```powershell
$env:PATHSNAP_DATA_ARCHIVE = "C:\Users\ojas9\Downloads\evaluator_test.zip"
$env:PATHSNAP_ARTIFACT = "artifacts\final_model.joblib"
$env:PATHSNAP_EVALUATOR_PARTITION = "evaluator_test"
.venv\Scripts\python.exe -m uvicorn src.api:app --host 127.0.0.1 --port 8001
```

UI window:

```powershell
.venv\Scripts\python.exe -m http.server 8502 --bind 127.0.0.1 --directory ui
```

Open `http://127.0.0.1:8502`. `ui/config.json` points to API port 8001. Check `http://127.0.0.1:8001/health` for readiness. The UI retrieves trip IDs from `/trips` and sends only the selected trip ID to `/predict`. The API logs received IDs and results; it never returns reference route IDs to the UI. CORS includes localhost UI origins on ports 8501 and 8502. For detailed two-laptop commands and firewall/CORS settings, see `docs/two_laptop_workflow.md`.

## Important files

- `src/`: data loading, candidate features, experiment estimators, decoder, inference and FastAPI service.
- `ui/`: evaluator interface and API address configuration.
- `notebooks/`: eight model runs, HGB tuning and HMM/Viterbi evaluation notebooks.
- `artifacts/`: final HGB model and metadata containing feature schema, candidate rule, decoder settings and versions.
- `results/`: actual audits, model comparisons, trial logs, decoder results and reserved evaluator metrics.
- `reports/PathSnap_Technical_Report.docx`: consolidated report draft with actual result tables and visual evidence.
- `scripts/`: reproducible audits, comparisons, tuning, map generation, final evaluation and report generation.

## Reproducibility notes

The HGB parameters are `learning_rate=0.0655946`, `max_iter=167`, `max_leaf_nodes=51`, and `l2_regularization=1.0`. Candidate generation uses an 80 m radius and top 8 arcs. The selected decoder settings are `max_gap_seconds=120`, `max_speed_mps=45`, `sigma_m=80`, and `disconnected_log_penalty=-12`. The selected student-test decoder score is a development selection measurement, not a blind test score. The independent reserved evaluator result is reported separately.

The local repository contains all code and results. No GitHub remote is configured yet. Keep raw archives out of the repository. Before external publication, the team should review the report and the event's AI assistance disclosure requirements.
