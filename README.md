# PathSnap

PathSnap is a trip-level map matcher for noisy GPS tracks and their supplied road graphs. The project reads the student archive at its configured location; the archive is not copied into this repository.

## Current state

The repository is being built from the supplied execution prompt and project guidelines. The archive audit is reproducible with `scripts/audit_dataset.py`, and its raw JSON output is in `results/dataset_audit.json`. No model results, report claims, or evaluator outputs are written until corresponding experiments have run.

The current experiment plan follows the team's latest instruction to compare eight approaches: Logistic Regression, distance-weighted KNN, Random Forest, Extra Trees, HistGradientBoosting, RBF SVM, shallow MLP, and deep MLP. These cover several model families while keeping the suite within scikit-learn. The remaining suggested approaches will be listed with the reason for exclusion in the experiment record.

## Data location and contract

Set `data_archive` in `configs/default.json` to the supplied ZIP. The default path is the original local path on the development machine. The ZIP has `train` (80 trips) and `student_test` (10 trips); it has no evaluator partition. Keep the evaluator package separate when organizers provide it. Do not place any partition labels in the prediction input.

Files are whitespace-separated and headerless: `.track` has longitude, latitude, timestamp seconds; `.nodes` has longitude, latitude (row number is node ID); `.arcs` has integer from/to node IDs (row number is arc ID); `.route` is an ordered list of reference arc IDs. A trip's route labels belong to that trip's own graph and are not class IDs that can be shared across trips.

## Planned commands

Create an environment and install the declared packages, then run:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python scripts\audit_dataset.py "C:\OJ\sem5\ML\hackathon\Student folder.zip" --output results\dataset_audit.json
python scripts\run_eda.py --config configs/default.json
python scripts\run_experiments.py --config configs/default.json
python scripts\tune_model.py --config configs/default.json
python scripts\evaluate_decoder.py --config configs/default.json
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

The audit and EDA commands have been run against the supplied archive. The experiment, tuning, API and UI commands are the implemented workflow; cloud training outputs are pending.

## Cloud model runs

The local machine is too slow for full candidate preparation and five-fold model comparison. Eight separate Colab notebooks are in `notebooks/`, one for each selected approach. Use a high-RAM CPU runtime; a GPU is not required. Put a copy of this repository and the supplied ZIP in Google Drive under `MyDrive/PathSnap/`, then open and run each notebook. The first notebook run builds `results/candidate_cache.joblib`; later notebooks reuse that cache so candidate generation is performed once. Each notebook writes its actual fold metrics, student-test metrics, timings and predictions under `results/cloud_<model>/`.

After each notebook finishes, download or share the executed `.ipynb` with its outputs and the corresponding `results/cloud_<model>/` folder. Those run artifacts are needed to consolidate the comparison, choose the final model, tune it, evaluate Viterbi decoding and write the evidence-based report. The notebooks install only this repository's declared dependencies and run the shared experiment implementation; they do not contain separate model code.

## Deployment

The intended local mode runs API and UI on one machine. A split-laptop option will use the same repository: set API bind host/port and UI API base URL via environment/configuration, allow the UI host through CORS and the network firewall, start the API first, and check `/health` before opening the UI. The API owns model loading, road graph, evaluator references, prediction, and post-prediction scoring. The UI sends only a trip ID and renders returned predictions and permitted summary metrics.

## Integrity and reproducibility

Training and cross-validation use complete trips as groups. Student-test trips are for development comparisons only. Evaluator trips and labels are not available in the supplied archive and will not be synthesized. Reports and plots will be generated from saved experiment outputs; unavailable results will remain marked as pending until measured.
