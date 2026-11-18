"""Build the cloud notebook for the selected HistGradientBoosting model."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)


def cell(kind: str, text: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


cells = [
    cell("markdown", """# PathSnap: Bayesian tuning of HistGradientBoosting

The team selected HistGradientBoosting from the eight initial comparisons. This notebook runs the planned Gaussian-process Bayesian search using five-fold trip-grouped CV on the 80 training trips. The 10 student-test trips are used only for the final tuned-versus-untuned comparison. It does not open or use evaluator data.

Use a high-RAM CPU runtime. GPU is not required. The search uses 12 trials by default and saves every fold result, the best parameters, compute time, comparison metrics, and a training artifact under the Drive project folder."""),
    cell("code", """from google.colab import drive, files
from pathlib import Path
import shutil, zipfile

drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/PathSnap')
ROOT.mkdir(parents=True, exist_ok=True)

# Refresh source files so the tuning script includes the current team-selected model and cache support.
print('Upload the latest PathSnap_source_bundle.zip.')
uploaded = files.upload()
bundle = next((Path(name) for name in uploaded if name.lower().endswith('.zip')), None)
if bundle is None:
    raise FileNotFoundError('Upload PathSnap_source_bundle.zip.')
with zipfile.ZipFile(bundle) as archive:
    archive.extractall(ROOT)

ARCHIVE = ROOT / 'Student folder.zip'
if not ARCHIVE.exists():
    print('Upload the supplied Student folder.zip (not evaluator_test.zip).')
    uploaded = files.upload()
    source = next((Path(name) for name in uploaded if name.lower().endswith('.zip')), None)
    if source is None:
        raise FileNotFoundError('Upload Student folder.zip.')
    shutil.copy2(source, ARCHIVE)

expected = [ROOT / 'results' / f'cloud_{name}' / 'experiment_results.json' for name in [
    'logistic_regression','distance_weighted_knn','random_forest','extra_trees',
    'hist_gradient_boosting','rbf_svm','mlp_shallow','mlp_deep']]
if not all(path.exists() for path in expected):
    print('Upload the ZIP containing the eight cloud_* result folders.')
    uploaded = files.upload()
    results_zip = next((Path(name) for name in uploaded if name.lower().endswith('.zip')), None)
    if results_zip is None:
        raise FileNotFoundError('Upload the ZIP of the eight model results.')
    with zipfile.ZipFile(results_zip) as archive:
        for info in archive.infolist():
            member = Path(info.filename)
            if len(member.parts) == 2 and member.parts[0].startswith('cloud_') and member.suffix == '.json':
                target = ROOT / 'results' / member.parts[0] / member.parts[1]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(info.filename))

print('Project:', ROOT)
print('Training and student data:', ARCHIVE)
"""),
    cell("code", """%cd /content/drive/MyDrive/PathSnap
%pip install -q -r requirements.txt
"""),
    cell("code", """import json, subprocess, sys
from pathlib import Path

# Consolidate the eight completed initial comparisons before tuning.
subprocess.run([sys.executable, 'scripts/consolidate_cloud_runs.py', '--results-root', 'results'], cwd=ROOT, check=True)
config = json.loads((ROOT / 'configs/default.json').read_text(encoding='utf-8'))
config['data_archive'] = str(ARCHIVE)
config_path = Path('/content/pathsnap_hgb_tuning_config.json')
config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')

command = [
    sys.executable, 'scripts/tune_model.py', '--config', str(config_path),
    '--initial-results', 'results/experiment_results.json', '--model', 'hist_gradient_boosting',
    '--trials', '12', '--cache-path', 'results/candidate_cache.joblib', '--output-dir', 'results',
]
print('Running 12 Bayesian trials for HistGradientBoosting.')
subprocess.run(command, cwd=ROOT, check=True)
"""),
    cell("code", """import json
import pandas as pd
from IPython.display import display

tuning = json.loads((ROOT / 'results/bayesian_tuning.json').read_text(encoding='utf-8'))
print('Selected model:', tuning['selected_model'])
print('Best parameters:', tuning['best_parameters'])
print('Best grouped-CV route Macro F1:', tuning['best_objective'])
print('Search compute seconds:', round(tuning['compute_seconds'], 1))
display(pd.DataFrame([tuning['baseline_student_test'], tuning['tuned_student_test']], index=['untuned','tuned']))
display(pd.DataFrame([{'trial': row['trial'], 'parameters': row['parameters'], 'cv_route_macro_f1': row['cv_route_macro_f1']} for row in tuning['trials']]))
"""),
    cell("markdown", """## Share the outputs

Keep `results/bayesian_tuning.json`, `artifacts/final_model.joblib`, `artifacts/final_metadata.json`, and this executed notebook with its outputs. Share those files back for the decoder comparison and report. This artifact is the tuned classifier; decoder settings still need the permitted development comparison before the final deployment package is frozen."""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "09_hgb_bayesian_tuning.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(NOTEBOOKS / "09_hgb_bayesian_tuning.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
