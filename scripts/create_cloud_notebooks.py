"""Create one runnable Google Colab notebook for each selected model."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)

MODELS = [
    ("01_logistic_regression.ipynb", "logistic_regression", "Logistic Regression"),
    ("02_distance_weighted_knn.ipynb", "distance_weighted_knn", "Distance-weighted KNN"),
    ("03_random_forest.ipynb", "random_forest", "Random Forest"),
    ("04_extra_trees.ipynb", "extra_trees", "Extra Trees"),
    ("05_hist_gradient_boosting.ipynb", "hist_gradient_boosting", "HistGradientBoosting"),
    ("06_rbf_svm.ipynb", "rbf_svm", "RBF SVM"),
    ("07_mlp_shallow.ipynb", "mlp_shallow", "Shallow MLP"),
    ("08_mlp_deep.ipynb", "mlp_deep", "Deep MLP"),
]


def source_cell(cell_type: str, source: str) -> dict:
    return {"cell_type": cell_type, "metadata": {}, "source": source.splitlines(keepends=True), **({"execution_count": None, "outputs": []} if cell_type == "code" else {})}


for filename, model_key, display in MODELS:
    cells = [
        source_cell("markdown", f"# PathSnap model run: {display}\n\nThis notebook trains only **{display}** using trip-grouped cross-validation on the 80 training trips and compares it on the 10 student-test trips. It uses the shared PathSnap feature and evaluation code. It does not access evaluator data.\n\nUse a high-RAM CPU runtime. The estimator is CPU based; a GPU is not required. Keep Google Drive mounted so the candidate cache and result files persist between notebooks."),
        source_cell("code", """from google.colab import drive, files
from pathlib import Path
import json, shutil, subprocess, sys, zipfile

drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/PathSnap')
if not (ROOT / 'scripts/run_experiments.py').exists():
    print('Upload PathSnap_source_bundle.zip, created by scripts/package_source.py')
    uploaded = files.upload()
    bundle = next((Path(name) for name in uploaded if name.lower().endswith('.zip')), None)
    if bundle is None:
        raise FileNotFoundError('Upload the PathSnap source bundle ZIP.')
    ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(ROOT)

ARCHIVE = ROOT / 'Student folder.zip'
if not ARCHIVE.exists():
    print('Upload the supplied Student folder.zip (do not upload evaluator data).')
    uploaded = files.upload()
    chosen = next((Path(name) for name in uploaded if name.lower().endswith('.zip')), None)
    if chosen is None:
        raise FileNotFoundError('Upload Student folder.zip.')
    shutil.copy2(chosen, ARCHIVE)

print('Project:', ROOT)
print('Student archive:', ARCHIVE)
"""),
        source_cell("code", """%cd /content/drive/MyDrive/PathSnap
%pip install -q -r requirements.txt
"""),
        source_cell("markdown", "## Run the selected model\n\nThe first notebook prepares and saves `results/candidate_cache.joblib` on Drive. Later model notebooks reuse it. Let the preparation step finish; it processes each complete trip and does not split trips across folds."),
        source_cell("code", f"""import json, subprocess, sys

config = json.loads((ROOT / 'configs/default.json').read_text(encoding='utf-8'))
config['data_archive'] = str(ARCHIVE)
config_path = Path('/content/pathsnap_{model_key}_config.json')
config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
output_dir = ROOT / 'results' / 'cloud_{model_key}'
cache_path = ROOT / 'results' / 'candidate_cache.joblib'
command = [
    sys.executable, str(ROOT / 'scripts/run_experiments.py'),
    '--config', str(config_path), '--model', '{model_key}',
    '--cache-path', str(cache_path), '--output-dir', str(output_dir),
]
print('Running:', ' '.join(command))
subprocess.run(command, cwd=ROOT, check=True)
"""),
        source_cell("code", f"""import pandas as pd

result = json.loads((output_dir / 'experiment_results.json').read_text(encoding='utf-8'))
model_result = result['models']['{model_key}']
print('Feature preparation seconds:', round(result['preparation_seconds'], 1))
print('Configuration:', model_result['configuration'])
display(pd.DataFrame(model_result['cv']['folds']))
display(pd.DataFrame([model_result['student_test']]))
print('Saved outputs:', output_dir)
"""),
        source_cell("markdown", "## Return these run artifacts\n\nSave this executed notebook with its cell outputs. Also share the matching `results/cloud_MODEL/` folder, which contains machine-readable fold metrics, student-test metrics and per-trip candidate predictions. Do not edit reported values by hand. The team will compare all eight run outputs before selecting and tuning a model."),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"name": filename, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (OUT / filename).write_text(json.dumps(notebook, indent=2), encoding="utf-8")
