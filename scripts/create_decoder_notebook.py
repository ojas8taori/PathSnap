"""Create the cloud notebook for the selected-model HMM/Viterbi evaluation."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"
OUT.mkdir(exist_ok=True)


def cell(kind: str, text: str) -> dict:
    result = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


cells = [
    cell("markdown", """# PathSnap: HMM/Viterbi route evaluation

This notebook compares the tuned HistGradientBoosting classifier's top candidate with the same model followed by HMM/Viterbi. It reports five-fold trip-grouped CV on the 80 training trips, then compares a small decoder-parameter grid on the 10 student-test trips. The selected student setting is development-tuned and must not be described as an independent test result. Evaluator data is not used here.

Use a high-RAM CPU runtime. The decoder runs graph transitions and can take several minutes."""),
    cell("code", """from google.colab import drive, files
from pathlib import Path
import shutil, zipfile

drive.mount('/content/drive')
ROOT = Path('/content/drive/MyDrive/PathSnap')
ROOT.mkdir(parents=True, exist_ok=True)

print('Upload the latest PathSnap_source_bundle.zip to refresh the decoder script.')
uploaded = files.upload()
bundle = next((Path(name) for name in uploaded if name.lower().endswith('.zip')), None)
if bundle is None:
    raise FileNotFoundError('Upload PathSnap_source_bundle.zip.')
with zipfile.ZipFile(bundle) as archive:
    archive.extractall(ROOT)

ARCHIVE = ROOT / 'Student folder.zip'
if not ARCHIVE.exists():
    print('Upload Student folder.zip (do not upload evaluator_test.zip).')
    uploaded = files.upload()
    source = next((Path(name) for name in uploaded if name.lower().endswith('.zip')), None)
    if source is None:
        raise FileNotFoundError('Upload Student folder.zip.')
    shutil.copy2(source, ARCHIVE)

METADATA = ROOT / 'artifacts/final_metadata.json'
MODEL = ROOT / 'artifacts/final_model.joblib'
if not METADATA.exists() or not MODEL.exists():
    print('Upload the HGB tuning artifact ZIP containing final_metadata.json and final_model.joblib.')
    uploaded = files.upload()
    package = next((Path(name) for name in uploaded if name.lower().endswith('.zip')), None)
    if package is None:
        raise FileNotFoundError('Upload the HGB tuning artifact ZIP.')
    with zipfile.ZipFile(package) as archive:
        for member in ('final_metadata.json', 'final_model.joblib'):
            target = ROOT / 'artifacts' / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))

print('Project:', ROOT)
print('Training and student data:', ARCHIVE)
"""),
    cell("code", """%cd /content/drive/MyDrive/PathSnap
%pip install -q -r requirements.txt
"""),
    cell("code", """import json, subprocess, sys
from pathlib import Path

config = json.loads((ROOT / 'configs/default.json').read_text(encoding='utf-8'))
config['data_archive'] = str(ARCHIVE)
config_path = Path('/content/pathsnap_decoder_config.json')
config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
command = [
    sys.executable, 'scripts/evaluate_decoder.py', '--config', str(config_path),
    '--metadata', 'artifacts/final_metadata.json',
    '--cache-path', 'results/candidate_cache.joblib',
    '--output', 'results/decoder_comparison.json',
]
print('Running grouped-CV decoder comparison and student decoder grid.')
subprocess.run(command, cwd=ROOT, check=True)
"""),
    cell("code", """import json
import pandas as pd
from IPython.display import display

result = json.loads((ROOT / 'results/decoder_comparison.json').read_text(encoding='utf-8'))
print('Model:', result['model'], result['model_parameters'])
print('CV summary:')
display(pd.DataFrame([result['cv_summary']['classifier_only'], result['cv_summary']['hmm_viterbi']], index=['classifier-only','HMM/Viterbi']))
print('Student-test decoder settings (used for development tuning):')
display(pd.DataFrame([{
  'setting': row['setting'], 'configuration': row['configuration'],
  'route_macro_f1': row['hmm_viterbi']['route_macro_f1'],
  'top1_accuracy': row['hmm_viterbi']['top1_accuracy'],
  'route_continuity': row['hmm_viterbi']['route_continuity'],
  'candidate_miss_rate': row['hmm_viterbi']['candidate_miss_rate']
} for row in result['student_grid']]))
print('Selected student setting:', result['selected_student_setting']['configuration'])
"""),
    cell("markdown", """## Return these outputs

Share the executed notebook and `results/decoder_comparison.json`. Keep the distinction between grouped-CV results and student-test decoder tuning: the selected student setting was chosen using student labels. Do not use evaluator routes to alter the model, search settings, or decoder."""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "10_hmm_viterbi_evaluation.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(OUT / "10_hmm_viterbi_evaluation.ipynb").write_text(json.dumps(notebook, indent=2), encoding="utf-8")
