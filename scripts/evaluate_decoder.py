"""Compare classifier-only routes with HMM/Viterbi on CV and student trips."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold

from run_experiments import prepare
from src.data import Trip
from src.decoder import viterbi_decode
from src.experiments import MODEL_CONFIGS, _sample_rows, _score, make_model


def route_metrics(record, predicted: list[int]) -> dict:
    prediction = np.asarray(predicted, dtype=np.int64)
    reference = record.reference_arc_ids
    valid = reference >= 0
    pointwise = float(accuracy_score(reference[valid], prediction[valid])) if valid.any() else float("nan")
    macro_f1 = float(f1_score(reference[valid], prediction[valid], average="macro", zero_division=0)) if valid.any() else float("nan")
    exact = float(np.all(reference[valid] == prediction[valid])) if valid.any() else float("nan")
    pair_ok = (prediction[:-1] >= 0) & (prediction[1:] >= 0)
    if pair_ok.any():
        a, b = prediction[:-1][pair_ok], prediction[1:][pair_ok]
        continuity = float(np.mean((a == b) | (record.arcs[a, 1] == record.arcs[b, 0])))
    else:
        continuity = float("nan")
    return {
        "top1_accuracy": pointwise,
        "route_macro_f1": macro_f1,
        "sequence_exact": exact,
        "route_continuity": continuity,
        "candidate_miss_rate": record.miss_rate,
    }


def score_record(model, record, hmm_config: dict) -> dict:
    scores, calibrated = _score(model, record.X)
    cursor = 0
    per_observation_scores = []
    classifier_ids = []
    for ids in record.candidates:
        end = cursor + len(ids)
        local = np.asarray(scores[cursor:end], dtype=float)
        per_observation_scores.append(local)
        classifier_ids.append(int(ids[int(np.argmax(local))]) if len(ids) else -1)
        cursor = end
    trip = Trip(record.trip_id, "inference", record.track, record.nodes, record.arcs, None)
    decoded = viterbi_decode(trip, record.candidates, per_observation_scores, **hmm_config)
    return {
        "classifier": route_metrics(record, classifier_ids),
        "hmm": route_metrics(record, decoded),
        "classifier_arc_ids": classifier_ids,
        "hmm_arc_ids": decoded,
        "probability_scores": bool(calibrated),
    }


def aggregate(rows: list[dict], key: str) -> dict:
    out = {}
    metric_names = ["top1_accuracy", "route_macro_f1", "sequence_exact", "route_continuity", "candidate_miss_rate"]
    for metric in metric_names:
        values = [r[key][metric] for r in rows if np.isfinite(r[key][metric])]
        out[metric] = float(np.mean(values)) if values else None
    return out


def fit_data(records, cap: int, seed: int):
    X = np.concatenate([record.X for record in records])
    y = np.concatenate([record.y for record in records])
    return _sample_rows(X, y, np.random.default_rng(seed), cap)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--metadata", default="artifacts/final_metadata.json")
    parser.add_argument("--cache-path", default="results/candidate_cache.joblib")
    parser.add_argument("--output", default="results/decoder_comparison.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    model_name = metadata["selected_model"]
    model_params = metadata["selected_parameters"]
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported selected model: {model_name}")
    cache_path = Path(args.cache_path)
    if cache_path.exists():
        cache = joblib.load(cache_path)
        train_records, student_records = cache["train"], cache["student_test"]
    else:
        from src.features import CandidateGenerator
        generator = CandidateGenerator(config["candidate_radius_m"], config["candidate_top_k"])
        train_records = prepare(config["data_archive"], config["train_partition"], generator)
        student_records = prepare(config["data_archive"], config["development_partition"], generator)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"train": train_records, "student_test": student_records, "preparation_seconds": None}, cache_path, compress=3)

    n_splits = int(config["cv_folds"])
    groups = np.asarray([r.trip_id for r in train_records])
    splitter = GroupKFold(n_splits=n_splits)
    cv_folds = []
    seed = int(config["random_seed"])
    base_hmm = metadata["hmm_configuration"]
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(train_records)), groups=groups), 1):
        fit_records = [train_records[i] for i in train_idx]
        valid_records = [train_records[i] for i in valid_idx]
        X, y = fit_data(fit_records, 120000, seed + fold)
        estimator = make_model(model_name, model_params)
        fit_start = time.perf_counter()
        estimator.fit(X, y)
        fit_seconds = time.perf_counter() - fit_start
        scored = []
        decode_seconds = 0.0
        print(f"CV fold {fold}/{n_splits}: fitting on {len(fit_records)} trips; decoding {len(valid_records)} trips", flush=True)
        for record in valid_records:
            tick = time.perf_counter()
            scored.append(score_record(estimator, record, base_hmm))
            decode_seconds += time.perf_counter() - tick
        cv_folds.append({
            "fold": fold,
            "train_trip_ids": [r.trip_id for r in fit_records],
            "validation_trip_ids": [r.trip_id for r in valid_records],
            "fit_seconds": fit_seconds,
            "inference_and_decode_seconds": decode_seconds,
            "classifier_only": aggregate(scored, "classifier"),
            "hmm_viterbi": aggregate(scored, "hmm"),
        })

    X, y = fit_data(train_records, 160000, seed)
    final_estimator = make_model(model_name, model_params)
    fit_start = time.perf_counter()
    final_estimator.fit(X, y)
    student_fit_seconds = time.perf_counter() - fit_start
    baseline = dict(base_hmm)
    parameter_grid = [
        baseline,
        {"max_gap_seconds": 60.0, "max_speed_mps": 35.0, "sigma_m": 40.0, "disconnected_log_penalty": -12.0},
        {"max_gap_seconds": 300.0, "max_speed_mps": 60.0, "sigma_m": 160.0, "disconnected_log_penalty": -12.0},
        {"max_gap_seconds": 120.0, "max_speed_mps": 45.0, "sigma_m": 80.0, "disconnected_log_penalty": -20.0},
    ]
    student_grid = []
    student_prediction_cache = []
    # Re-score the student set once and decode with a small documented parameter grid.
    for record_index, record in enumerate(student_records, 1):
        print(f"Scoring student-test trip {record_index}/{len(student_records)}: {record.trip_id}", flush=True)
        scores, calibrated = _score(final_estimator, record.X)
        cursor = 0
        per_observation_scores = []
        classifier_ids = []
        for ids in record.candidates:
            end = cursor + len(ids)
            local = np.asarray(scores[cursor:end], dtype=float)
            per_observation_scores.append(local)
            classifier_ids.append(int(ids[int(np.argmax(local))]) if len(ids) else -1)
            cursor = end
        trip = Trip(record.trip_id, "student_test", record.track, record.nodes, record.arcs, None)
        decoded_by_setting = []
        for hmm_cfg in parameter_grid:
            decoded = viterbi_decode(trip, record.candidates, per_observation_scores, **hmm_cfg)
            decoded_by_setting.append(decoded)
        student_prediction_cache.append((record, classifier_ids, decoded_by_setting, bool(calibrated)))
    for setting_id, hmm_cfg in enumerate(parameter_grid, 1):
        rows = []
        for record, classifier_ids, decoded_by_setting, calibrated in student_prediction_cache:
            rows.append({"trip_id": record.trip_id, "classifier": route_metrics(record, classifier_ids), "hmm": route_metrics(record, decoded_by_setting[setting_id - 1])})
        student_grid.append({"setting": setting_id, "configuration": hmm_cfg, "classifier_only": aggregate(rows, "classifier"), "hmm_viterbi": aggregate(rows, "hmm"), "per_trip": rows})
    selected_setting = max(student_grid, key=lambda row: (row["hmm_viterbi"]["route_macro_f1"], row["hmm_viterbi"]["route_continuity"]))
    selected_predictions = {}
    for record, classifier_ids, decoded_by_setting, calibrated in student_prediction_cache:
        decoded = decoded_by_setting[selected_setting["setting"] - 1]
        selected_predictions[record.trip_id] = {
            "classifier_arc_ids": classifier_ids,
            "hmm_arc_ids": decoded,
            "reference_arc_ids": record.reference_arc_ids.tolist(),
            "candidate_miss_rate": record.miss_rate,
        }
    result = {
        "model": model_name,
        "model_parameters": model_params,
        "partition_provenance": {"grouped_cv": config["train_partition"], "decoder_parameter_development": config["development_partition"], "evaluator": "not used"},
        "selection_rule": "Among the small documented decoder grid, select the setting with highest mean student-test route Macro F1; continuity breaks ties. Treat this as development-tuned, not an independent test score.",
        "hmm_base_configuration": base_hmm,
        "cv_folds": cv_folds,
        "cv_summary": {
            "classifier_only": {metric: float(np.mean([f["classifier_only"][metric] for f in cv_folds])) for metric in cv_folds[0]["classifier_only"]},
            "hmm_viterbi": {metric: float(np.mean([f["hmm_viterbi"][metric] for f in cv_folds])) for metric in cv_folds[0]["hmm_viterbi"]},
        },
        "student_fit_seconds": student_fit_seconds,
        "student_grid": student_grid,
        "selected_student_setting": selected_setting,
        "student_predictions_selected": selected_predictions,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved decoder comparison to {output_path}")


if __name__ == "__main__":
    main()
