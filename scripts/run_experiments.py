"""Generate trip-grouped CV and student-test results for eight models."""

from __future__ import annotations

import sys
from pathlib import Path as _RootPath
sys.path.insert(0, str(_RootPath(__file__).resolve().parents[1]))

import argparse
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import sklearn
import joblib

from src.data import load_trip, trip_ids
from src.experiments import MODEL_CONFIGS, TripRows, evaluate_model, fit_and_evaluate_student
from src.features import CandidateGenerator


def prepare(archive_path: str, partition: str, generator: CandidateGenerator) -> list[TripRows]:
    out = []
    with zipfile.ZipFile(archive_path) as archive:
        ids = trip_ids(archive_path, partition)
        for index, trip_id in enumerate(ids, start=1):
            print(f"Preparing {partition} trip {index}/{len(ids)}: {trip_id}", flush=True)
            trip = load_trip(archive, partition, trip_id)
            batch = generator.build(trip, with_labels=True)
            if batch.labels is None or trip.route is None:
                raise ValueError(f"{partition}/{trip_id}: supervised comparison requires reference route labels")
            ref = batch.reference_arc_ids
            out.append(TripRows(trip_id, batch.features, batch.labels, batch.candidate_arc_ids, ref, float(batch.candidate_miss_rate or 0.0), trip.arcs, trip.track, trip.nodes))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model", choices=list(MODEL_CONFIGS), help="Run one approach; omit to run the eight-model suite")
    parser.add_argument("--cache-path", default="results/candidate_cache.joblib")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    archive_path = config["data_archive"]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    generator = CandidateGenerator(config["candidate_radius_m"], config["candidate_top_k"])
    cache_path = Path(args.cache_path)
    if cache_path.exists() and not args.rebuild_cache:
        cache = joblib.load(cache_path)
        train, student = cache["train"], cache["student_test"]
        preparation_seconds = cache["preparation_seconds"]
    else:
        started = time.perf_counter()
        train = prepare(archive_path, config["train_partition"], generator)
        student = prepare(archive_path, config["development_partition"], generator)
        preparation_seconds = time.perf_counter() - started
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"train": train, "student_test": student, "preparation_seconds": preparation_seconds}, cache_path, compress=3)
    output = {
        "seed": config["random_seed"], "cv_folds": config["cv_folds"],
        "partition_provenance": {"fit_cv": config["train_partition"], "development_comparison": config["development_partition"], "evaluator": "not supplied"},
        "candidate_rule": {"search": "top-k nearest spherical arc midpoints, exact local segment ranking", "radius_m": generator.radius_m, "top_k": generator.top_k},
        "label_rule": "For each GPS observation, the nearest arc in that trip's supplied ordered reference route is the positive candidate label. Route and arc IDs remain trip-local.",
        "preparation_seconds": preparation_seconds,
        "python_version": platform.python_version(), "scikit_learn_version": sklearn.__version__, "numpy_version": np.__version__,
        "models": {},
    }
    selected_models = [args.model] if args.model else list(MODEL_CONFIGS)
    for name in selected_models:
        cv = evaluate_model(name, train, folds=int(config["cv_folds"]))
        student_results, fitted = fit_and_evaluate_student(name, train, student)
        output["models"][name] = {"configuration": MODEL_CONFIGS[name], "cv": cv, "student_test": student_results["summary"], "student_per_trip": student_results["per_trip"], "training_history": {"student_fit": student_results["training_history"], "cv_folds": [fold["training_history"] for fold in cv["folds"]]}}
        (Path(args.output_dir) / f"student_predictions_{name}.json").write_text(json.dumps(student_results["predictions"], separators=(",", ":")), encoding="utf-8")
        del fitted
    (Path(args.output_dir) / "experiment_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
