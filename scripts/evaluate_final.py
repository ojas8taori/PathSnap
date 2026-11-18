"""Score the frozen HGB + HMM pipeline on the reserved evaluator package.

The model is called with route-free trips through PathSnapEngine. Reference
routes are loaded only afterward for this script's server-side scoring.
This script does not fit a model or change decoder settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score

from src.data import load_trip
from src.inference import PathSnapEngine


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def route_metrics(trip, truth: np.ndarray, prediction: list[int]) -> dict:
    pred = np.asarray(prediction, dtype=np.int64)
    valid = truth >= 0
    adjacent = (pred[:-1] >= 0) & (pred[1:] >= 0)
    if adjacent.any():
        a, b = pred[:-1][adjacent], pred[1:][adjacent]
        continuity = float(np.mean((a == b) | (trip.arcs[a, 1] == trip.arcs[b, 0])))
    else:
        continuity = None
    return {
        "top1_accuracy": float(accuracy_score(truth[valid], pred[valid])) if valid.any() else None,
        "route_macro_f1": float(f1_score(truth[valid], pred[valid], average="macro", zero_division=0)) if valid.any() else None,
        "sequence_exact": bool(np.all(truth[valid] == pred[valid])) if valid.any() else None,
        "route_continuity": continuity,
        "candidate_miss_rate": float(np.mean([int(int(ref) not in ids) for ref, ids in zip(truth, trip._candidate_ids) if ref >= 0])) if valid.any() else None,
        "scored_observations": int(valid.sum()),
    }


def mean_metrics(rows: list[dict]) -> dict:
    names = ("top1_accuracy", "route_macro_f1", "sequence_exact", "route_continuity", "candidate_miss_rate")
    result = {}
    for name in names:
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        result[name] = float(np.mean(values)) if values else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluator_archive")
    parser.add_argument("--artifact", default="artifacts/final_model.joblib")
    parser.add_argument("--partition", default="evaluator_test")
    parser.add_argument("--output", default="results/final_evaluator.json")
    args = parser.parse_args()

    engine = PathSnapEngine(args.evaluator_archive, args.artifact, args.partition)
    started = time.perf_counter()
    rows = []
    with zipfile.ZipFile(args.evaluator_archive) as archive:
        for trip_id in engine.available_trip_ids():
            trip_with_reference = load_trip(archive, args.partition, trip_id)
            # PathSnapEngine strips the route before feature generation, decoding,
            # and prediction. It uses labels only for a separate returned score.
            prediction = engine.predict(trip_id)
            truth = engine.generator._observation_route_labels(
                trip_with_reference,
                trip_with_reference.nodes[trip_with_reference.arcs],
            )
            batch = engine.generator.build(
                type(trip_with_reference)(trip_id, args.partition, trip_with_reference.track,
                                          trip_with_reference.nodes, trip_with_reference.arcs, None),
                with_labels=False,
            )
            trip_with_reference._candidate_ids = batch.candidate_arc_ids
            classifier = route_metrics(trip_with_reference, truth, prediction["classifier_top_arc_ids"])
            hmm = route_metrics(trip_with_reference, truth, prediction["predicted_arc_ids"])
            rows.append({
                "trip_id": trip_id,
                "observations": int(len(trip_with_reference.track)),
                "classifier_only": classifier,
                "hmm_viterbi": hmm,
            })

    output = {
        "evaluation": "reserved evaluator package; final frozen pipeline only",
        "partition": args.partition,
        "trip_count": len(rows),
        "model": engine.metadata["selected_model"],
        "model_parameters": engine.metadata["selected_parameters"],
        "candidate_configuration": engine.metadata["candidate_configuration"],
        "hmm_configuration": engine.metadata["hmm_configuration"],
        "scikit_learn_version": sklearn.__version__,
        "artifact_sha256": file_sha256(args.artifact),
        "evaluator_archive_sha256": file_sha256(args.evaluator_archive),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "means_across_trips": {
            "classifier_only": mean_metrics([row["classifier_only"] for row in rows]),
            "hmm_viterbi": mean_metrics([row["hmm_viterbi"] for row in rows]),
        },
        "per_trip": rows,
        "scoring_note": "Predictions were generated from route-free trip inputs. Reference routes were read separately for server-side scoring after inference. Evaluator results were not used to select or modify model/decoder settings.",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "trip_count": len(rows), "means_across_trips": output["means_across_trips"]}, indent=2))


if __name__ == "__main__":
    main()
