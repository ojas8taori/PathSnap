"""Inference path shared by the API and local evaluation scripts."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .data import Trip, load_trip, trip_ids
from .decoder import viterbi_decode
from .experiments import _score
from .features import CandidateGenerator


class PathSnapEngine:
    def __init__(self, archive_path: str, artifact_path: str, partition: str = "evaluator_test"):
        self.archive_path = archive_path
        self.artifact_path = artifact_path
        self.partition = partition
        metadata_path = Path(artifact_path).with_name("final_metadata.json")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.model = joblib.load(artifact_path)
        self.generator = CandidateGenerator(**self.metadata["candidate_configuration"])
        self.hmm = self.metadata["hmm_configuration"]
        self._trip_ids = trip_ids(self.archive_path, self.partition)

    def available_trip_ids(self) -> list[str]:
        return list(self._trip_ids)

    def predict(self, trip_id: str) -> dict:
        with zipfile.ZipFile(self.archive_path) as archive:
            trip_with_reference = load_trip(archive, self.partition, trip_id)
        if trip_with_reference.route is None:
            raise ValueError(f"{trip_id} has no server-side reference route for scoring")
        # The estimator and decoder only receive this route-free object.
        trip = Trip(trip_id, trip_with_reference.partition, trip_with_reference.track, trip_with_reference.nodes, trip_with_reference.arcs, None)
        batch = self.generator.build(trip, with_labels=False)
        scores, _ = _score(self.model, batch.features)
        per_observation_scores = []
        cursor = 0
        classifier_top = []
        for ids in batch.candidate_arc_ids:
            end = cursor + len(ids)
            local_scores = np.asarray(scores[cursor:end], dtype=float)
            per_observation_scores.append(local_scores)
            classifier_top.append(int(ids[int(np.argmax(local_scores))]) if len(ids) else -1)
            cursor = end
        decoded = viterbi_decode(
            trip, batch.candidate_arc_ids, per_observation_scores,
            **self.hmm,
        )
        truth = self.generator._observation_route_labels(trip_with_reference, trip_with_reference.nodes[trip_with_reference.arcs])
        valid = truth >= 0
        candidate_hits = sum(int(int(ref) in ids) for ref, ids in zip(truth, batch.candidate_arc_ids) if ref >= 0)
        candidate_miss = 1.0 - candidate_hits / max(1, int(valid.sum()))
        point_accuracy = float(accuracy_score(truth[valid], np.asarray(decoded)[valid])) if valid.any() else None
        macro_f1 = float(f1_score(truth[valid], np.asarray(decoded)[valid], average="macro", zero_division=0)) if valid.any() else None
        path = []
        for arc_id in decoded:
            if arc_id < 0:
                path.append(None)
                continue
            u, v = trip.arcs[arc_id]
            path.append([[float(trip.nodes[u, 1]), float(trip.nodes[u, 0])], [float(trip.nodes[v, 1]), float(trip.nodes[v, 0])]])
        return {
            "trip_id": trip_id,
            "predicted_arc_ids": decoded,
            "classifier_top_arc_ids": classifier_top,
            "gps_points": [[float(lat), float(lon)] for lon, lat in trip.track[:, :2]],
            "predicted_path": path,
            "metrics": {"pointwise_accuracy": point_accuracy, "macro_f1": macro_f1, "candidate_miss_rate": candidate_miss},
        }
