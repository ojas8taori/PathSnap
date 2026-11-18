"""Grouped-CV and student-test comparison for the selected eight approaches."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SEED = 352
MODEL_CONFIGS = {
    "logistic_regression": {"family": "linear/probabilistic", "C": 1.0, "max_iter": 300, "class_weight": "balanced"},
    "distance_weighted_knn": {"family": "instance-based", "n_neighbors": 15, "weights": "distance", "p": 2, "train_row_cap": 30000},
    "random_forest": {"family": "ensemble", "n_estimators": 200, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced_subsample"},
    "extra_trees": {"family": "tree ensemble", "n_estimators": 250, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced"},
    "hist_gradient_boosting": {"family": "boosting", "learning_rate": 0.08, "max_iter": 150, "max_leaf_nodes": 31, "l2_regularization": 1.0},
    "rbf_svm": {"family": "kernel", "C": 1.0, "gamma": "scale", "probability": False, "train_row_cap": 10000},
    "mlp_shallow": {"family": "neural network", "hidden_layer_sizes": [64], "alpha": 0.001, "max_iter": 150, "early_stopping": True},
    "mlp_deep": {"family": "neural network", "hidden_layer_sizes": [128, 64, 32], "alpha": 0.001, "max_iter": 150, "early_stopping": True},
}


@dataclass
class TripRows:
    trip_id: str
    X: np.ndarray
    y: np.ndarray
    candidates: list[np.ndarray]
    reference_arc_ids: np.ndarray
    miss_rate: float
    arcs: np.ndarray
    track: np.ndarray
    nodes: np.ndarray


def make_model(name: str, config: dict | None = None):
    params = dict(MODEL_CONFIGS[name])
    params.update(config or {})
    if name == "logistic_regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=params["C"], max_iter=params["max_iter"], class_weight=params["class_weight"], random_state=SEED))
    if name == "distance_weighted_knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=params["n_neighbors"], weights=params["weights"], p=params["p"], n_jobs=-1))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=params["n_estimators"], min_samples_leaf=params["min_samples_leaf"], max_features=params["max_features"], class_weight=params["class_weight"], random_state=SEED, n_jobs=-1)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=params["n_estimators"], min_samples_leaf=params["min_samples_leaf"], max_features=params["max_features"], class_weight=params["class_weight"], random_state=SEED, n_jobs=-1)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(learning_rate=params["learning_rate"], max_iter=params["max_iter"], max_leaf_nodes=params["max_leaf_nodes"], l2_regularization=params["l2_regularization"], random_state=SEED)
    if name == "rbf_svm":
        return make_pipeline(StandardScaler(), SVC(C=params["C"], gamma=params["gamma"], probability=False, random_state=SEED))
    if name in {"mlp_shallow", "mlp_deep"}:
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=tuple(params["hidden_layer_sizes"]), alpha=params["alpha"], max_iter=params["max_iter"], early_stopping=params["early_stopping"], random_state=SEED))
    raise KeyError(name)


def _score(estimator, X: np.ndarray) -> tuple[np.ndarray, bool]:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1], True
    decision = np.asarray(estimator.decision_function(X)).reshape(-1)
    # Scores preserve order, but an uncalibrated sigmoid is not reported as log loss.
    return decision, False


def _training_history(estimator) -> dict | None:
    fitted = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    if not hasattr(fitted, "loss_curve_"):
        return None
    return {
        "training_loss": [float(value) for value in fitted.loss_curve_],
        "validation_scores": [float(value) for value in getattr(fitted, "validation_scores_", [])],
        "epochs_run": int(getattr(fitted, "n_iter_", len(fitted.loss_curve_))),
    }


def _sample_rows(X, y, rng, max_rows: int, negative_per_positive: int = 5):
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if len(positive) == 0:
        keep = rng.choice(negative, size=min(max_rows, len(negative)), replace=False)
    else:
        negatives = rng.choice(negative, size=min(len(negative), len(positive) * negative_per_positive), replace=False)
        keep = np.concatenate((positive, negatives))
        if len(keep) > max_rows:
            pos_count = min(len(positive), max(1, max_rows // (negative_per_positive + 1)))
            pos = rng.choice(positive, size=pos_count, replace=False)
            neg = rng.choice(negative, size=max_rows - len(pos), replace=False)
            keep = np.concatenate((pos, neg))
    rng.shuffle(keep)
    return X[keep], y[keep]


def _classify_trip(estimator, record: TripRows) -> dict:
    scores, has_proba = _score(estimator, record.X)
    predicted = (scores >= 0.5).astype(np.int8) if has_proba else (scores >= 0).astype(np.int8)
    cursor = 0
    top_ids = []
    for ids in record.candidates:
        end = cursor + len(ids)
        local = scores[cursor:end]
        top_ids.append(int(ids[int(np.argmax(local))]) if len(ids) else -1)
        cursor = end
    top_ids = np.asarray(top_ids, dtype=np.int64)
    valid = record.reference_arc_ids >= 0
    top1 = float(np.mean(top_ids[valid] == record.reference_arc_ids[valid])) if valid.any() else float("nan")
    route_macro_f1 = float(f1_score(record.reference_arc_ids[valid], top_ids[valid], average="macro", zero_division=0)) if valid.any() else float("nan")
    sequence_exact = float(np.all(top_ids[valid] == record.reference_arc_ids[valid])) if valid.any() else float("nan")
    adjacent = (top_ids[:-1] >= 0) & (top_ids[1:] >= 0)
    if adjacent.any():
        previous, following = top_ids[:-1][adjacent], top_ids[1:][adjacent]
        continuity = float(np.mean((previous == following) | (record.arcs[previous, 1] == record.arcs[following, 0])))
    else:
        continuity = float("nan")
    candidate_y = record.y
    return {
        "candidate_accuracy": float(accuracy_score(candidate_y, predicted)) if len(candidate_y) else float("nan"),
        "candidate_macro_f1": float(f1_score(candidate_y, predicted, average="macro", zero_division=0)) if len(candidate_y) else float("nan"),
        "candidate_precision": float(precision_score(candidate_y, predicted, zero_division=0)) if len(candidate_y) else float("nan"),
        "candidate_recall": float(recall_score(candidate_y, predicted, zero_division=0)) if len(candidate_y) else float("nan"),
        "top1_accuracy": top1,
        "route_macro_f1": route_macro_f1,
        "sequence_exact": sequence_exact,
        "route_continuity": continuity,
        "candidate_miss_rate": record.miss_rate,
        "log_loss": float(log_loss(candidate_y, np.column_stack((1 - np.clip(scores, 1e-7, 1 - 1e-7), np.clip(scores, 1e-7, 1 - 1e-7))))) if has_proba and len(np.unique(candidate_y)) > 1 else None,
        "scores": scores.tolist(), "predicted_arc_ids": top_ids.tolist(),
        "log_loss_applicable": has_proba,
    }


def evaluate_model(name: str, records: list[TripRows], folds: int = 5, max_train_rows: int = 120000, model_config: dict | None = None) -> dict:
    groups = np.asarray([r.trip_id for r in records])
    splitter = GroupKFold(n_splits=folds)
    fold_results = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(records)), groups=groups), start=1):
        train_records = [records[i] for i in train_idx]
        valid_records = [records[i] for i in valid_idx]
        X_train = np.concatenate([r.X for r in train_records])
        y_train = np.concatenate([r.y for r in train_records])
        rng = np.random.default_rng(SEED + fold)
        X_train, y_train = _sample_rows(X_train, y_train, rng, max_train_rows)
        model = make_model(name, model_config)
        if name in ("distance_weighted_knn", "rbf_svm"):
            cap = MODEL_CONFIGS[name].get("train_row_cap", 30000)
            X_train, y_train = _sample_rows(X_train, y_train, rng, cap)
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started
        history = _training_history(model)
        per_trip = []
        infer_seconds = 0.0
        for record in valid_records:
            start = time.perf_counter()
            metrics = _classify_trip(model, record)
            infer_seconds += time.perf_counter() - start
            per_trip.append(metrics)
        fold_results.append({
            "fold": fold,
            "train_trip_ids": [r.trip_id for r in train_records],
            "validation_trip_ids": [r.trip_id for r in valid_records],
            "fit_seconds": fit_seconds, "inference_seconds": infer_seconds,
            "training_history": history,
            "candidate_macro_f1": float(np.nanmean([m["candidate_macro_f1"] for m in per_trip])),
            "top1_accuracy": float(np.nanmean([m["top1_accuracy"] for m in per_trip])),
            "route_macro_f1": float(np.nanmean([m["route_macro_f1"] for m in per_trip])),
            "sequence_exact": float(np.nanmean([m["sequence_exact"] for m in per_trip])),
            "candidate_miss_rate": float(np.nanmean([m["candidate_miss_rate"] for m in per_trip])),
            "route_continuity": float(np.nanmean([m["route_continuity"] for m in per_trip])),
            "log_loss": float(np.nanmean([m["log_loss"] for m in per_trip if m["log_loss"] is not None])) if any(m["log_loss"] is not None for m in per_trip) else None,
        })
    configuration = dict(MODEL_CONFIGS[name])
    configuration.update(model_config or {})
    return {"model": name, "configuration": configuration, "folds": fold_results}


def fit_and_evaluate_student(name: str, train_records: list[TripRows], student_records: list[TripRows], max_train_rows: int = 160000, model_config: dict | None = None) -> tuple[dict, object]:
    X = np.concatenate([r.X for r in train_records])
    y = np.concatenate([r.y for r in train_records])
    rng = np.random.default_rng(SEED)
    X, y = _sample_rows(X, y, rng, max_train_rows)
    if name in ("distance_weighted_knn", "rbf_svm"):
        X, y = _sample_rows(X, y, rng, MODEL_CONFIGS[name].get("train_row_cap", 30000))
    model = make_model(name, model_config)
    started = time.perf_counter()
    model.fit(X, y)
    fit_seconds = time.perf_counter() - started
    history = _training_history(model)
    metrics = []
    predictions = {}
    infer_seconds = 0.0
    for record in student_records:
        start = time.perf_counter()
        result = _classify_trip(model, record)
        infer_seconds += time.perf_counter() - start
        metrics.append(result)
        candidate_scores = []
        score_cursor = 0
        for ids in record.candidates:
            next_cursor = score_cursor + len(ids)
            candidate_scores.append(result["scores"][score_cursor:next_cursor])
            score_cursor = next_cursor
        predictions[record.trip_id] = {
            "predicted_arc_ids": result["predicted_arc_ids"],
            "reference_arc_ids": record.reference_arc_ids.tolist(),
            "candidate_miss_rate": record.miss_rate,
            "candidate_arc_ids": [ids.tolist() for ids in record.candidates],
            "candidate_scores": candidate_scores,
        }
    summary = {
        "model": name,
        "fit_seconds": fit_seconds,
        "inference_seconds": infer_seconds,
        "candidate_macro_f1": float(np.nanmean([m["candidate_macro_f1"] for m in metrics])),
        "candidate_precision": float(np.nanmean([m["candidate_precision"] for m in metrics])),
        "candidate_recall": float(np.nanmean([m["candidate_recall"] for m in metrics])),
        "top1_accuracy": float(np.nanmean([m["top1_accuracy"] for m in metrics])),
        "route_macro_f1": float(np.nanmean([m["route_macro_f1"] for m in metrics])),
        "sequence_exact": float(np.nanmean([m["sequence_exact"] for m in metrics])),
        "candidate_miss_rate": float(np.nanmean([m["candidate_miss_rate"] for m in metrics])),
        "route_continuity": float(np.nanmean([m["route_continuity"] for m in metrics])),
        "log_loss": float(np.nanmean([m["log_loss"] for m in metrics if m["log_loss"] is not None])) if any(m["log_loss"] is not None for m in metrics) else None,
        "log_loss_applicable": all(m["log_loss_applicable"] for m in metrics),
    }
    return {"summary": summary, "per_trip": metrics, "predictions": predictions, "training_history": history}, model
