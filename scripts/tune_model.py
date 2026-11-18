"""Bayesian tuning of the best measured initial model using grouped CV."""

from __future__ import annotations

import sys
from pathlib import Path as _RootPath
sys.path.insert(0, str(_RootPath(__file__).resolve().parents[1]))

import argparse
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from run_experiments import prepare
from src.experiments import MODEL_CONFIGS, evaluate_model, fit_and_evaluate_student
from src.features import CandidateGenerator, FEATURE_NAMES


def proposal(name: str, x: np.ndarray) -> dict:
    if name == "logistic_regression":
        return {"C": float(10 ** (-3 + 5 * x[0]))}
    if name == "distance_weighted_knn":
        return {"n_neighbors": int(5 + round(25 * x[0])), "weights": "distance" if x[1] > .5 else "uniform"}
    if name in {"random_forest", "extra_trees"}:
        return {"n_estimators": int(100 + round(300 * x[0])), "min_samples_leaf": int(1 + round(7 * x[1]))}
    if name == "hist_gradient_boosting":
        return {"learning_rate": float(.02 + .18 * x[0]), "max_iter": int(50 + round(200 * x[1])), "max_leaf_nodes": int(15 + round(48 * x[2]))}
    if name == "rbf_svm":
        return {"C": float(10 ** (-2 + 4 * x[0]))}
    if name == "mlp_shallow":
        return {"alpha": float(10 ** (-5 + 4 * x[0])), "hidden_layer_sizes": [int(32 + round(96 * x[1]))]}
    if name == "mlp_deep":
        return {"alpha": float(10 ** (-5 + 4 * x[0])), "hidden_layer_sizes": [int(64 + round(128 * x[1])), 64, 32]}
    raise KeyError(name)


def dimensions(name: str) -> int:
    return {"logistic_regression": 1, "distance_weighted_knn": 2, "random_forest": 2, "extra_trees": 2,
            "hist_gradient_boosting": 3, "rbf_svm": 1, "mlp_shallow": 2, "mlp_deep": 2}[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--initial-results", default="results/experiment_results.json")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    initial = json.loads(Path(args.initial_results).read_text(encoding="utf-8"))
    selected = max(initial["models"], key=lambda name: np.mean([fold["route_macro_f1"] for fold in initial["models"][name]["cv"]["folds"]]))
    generator = CandidateGenerator(config["candidate_radius_m"], config["candidate_top_k"])
    train = prepare(config["data_archive"], config["train_partition"], generator)
    student = prepare(config["data_archive"], config["development_partition"], generator)
    dim = dimensions(selected)
    rng = np.random.default_rng(int(config["random_seed"]))
    observed_x, observed_y, trials = [], [], []
    start = time.perf_counter()
    for trial_id in range(args.trials):
        if trial_id < min(3, args.trials) or len(observed_x) < 3:
            candidate = rng.uniform(0, 1, dim)
        else:
            pool = rng.uniform(0, 1, (256, dim))
            gp = GaussianProcessRegressor(kernel=Matern(nu=2.5) + WhiteKernel(1e-4), normalize_y=True, random_state=int(config["random_seed"]) + trial_id, n_restarts_optimizer=1)
            gp.fit(np.asarray(observed_x), np.asarray(observed_y))
            mean, std = gp.predict(pool, return_std=True)
            improvement = mean - max(observed_y) - 0.01
            z = improvement / np.maximum(std, 1e-9)
            acquisition = improvement * norm.cdf(z) + std * norm.pdf(z)
            candidate = pool[int(np.argmax(acquisition))]
        params = proposal(selected, candidate)
        result = evaluate_model(selected, train, folds=int(config["cv_folds"]), model_config=params)
        fold_scores = [fold["route_macro_f1"] for fold in result["folds"]]
        objective = float(np.nanmean(fold_scores))
        observed_x.append(candidate.tolist())
        observed_y.append(objective)
        trials.append({"trial": trial_id + 1, "parameters": params, "cv_route_macro_f1": objective, "folds": result["folds"]})
    best_idx = int(np.argmax(observed_y))
    best_params = trials[best_idx]["parameters"]
    student_output, model = fit_and_evaluate_student(selected, train, student, model_config=best_params)
    best_cv = evaluate_model(selected, train, folds=int(config["cv_folds"]), model_config=best_params)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result_doc = {
        "selected_model": selected, "search_method": "Gaussian-process expected improvement over normalized continuous proposal space",
        "objective": "mean trip-grouped CV route Macro F1", "trial_budget": args.trials,
        "compute_seconds": time.perf_counter() - start,
        "seed": config["random_seed"], "trials": trials,
        "best_parameters": best_params, "best_objective": observed_y[best_idx],
        "tuned_cv": best_cv, "tuned_student_test": student_output["summary"],
        "baseline_cv": initial["models"][selected]["cv"], "baseline_student_test": initial["models"][selected]["student_test"],
    }
    (Path(args.output_dir) / "bayesian_tuning.json").write_text(json.dumps(result_doc, indent=2), encoding="utf-8")
    model_path = Path("artifacts/final_model.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata = {
        "selected_model": selected, "selected_parameters": best_params,
        "feature_names": FEATURE_NAMES,
        "candidate_configuration": {"radius_m": float(config["candidate_radius_m"]), "top_k": int(config["candidate_top_k"])},
        "class_mapping": {"0": "candidate arc does not equal the nearest arc on this trip's reference route", "1": "candidate arc equals the nearest arc on this trip's reference route"},
        "hmm_configuration": {"max_gap_seconds": 120.0, "max_speed_mps": 45.0, "sigma_m": 80.0, "disconnected_log_penalty": -12.0},
        "seed": config["random_seed"], "python_version": __import__("platform").python_version(), "scikit_learn_version": sklearn.__version__,
    }
    Path("artifacts/final_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (Path(args.output_dir) / "student_predictions_tuned.json").write_text(json.dumps(student_output["predictions"], separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
