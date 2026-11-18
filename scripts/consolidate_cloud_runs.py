"""Merge the eight executed notebook outputs into one evidence table and plots."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/matplotlib-config").resolve()))
Path("results/matplotlib-config").mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.experiments import MODEL_CONFIGS


def mean_std(folds: list[dict], key: str) -> tuple[float | None, float | None]:
    values = [float(row[key]) for row in folds if row.get(key) is not None and np.isfinite(row[key])]
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()
    root = Path(args.results_root)
    combined = {"models": {}, "source": "Executed cloud notebook outputs; no values are synthesized."}
    rows = []
    missing = []
    for name, config in MODEL_CONFIGS.items():
        path = root / f"cloud_{name}" / "experiment_results.json"
        if not path.exists():
            missing.append(name)
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        if name not in run.get("models", {}):
            raise ValueError(f"{path} does not contain {name}")
        model = run["models"][name]
        combined["models"][name] = model
        folds = model["cv"]["folds"]
        cv_f1, cv_f1_std = mean_std(folds, "route_macro_f1")
        cv_top1, cv_top1_std = mean_std(folds, "top1_accuracy")
        cv_cont, cv_cont_std = mean_std(folds, "route_continuity")
        cv_miss, _ = mean_std(folds, "candidate_miss_rate")
        cv_fit, _ = mean_std(folds, "fit_seconds")
        cv_infer, _ = mean_std(folds, "inference_seconds")
        student = model["student_test"]
        rows.append({
            "model": name, "family": config["family"], "configuration": json.dumps(model["configuration"], sort_keys=True),
            "cv_route_macro_f1_mean": cv_f1, "cv_route_macro_f1_std": cv_f1_std,
            "cv_top1_accuracy_mean": cv_top1, "cv_top1_accuracy_std": cv_top1_std,
            "cv_route_continuity_mean": cv_cont, "cv_route_continuity_std": cv_cont_std,
            "cv_candidate_miss_rate": cv_miss,
            "student_route_macro_f1": student["route_macro_f1"], "student_top1_accuracy": student["top1_accuracy"],
            "student_route_continuity": student["route_continuity"], "student_candidate_miss_rate": student["candidate_miss_rate"],
            "student_candidate_macro_f1": student["candidate_macro_f1"],
            "student_candidate_log_loss": student["log_loss"],
            "student_fit_seconds": student["fit_seconds"], "student_inference_seconds": student["inference_seconds"],
        })
    if missing:
        raise FileNotFoundError("Executed outputs are missing for: " + ", ".join(missing))
    combined["model_order"] = list(MODEL_CONFIGS)
    (root / "experiment_results.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    import csv
    with (root / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = [row["model"].replace("_", "\n") for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].bar(labels, [row["cv_route_macro_f1_mean"] for row in rows], yerr=[row["cv_route_macro_f1_std"] for row in rows], color="#286b9c", capsize=3)
    axes[0].set_title("Training CV route Macro F1")
    axes[1].bar(labels, [row["student_top1_accuracy"] for row in rows], color="#29945e")
    axes[1].set_title("Student-test Top-1 accuracy")
    axes[2].bar(labels, [row["student_route_macro_f1"] for row in rows], color="#c35a3c")
    axes[2].set_title("Student-test route Macro F1")
    for axis in axes:
        axis.tick_params(axis="x", labelsize=8)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(root / "model_comparison.png", dpi=170)
    plt.close(fig)
    print(f"Consolidated {len(rows)} actual model runs into {root}")


if __name__ == "__main__":
    main()
