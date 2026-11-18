"""Build report plots from saved PathSnap experiment outputs."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results/matplotlib-config").resolve()))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
NAVY = "#17324D"
BLUE = "#2474A6"
GREEN = "#29945E"
RED = "#C23C32"
GOLD = "#D89B24"


def main() -> None:
    rows = list(csv.DictReader((ROOT / "results/model_comparison.csv").open(encoding="utf-8-sig", newline="")))
    tuning = json.loads((ROOT / "results/bayesian_tuning.json").read_text(encoding="utf-8"))
    experiments = json.loads((ROOT / "results/experiment_results.json").read_text(encoding="utf-8"))
    decoder = json.loads((ROOT / "results/decoder_comparison.json").read_text(encoding="utf-8"))
    final = json.loads((ROOT / "results/final_evaluator.json").read_text(encoding="utf-8"))

    tuned_folds = tuning["tuned_cv"]["folds"]
    tuned = {
        "cv_route_macro_f1_mean": float(np.mean([r["route_macro_f1"] for r in tuned_folds])),
        "cv_route_macro_f1_std": float(np.std([r["route_macro_f1"] for r in tuned_folds], ddof=1)),
        "cv_top1_accuracy_mean": float(np.mean([r["top1_accuracy"] for r in tuned_folds])),
        "cv_route_continuity_mean": float(np.mean([r["route_continuity"] for r in tuned_folds])),
        "student_top1_accuracy": tuning["tuned_student_test"]["top1_accuracy"],
        "student_route_macro_f1": tuning["tuned_student_test"]["route_macro_f1"],
        "student_route_continuity": tuning["tuned_student_test"]["route_continuity"],
        "student_candidate_log_loss": tuning["tuned_student_test"]["log_loss"],
        "student_inference_seconds": tuning["tuned_student_test"]["inference_seconds"],
    }
    for row in rows:
        if row["model"] == "hist_gradient_boosting":
            row.update(tuned)
            row["model"] = "hgb_tuned"
            break

    names = [r["model"] for r in rows]
    labels = ["Logistic\nRegression", "Weighted\nKNN", "Random\nForest", "Extra\nTrees", "HGB\n(tuned)", "RBF\nSVM", "Shallow\nMLP", "Deep\nMLP"]
    cv = np.array([float(r["cv_route_macro_f1_mean"]) for r in rows])
    cv_std = np.array([float(r["cv_route_macro_f1_std"]) for r in rows])
    top1 = np.array([float(r["student_top1_accuracy"]) for r in rows])
    route_f1 = np.array([float(r["student_route_macro_f1"]) for r in rows])
    logloss = np.array([float(r["student_candidate_log_loss"]) if r["student_candidate_log_loss"] else np.nan for r in rows])
    inference = np.array([float(r["student_inference_seconds"]) for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), dpi=180)
    x = np.arange(len(rows))
    axes[0, 0].bar(x, cv, yerr=cv_std, color=BLUE, capsize=3)
    axes[0, 0].set_title("Five-fold grouped CV route Macro F1")
    axes[0, 0].set_ylim(0.52, 0.69)
    axes[0, 1].bar(x - 0.18, top1, 0.36, label="Top-1 accuracy", color=GREEN)
    axes[0, 1].bar(x + 0.18, route_f1, 0.36, label="Route Macro F1", color=RED)
    axes[0, 1].set_title("Student-test route performance")
    axes[0, 1].set_ylim(0.55, 0.90)
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].bar(x, logloss, color=GOLD)
    axes[1, 0].set_title("Student-test candidate log loss")
    axes[1, 0].set_ylim(0, 0.37)
    axes[1, 0].text(5, 0.34, "RBF SVM: N/A\n(un-calibrated margins)", ha="center", va="top", fontsize=8)
    axes[1, 1].bar(x, inference, color=NAVY)
    axes[1, 1].set_title("Student-test inference time in seconds")
    axes[1, 1].set_ylim(0, 67)
    for ax in axes.flat:
        ax.set_xticks(x, labels, fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Model comparison using saved run outputs", color=NAVY, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)

    base = tuning["baseline_student_test"]
    opt = tuning["tuned_student_test"]
    base_cv = tuning["baseline_cv"]["folds"]
    opt_cv = tuning["tuned_cv"]["folds"]
    compare = [
        ("CV route Macro F1", np.mean([r["route_macro_f1"] for r in base_cv]), np.mean([r["route_macro_f1"] for r in opt_cv])),
        ("Student Top-1 accuracy", base["top1_accuracy"], opt["top1_accuracy"]),
        ("Student route Macro F1", base["route_macro_f1"], opt["route_macro_f1"]),
        ("Student candidate log loss", base["log_loss"], opt["log_loss"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), dpi=180)
    for ax, (title, baseline, selected) in zip(axes.flat, compare):
        ax.bar([0, 1], [baseline, selected], color=["#91A7B5", GREEN], width=0.58)
        ax.set_xticks([0, 1], ["Untuned", "Tuned"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        for xpos, val in enumerate([baseline, selected]):
            ax.text(xpos, val, f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("HistGradientBoosting tuning comparison", color=NAVY, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "hgb_tuning.png", bbox_inches="tight")
    plt.close(fig)

    groups = [
        ("Training CV", decoder["cv_summary"]["classifier_only"], decoder["cv_summary"]["hmm_viterbi"]),
        ("Student test", decoder["selected_student_setting"]["classifier_only"], decoder["selected_student_setting"]["hmm_viterbi"]),
        ("Reserved evaluator", final["means_across_trips"]["classifier_only"], final["means_across_trips"]["hmm_viterbi"]),
    ]
    metric_keys = ["top1_accuracy", "route_macro_f1", "route_continuity"]
    metric_titles = ["Top-1 accuracy", "Route Macro F1", "Route continuity"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), dpi=180)
    for ax, key, title in zip(axes, metric_keys, metric_titles):
        basevals = [g[1][key] for g in groups]
        hmmvals = [g[2][key] for g in groups]
        xx = np.arange(3)
        ax.bar(xx - 0.18, basevals, width=0.36, label="Classifier only", color="#91A7B5")
        ax.bar(xx + 0.18, hmmvals, width=0.36, label="HMM/Viterbi", color=RED)
        ax.set_title(title)
        ax.set_xticks(xx, [g[0] for g in groups], rotation=12, ha="right", fontsize=8)
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Route decoding results across partitions", color=NAVY, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT / "decoder_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), dpi=180)
    for model, color, column in (("mlp_shallow", BLUE, 0), ("mlp_deep", RED, 1)):
        history = experiments["models"][model]["training_history"]["student_fit"]
        epochs = np.arange(1, history["epochs_run"] + 1)
        axes[0, column].plot(epochs, history["training_loss"], color=color, linewidth=1.6)
        axes[0, column].set_title(f"{model.replace('_', ' ').title()} training loss")
        axes[0, column].set_xlabel("Epoch")
        axes[0, column].set_ylabel("Training loss")
        axes[1, column].plot(epochs, history["validation_scores"], color=color, linewidth=1.6)
        axes[1, column].set_title("Held-out validation accuracy")
        axes[1, column].set_xlabel("Epoch")
        axes[1, column].set_ylabel("Candidate validation accuracy")
        for ax in (axes[0, column], axes[1, column]):
            ax.grid(alpha=0.2)
    fig.suptitle("Student-fit MLP epoch histories from executed runs", color=NAVY, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "mlp_histories.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Report plots written to {OUT}")


if __name__ == "__main__":
    main()
