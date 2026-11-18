"""Create the editable PathSnap technical report from saved project evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ASSETS = REPORTS / "assets"
NAVY = "17324D"
BLUE = "2474A6"
PALE = "EEF3F7"
MID = "D8E2EA"
GREEN = "29945E"
RED = "C23C32"
GRAY = "536574"


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def f4(value) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def pct(value) -> str:
    return "N/A" if value is None else f"{float(value) * 100:.2f}%"


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_margins(cell, top=70, start=80, bottom=70, end=80) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def keep_with_next(paragraph) -> None:
    paragraph.paragraph_format.keep_with_next = True


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    tr_pr.append(repeat)


def set_cell_text(cell, text: str, *, bold=False, color="263746", size=8.2, align=None) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    if align is not None:
        p.alignment = align
    run = p.add_run(str(text))
    run.bold = bold
    run.font.name = "Aptos"
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    set_cell_margins(cell)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float] | None = None, font=8.1):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_repeat_table_header(table.rows[0])
    for idx, (cell, label) in enumerate(zip(table.rows[0].cells, headers)):
        set_cell_text(cell, label, bold=True, color="FFFFFF", size=font)
        set_cell_shading(cell, NAVY)
        if widths:
            cell.width = Inches(widths[idx])
    for ridx, values in enumerate(rows):
        cells = table.add_row().cells
        for idx, (cell, value) in enumerate(zip(cells, values)):
            set_cell_text(cell, value, size=font)
            if ridx % 2 == 1:
                set_cell_shading(cell, "F4F7F9")
            if widths:
                cell.width = Inches(widths[idx])
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_caption(doc: Document, text: str) -> None:
    p = doc.add_paragraph(style="Caption")
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(9)
    r = p.add_run(text)
    r.italic = True
    r.font.color.rgb = RGBColor.from_string(GRAY)
    keep_with_next(p)


def add_image(doc: Document, filename: str, width=6.85, caption: str | None = None) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(2)
    p.add_run().add_picture(str(filename), width=Inches(width))
    if caption:
        add_caption(doc, caption)


def add_bullet(doc: Document, text: str) -> None:
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.space_after = Pt(2)


def add_footer(section) -> None:
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.paragraph_format.space_before = Pt(0)
    run = footer.add_run("PathSnap Technical Report  |  ")
    run.font.name = "Aptos"
    run.font.size = Pt(8)
    run.font.color.rgb = RGBColor.from_string(GRAY)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)


def set_style_font(style, name="Aptos", size=10, color="263746", bold=False):
    style.font.name = name
    style.font.size = Pt(size)
    style.font.color.rgb = RGBColor.from_string(color)
    style.font.bold = bold
    style._element.rPr.rFonts.set(qn("w:ascii"), name)
    style._element.rPr.rFonts.set(qn("w:hAnsi"), name)


def setup_doc() -> Document:
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.62)
    sec.bottom_margin = Inches(0.62)
    sec.left_margin = Inches(0.68)
    sec.right_margin = Inches(0.68)
    add_footer(sec)
    styles = doc.styles
    normal = styles["Normal"]
    set_style_font(normal, size=9.7)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.1
    set_style_font(styles["Title"], size=27, color=NAVY, bold=True)
    set_style_font(styles["Subtitle"], size=14, color=GRAY)
    for name, size in (("Heading 1", 17), ("Heading 2", 12.5), ("Heading 3", 10.5)):
        set_style_font(styles[name], size=size, color=NAVY, bold=True)
        styles[name].paragraph_format.space_before = Pt(13 if name == "Heading 1" else 9)
        styles[name].paragraph_format.space_after = Pt(4)
        styles[name].paragraph_format.keep_with_next = True
    set_style_font(styles["Caption"], size=8.2, color=GRAY)
    return doc


def add_model_subsection(doc, row, human_name, reason, observed, config, objective):
    doc.add_heading(human_name, level=2)
    s = row["student_test"]
    p = doc.add_paragraph()
    p.add_run("Why we included it. ").bold = True
    p.add_run(reason)
    p = doc.add_paragraph()
    p.add_run("Configuration and objective. ").bold = True
    p.add_run(f"{config}. {objective}")
    p = doc.add_paragraph()
    p.add_run("Observed results. ").bold = True
    p.add_run(
        f"Five-fold grouped-CV route Macro F1 was {float(np_mean(row['cv']['folds'], 'route_macro_f1')):.4f}; "
        f"student-test Top-1 was {float(s['top1_accuracy']):.4f}, route Macro F1 was {float(s['route_macro_f1']):.4f}, "
        f"candidate-row precision, recall and Macro F1 were {float(s['candidate_precision']):.4f}, {float(s['candidate_recall']):.4f} and {float(s['candidate_macro_f1']):.4f}; "
        f"candidate log loss was {f4(s['log_loss'])}. Fit took {float(s['fit_seconds']):.2f} s and inference took {float(s['inference_seconds']):.2f} s. {observed}"
    )
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    p.add_run("Fit assessment. ").bold = True
    if s["model"] in ("mlp_shallow", "mlp_deep"):
        hist = row["training_history"]["student_fit"]
        p.add_run(
            f"The saved training loss fell over {hist['epochs_run']} epochs and validation candidate accuracy rose overall. "
            "The held-out curve did not show a sharp late decline. This is limited evidence against severe overfitting on the candidate-row validation split; it is not a route-level generalization guarantee."
        )
    else:
        p.add_run(
            "The saved experiment does not include a training-versus-validation learning curve for this estimator, so it does not support a direct overfitting claim. "
            "Its grouped-CV and student-test route scores are the available generalization evidence."
        )


def np_mean(rows, key):
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(values) / max(1, len(values))


def main() -> None:
    audit = load_json("results/dataset_audit.json")
    results = load_json("results/experiment_results.json")
    tuning = load_json("results/bayesian_tuning.json")
    decoder = load_json("results/decoder_comparison.json")
    final = load_json("results/final_evaluator.json")
    metadata = load_json("artifacts/final_metadata.json")
    screenshot = REPORTS / "evidence" / "pathsnap_ui_trip_00000067.png"
    api_log = (REPORTS / "evidence" / "api_prediction_log.txt").read_text(encoding="utf-8").strip()

    doc = setup_doc()
    title = doc.add_paragraph(style="Title")
    title.paragraph_format.space_before = Pt(38)
    title.add_run("PathSnap Technical Report")
    subtitle = doc.add_paragraph(style="Subtitle")
    subtitle.add_run("Trip Level GPS Map Matching with HistGradientBoosting and HMM Viterbi")
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(9)
    p.add_run("Project report draft  |  18 November 2026").font.color.rgb = RGBColor.from_string(GRAY)
    doc.add_heading("Executive Summary", level=1)
    doc.add_paragraph(
        "We built PathSnap to turn a noisy GPS trace into a plausible sequence of road arcs from the trip's supplied graph. "
        "The system generates up to eight nearby arc candidates per observation, scores each candidate with a binary HistGradientBoosting classifier, and applies Viterbi decoding to account for directed graph connectivity and movement between observations."
    )
    doc.add_paragraph(
        f"We compared eight approaches on the same candidate features. HistGradientBoosting was selected for tuning because it was effectively tied with Random Forest on grouped-CV route Macro F1, had slightly better student-test route scores, lower candidate log loss, and much shorter fit time in the recorded run. "
        f"The frozen HGB plus HMM/Viterbi pipeline achieved mean trip Top-1 {final['means_across_trips']['hmm_viterbi']['top1_accuracy']:.4f}, route Macro F1 {final['means_across_trips']['hmm_viterbi']['route_macro_f1']:.4f}, and continuity {final['means_across_trips']['hmm_viterbi']['route_continuity']:.4f} across the 10 reserved evaluator trips. "
        "No evaluator results were used to select the model or decoder."
    )
    add_table(doc, ["Final reserved evaluator result", "Classifier only", "HGB plus HMM/Viterbi"], [
        ["Mean trip Top-1 accuracy", f4(final["means_across_trips"]["classifier_only"]["top1_accuracy"]), f4(final["means_across_trips"]["hmm_viterbi"]["top1_accuracy"])],
        ["Mean trip route Macro F1", f4(final["means_across_trips"]["classifier_only"]["route_macro_f1"]), f4(final["means_across_trips"]["hmm_viterbi"]["route_macro_f1"])],
        ["Mean trip route continuity", f4(final["means_across_trips"]["classifier_only"]["route_continuity"]), f4(final["means_across_trips"]["hmm_viterbi"]["route_continuity"])],
        ["Candidate miss rate", f4(final["means_across_trips"]["classifier_only"]["candidate_miss_rate"]), f4(final["means_across_trips"]["hmm_viterbi"]["candidate_miss_rate"])],
        ["Exact full-sequence match", f4(final["means_across_trips"]["classifier_only"]["sequence_exact"]), f4(final["means_across_trips"]["hmm_viterbi"]["sequence_exact"])],
    ], widths=[2.4, 1.7, 2.35], font=8.5)
    doc.add_paragraph(
        "The principal remaining limits are candidate misses, trip-local reference-route label noise, no exact full-trip matches, and improvement that is not uniform across individual trips or metrics. The team should review the analysis and comply with the event's AI-assistance disclosure policy before submission."
    )
    doc.add_page_break()

    doc.add_heading("Contents", level=1)
    for item in [
        "1 Problem and System Overview",
        "2 Dataset Understanding and Exploratory Data Analysis",
        "3 Model Experimentation and Comparison",
        "4 Bayesian Hyperparameter Optimization",
        "5 HMM and Viterbi Route Reconstruction",
        "6 Geographical Visualization Using Folium",
        "7 API UI and Deployment",
        "8 Limitations Reproducibility and Conclusions",
    ]:
        add_bullet(doc, item)

    doc.add_heading("1 Problem and System Overview", level=1)
    doc.add_paragraph(
        "PathSnap receives a time-ordered sequence of GPS coordinates and a directed road graph for that trip. Its task is to return one trip-local road-arc ID for each observation when a candidate exists. A GPS row does not correspond to a node or arc row. Each trip has a separate graph and a separate reference-route vocabulary, so a global multiclass classifier over arc IDs would not be valid."
    )
    add_table(doc, ["System stage", "Implementation"], [
        ["Candidate search", "Spherical midpoint index; local segment-distance ranking; 80 m radius and top 8 arcs."],
        ["Candidate features", "Twelve observation and arc features from GPS, graph geometry and local node degree."],
        ["Classifier", "Binary HistGradientBoosting candidate score; positive means candidate equals the nearest reference-route arc for that observation."],
        ["Route decoder", "HMM/Viterbi uses classifier scores as emissions and directed network path distance as transition evidence."],
        ["Serving", "FastAPI owns the model, archive, route-free prediction path and server-side scoring; UI sends only a trip ID."],
    ], widths=[1.45, 5.0], font=8.4)
    doc.add_paragraph(
        "Training and model comparison use complete trips. The 80 training trips are split into five trip-grouped folds; observations from one trip cannot occur in both fit and validation folds. Ten student-test trips support development comparisons and decoder setting selection. Ten separate evaluator trips were used only after the settings were frozen."
    )

    doc.add_heading("2 Dataset Understanding and Exploratory Data Analysis", level=1)
    train = audit["partitions"]["train"]
    student = audit["partitions"]["student_test"]
    eval_count = final["trip_count"]
    add_table(doc, ["Partition", "Trips", "GPS observations", "Nodes", "Arcs", "Route rows"], [
        ["Train", "80", f"{train['total_rows_by_extension']['track']:,}", f"{train['total_rows_by_extension']['nodes']:,}", f"{train['total_rows_by_extension']['arcs']:,}", f"{train['total_rows_by_extension']['route']:,}"],
        ["Student test", "10", f"{student['total_rows_by_extension']['track']:,}", f"{student['total_rows_by_extension']['nodes']:,}", f"{student['total_rows_by_extension']['arcs']:,}", f"{student['total_rows_by_extension']['route']:,}"],
        ["Reserved evaluator", str(eval_count), "Scored after settings were frozen", "Trip-specific", "Trip-specific", "Kept server-side"],
    ], widths=[1.25, 0.55, 1.1, 0.9, 0.9, 1.5], font=8.1)
    doc.add_paragraph(
        "The four files are whitespace-separated and headerless. Track columns are longitude, latitude and timestamp seconds. Node rows define longitude and latitude, with the row index used as node ID. Arc rows define integer from-node and to-node IDs, with row index used as arc ID. Route rows give an ordered sequence of arc IDs. All arc endpoints and route IDs in the supplied student archive were within range. Every supplied trip had all four files. The audit found no non-finite GPS or node coordinates, duplicate consecutive GPS rows, or non-positive time intervals."
    )
    doc.add_paragraph(
        f"Training track intervals had a median of {train['timestamp_interval_seconds']['median']:.0f} second and a maximum of {train['timestamp_interval_seconds']['max']:.0f} seconds. Student-test intervals had a median of {student['timestamp_interval_seconds']['median']:.0f} second and a maximum of {student['timestamp_interval_seconds']['max']:.0f} seconds. The dataset spans multiple continents (training longitude {train['coordinate_bounds_lon_lat']['lon_min']:.2f} to {train['coordinate_bounds_lon_lat']['lon_max']:.2f}; latitude {train['coordinate_bounds_lon_lat']['lat_min']:.2f} to {train['coordinate_bounds_lon_lat']['lat_max']:.2f}), so global planar degree distances are inappropriate. Candidate segment distance uses a local tangent approximation and the pipeline uses haversine distance for geographic displacements."
    )
    doc.add_paragraph(
        f"The training graphs contained {train['route_disconnected_transitions']} disconnected transitions along the supplied reference routes; the student-test set contained {student['route_disconnected_transitions']}. These cases show that the supplied route is a reference label with data limitations, not guaranteed physical ground truth. Graphs also contain multiple components, and timestamp gaps can be large."
    )
    add_image(doc, ROOT / "results/trip_and_sampling_summary.png", width=6.7,
              caption="Figure 1. Saved EDA summaries for largest within-trip time gaps and GPS observations per trip. Partitions shown are training and student test.")
    doc.add_heading("Preprocessing and Feature Choices", level=2)
    doc.add_paragraph(
        "For each GPS observation, the candidate generator indexes spherical arc midpoints, examines a bounded nearest set, computes local point-to-segment distance, retains candidates within 80 m, and returns at most eight. If no arc falls within the radius, the closest scored arc candidates provide a fallback. The candidate miss rate is measured before interpreting classifier results because a decoder cannot recover a reference arc that was never proposed."
    )
    add_table(doc, ["Feature", "Reason it is available and useful at inference"], [
        ["Distance to arc; projection fraction; arc length", "Describe geometric fit and location along each directed segment."],
        ["Heading cosine and sine", "Compare observed movement direction with candidate arc bearing without a wraparound angle discontinuity."],
        ["Speed; time gap; previous and next step distance", "Describe motion and account for irregular sampling. The full track is available to the API."],
        ["Distance rank", "Represent the candidate's relative position in the local distance ordering."],
        ["From-node and to-node degree", "Describe local graph connectivity at the candidate endpoints."],
    ], widths=[2.1, 4.4], font=8.3)
    doc.add_paragraph(
        "The positive binary label is the candidate arc that matches the nearest geometric arc on that same trip's ordered reference route for the GPS observation. The route is used only to create labels during permitted training/development and for scoring after inference. StandardScaler is part of the Logistic Regression, KNN, SVM and MLP pipelines; tree models use the numeric features directly. No transformation is fitted across validation trips."
    )

    doc.add_heading("3 Model Experimentation and Comparison", level=1)
    doc.add_paragraph(
        "We narrowed the suggested approaches to eight families that share the candidate-label task. This scope decision was made before reading run scores. Every approach used the same candidate generation, features, five grouped folds, random seed 352 and student-test trips. Model and candidate IDs remain trip-local. The selection emphasized route Macro F1, then probability quality, speed and compatibility with the HMM."
    )
    rows_csv = list(csv.DictReader((ROOT / "results/model_comparison.csv").open(encoding="utf-8-sig", newline="")))
    hgb_student = tuning["tuned_student_test"]
    hgb_cv = tuning["tuned_cv"]["folds"]
    table_rows = []
    order = ["logistic_regression", "distance_weighted_knn", "random_forest", "extra_trees", "hist_gradient_boosting", "rbf_svm", "mlp_shallow", "mlp_deep"]
    display = {"logistic_regression": "Logistic\nRegression", "distance_weighted_knn": "Distance weighted\nKNN", "random_forest": "Random\nForest", "extra_trees": "Extra\nTrees", "hist_gradient_boosting": "HistGradient\nBoosting tuned", "rbf_svm": "RBF\nSVM", "mlp_shallow": "Shallow\nMLP", "mlp_deep": "Deep\nMLP"}
    families = {r["model"]: r["family"] for r in rows_csv}
    cfgs = {r["model"]: json.loads(r["configuration"]) for r in rows_csv}
    for name in order:
        row = next(r for r in rows_csv if r["model"] == name)
        if name == "hist_gradient_boosting":
            cv_f1 = np_mean(hgb_cv, "route_macro_f1")
            s = hgb_student
            config = "lr .0656; iter 167; leaves 51"
        else:
            cv_f1 = float(row["cv_route_macro_f1_mean"])
            s = results["models"][name]["student_test"]
            conf = cfgs[name]
            config = short_config(name, conf)
        table_rows.append([
            display[name], families[name], config, f"{cv_f1:.3f}", f"{float(s['top1_accuracy']):.3f}",
            f"{float(s['route_continuity']):.3f}", f"{float(s['route_macro_f1']):.3f}",
            f4(s.get("log_loss") if "log_loss" in s else s.get("student_candidate_log_loss")),
            f"{float(s['inference_seconds']):.2f}",
        ])
    add_table(doc,
        ["Model", "Family", "Key configuration", "CV route\nMacro F1", "Student\nTop-1", "Student\ncontinuity", "Student route\nMacro F1", "Student\ncandidate log loss", "Inference\nseconds"],
        table_rows,
        widths=[0.88, 0.76, 1.3, 0.58, 0.52, 0.63, 0.65, 0.63, 0.54], font=6.8)
    add_caption(doc, "Table 2. Route scores are means across trips. CV values average the five trip-grouped folds. HGB row uses the selected tuned model; other rows show their recorded initial configurations. RBF SVM log loss is N/A because the model used uncalibrated decision scores.")
    add_image(doc, ASSETS / "model_comparison.png", width=6.8,
              caption="Figure 2. Grouped-CV Macro F1, student-test Top-1 and route Macro F1, candidate log loss, and inference time. The plot uses saved run outputs; lower log loss is better.")
    doc.add_paragraph(
        "HistGradientBoosting and Random Forest were nearly tied before tuning on CV route Macro F1 (0.6271 and 0.6270). HGB was slightly higher on student-test Top-1 and route Macro F1, had lower candidate log loss, and took 3.58 seconds to fit the student run versus 119.24 seconds for Random Forest. The selection therefore reflects the recorded trade-off, not a large CV accuracy separation."
    )

    # Individual model subsections use saved per-model result records.
    doc.add_heading("Individual Model Results", level=2)
    model_notes = [
        ("logistic_regression", "Model 1 Logistic Regression", "A scaled linear probabilistic baseline provides a simple reference and candidate probabilities.", "It had the lowest student candidate Macro F1 among probability-producing models and the highest log loss in this comparison; its fast 0.33 s inference is useful as a baseline." , "C=1.0, max_iter=300, balanced class weights", "Binary logistic loss with L2 regularization; class weights address the candidate-label imbalance."),
        ("distance_weighted_knn", "Model 2 Distance Weighted KNN", "A local-neighbor method tests whether candidates with similar geometry and motion features tend to share the positive label.", "Candidate-row Macro F1 was relatively high, but route Macro F1 was 0.6093 and inference took 60.31 s, the slowest of the eight in the recorded run." , "15 neighbors, inverse-distance weights, Euclidean distance (p=2), maximum 30,000 training rows", "No fitted loss objective; prediction uses distance-weighted neighbor votes."),
        ("random_forest", "Model 3 Random Forest", "Bagged nonlinear trees can represent interactions among geometry, motion and graph degree while reducing single-tree variance.", "It was the strongest untuned student route result among the initial models, at 0.6460 Macro F1, and nearly tied HGB in grouped CV; fit time was much longer." , "200 trees, min leaf 2, sqrt feature sampling, balanced_subsample class weights", "Bootstrap tree ensemble using the scikit-learn Gini split criterion."),
        ("extra_trees", "Model 4 Extra Trees", "Randomized tree splits provide a second tree-ensemble comparison with less split-search computation per node.", "It reached student Top-1 0.8190 and route Macro F1 0.6309; candidate log loss 0.1595 was worse than HGB and Random Forest." , "250 trees, min leaf 2, sqrt feature sampling, balanced class weights", "Extremely randomized tree ensemble using the scikit-learn Gini split criterion."),
        ("hist_gradient_boosting", "Model 5 HistGradientBoosting", "Histogram boosting offers nonlinear interactions with comparatively low fit cost on the large candidate table.", "The selected model improved student route Macro F1 from 0.6467 to 0.6518 after tuning, while Top-1 changed from 0.8237 to 0.8232. Tuned candidate log loss was 0.1270." , "Tuned learning_rate 0.0655946, max_iter 167, max_leaf_nodes 51, l2_regularization 1.0", "Binary log-loss fitting; Bayesian search optimized mean route Macro F1 over five grouped folds."),
        ("rbf_svm", "Model 6 RBF Support Vector Machine", "The RBF kernel tests a nonlinear margin boundary after feature scaling.", "Student Top-1 was 0.8159 and route Macro F1 0.6249. Inference took 20.12 s. Log loss is N/A because probability calibration was disabled and decision values are ranking scores." , "C=1.0, gamma=scale, maximum 10,000 sampled training rows", "C-SVC margin objective; decision scores are converted to within-observation softmax emissions for decoding, not reported as calibrated probabilities."),
        ("mlp_shallow", "Model 7 Shallow MLP", "One hidden layer tests whether a small learned nonlinear representation improves candidate ranking.", "It reached student Top-1 0.8194, route Macro F1 0.6269, candidate log loss 0.1346 and inference time 0.84 s." , "One hidden layer of 64 units, alpha 0.001, max_iter 150, early_stopping=True", "Adam optimization of binary cross entropy; student fit stopped after 53 epochs."),
        ("mlp_deep", "Model 8 Deep MLP", "Three hidden layers test whether additional representation depth helps on the same twelve features.", "It reached route Macro F1 0.6328 and candidate log loss 0.1314, better than the shallow MLP on those metrics, but still below tuned HGB route Macro F1. Student fit stopped after 29 epochs." , "Hidden layers 128, 64 and 32; alpha 0.001, max_iter 150, early_stopping=True", "Adam optimization of binary cross entropy; early stopping ended the student run after 29 epochs."),
    ]
    for name, human, reason, observed, config, objective in model_notes:
        if name == "hist_gradient_boosting":
            row = results["models"][name]
            # Replace the initial HGB result with tuned fold and student values for this final model subsection.
            row = dict(row)
            row["cv"] = {"folds": hgb_cv}
            row["student_test"] = hgb_student
        else:
            row = results["models"][name]
        add_model_subsection(doc, row, human, reason, observed, config, objective)
        if name == "mlp_shallow":
            add_image(doc, ASSETS / "mlp_histories.png", width=6.65,
                      caption="Figure 3. Actual student-fit MLP training-loss and candidate validation-accuracy histories. The stopping epoch differs by model; these are candidate-level curves, not route-level scores.")
    doc.add_paragraph(
        "The other suggested approaches were excluded under the team's explicit eight-model scope, not because they were measured as inferior. The reasons and full list of exclusions are in docs/model_scope.md. The plot in Figure 2 provides result visualization for all eight models; only MLPs produced saved epoch-wise histories."
    )

    doc.add_heading("4 Bayesian Hyperparameter Optimization", level=1)
    best = tuning["best_parameters"]
    doc.add_paragraph(
        f"We tuned HistGradientBoosting after the initial comparison. The search used {tuning['trial_budget']} Gaussian-process expected-improvement trials. It varied learning rate from 0.02 to 0.20, max_iter from 50 to 250, and max_leaf_nodes from 15 to 63. The objective was mean route Macro F1 across five GroupKFold splits of the 80 training trips. The best trial used learning_rate={best['learning_rate']:.7f}, max_iter={best['max_iter']}, and max_leaf_nodes={best['max_leaf_nodes']}; l2_regularization stayed at 1.0. Search time was {tuning['compute_seconds']:.2f} seconds. All trial records are in results/bayesian_tuning.json."
    )
    add_image(doc, ASSETS / "hgb_tuning.png", width=6.65,
              caption="Figure 4. Untuned and tuned HGB results on matching training CV folds and student-test trips. Candidate log loss decreased slightly; student Top-1 changed slightly downward.")
    baseline = tuning["baseline_student_test"]
    tuned = tuning["tuned_student_test"]
    cv_base = tuning["baseline_cv"]["folds"]
    cv_tuned = tuning["tuned_cv"]["folds"]
    add_table(doc, ["Metric", "Untuned", "Tuned", "Change"], [
        ["Grouped-CV route Macro F1", f4(np_mean(cv_base, "route_macro_f1")), f4(np_mean(cv_tuned, "route_macro_f1")), f"{np_mean(cv_tuned, 'route_macro_f1')-np_mean(cv_base, 'route_macro_f1'):+.4f}"],
        ["Student Top-1 accuracy", f4(baseline["top1_accuracy"]), f4(tuned["top1_accuracy"]), f"{tuned['top1_accuracy']-baseline['top1_accuracy']:+.4f}"],
        ["Student route Macro F1", f4(baseline["route_macro_f1"]), f4(tuned["route_macro_f1"]), f"{tuned['route_macro_f1']-baseline['route_macro_f1']:+.4f}"],
        ["Student candidate log loss", f4(baseline["log_loss"]), f4(tuned["log_loss"]), f"{tuned['log_loss']-baseline['log_loss']:+.4f}"],
        ["Student inference seconds", f"{baseline['inference_seconds']:.3f}", f"{tuned['inference_seconds']:.3f}", f"{tuned['inference_seconds']-baseline['inference_seconds']:+.3f}"],
        ["Student fit seconds", f"{baseline['fit_seconds']:.3f}", f"{tuned['fit_seconds']:.3f}", f"{tuned['fit_seconds']-baseline['fit_seconds']:+.3f}"],
    ], widths=[2.2, 1.35, 1.35, 1.5], font=8.4)
    doc.add_paragraph(
        "Tuning raised CV route Macro F1 by 0.0013 and student route Macro F1 by 0.0051. Student Top-1 and route continuity moved down by less than 0.001. Candidate miss rate remained 0.00919 because the generator did not change. This is a modest improvement, not a broad gain across every metric. The final fitted model, schema and decoder settings are persisted in artifacts/final_model.joblib and artifacts/final_metadata.json."
    )

    doc.add_heading("5 HMM and Viterbi Route Reconstruction", level=1)
    doc.add_paragraph(
        "The hidden state at each observation is one of that observation's candidate road arc IDs. Candidate classifier scores become within-observation softmax emission weights. Probability-producing classifiers provide their candidate probability scores; for RBF SVM, comparative decision values are used as uncalibrated ranking scores. The selected final HGB supplies probabilities."
    )
    doc.add_paragraph(
        "For a transition from a previous candidate to a next candidate, the decoder finds a directed shortest path from the previous arc's end node to the next arc's start node and adds the next arc length. It compares that route distance with the haversine distance travelled between GPS fixes. The log transition score is the negative absolute mismatch divided by sigma plus one quarter of observed displacement. A disconnected pair receives a configurable log penalty. Viterbi sums log emissions and transitions and chooses the highest-scoring sequence."
    )
    add_table(doc, ["Condition", "Decoder behavior"], [
        ["No candidates at an observation", "Emit -1 for that GPS fix; do not invent a road arc."],
        ["Missing candidate index or invalid/nonpositive time gap", "Start a new contiguous decoding block."],
        ["Time gap greater than 120 seconds", "Reset the chain and decode a new block."],
        ["Disconnected candidate transition", "Use log penalty -12; finite transitions use network distance consistency."],
        ["Plausible path search", "Limit directed network search using max speed 45 m/s and a displacement allowance."],
    ], widths=[2.05, 4.55], font=8.3)
    doc.add_paragraph(
        "The decoder was compared to classifier-only top-candidate predictions over grouped CV, student test and then the reserved evaluator set. The small decoder grid selected max_gap_seconds=120, max_speed_mps=45, sigma_m=80 m and disconnected_log_penalty=-12 by mean student-test route Macro F1, with continuity as tie-breaker. Because student labels selected this setting, the student result is development-tuned. The evaluator result below is separate and did not feed back into the choice."
    )
    add_image(doc, ASSETS / "decoder_comparison.png", width=6.75,
              caption="Figure 5. Classifier-only and HMM/Viterbi route metrics across training CV, student-test development, and reserved evaluator partitions. Candidate miss rate is unchanged by decoding.")
    add_table(doc, ["Partition", "Method", "Top-1", "Route Macro F1", "Continuity", "Candidate miss", "Exact sequence"], [
        ["Training CV", "Classifier only", f4(decoder["cv_summary"]["classifier_only"]["top1_accuracy"]), f4(decoder["cv_summary"]["classifier_only"]["route_macro_f1"]), f4(decoder["cv_summary"]["classifier_only"]["route_continuity"]), f4(decoder["cv_summary"]["classifier_only"]["candidate_miss_rate"]), f4(decoder["cv_summary"]["classifier_only"]["sequence_exact"])],
        ["Training CV", "HMM/Viterbi", f4(decoder["cv_summary"]["hmm_viterbi"]["top1_accuracy"]), f4(decoder["cv_summary"]["hmm_viterbi"]["route_macro_f1"]), f4(decoder["cv_summary"]["hmm_viterbi"]["route_continuity"]), f4(decoder["cv_summary"]["hmm_viterbi"]["candidate_miss_rate"]), f4(decoder["cv_summary"]["hmm_viterbi"]["sequence_exact"])],
        ["Student test", "Classifier only", f4(decoder["selected_student_setting"]["classifier_only"]["top1_accuracy"]), f4(decoder["selected_student_setting"]["classifier_only"]["route_macro_f1"]), f4(decoder["selected_student_setting"]["classifier_only"]["route_continuity"]), f4(decoder["selected_student_setting"]["classifier_only"]["candidate_miss_rate"]), f4(decoder["selected_student_setting"]["classifier_only"]["sequence_exact"])],
        ["Student test", "HMM/Viterbi", f4(decoder["selected_student_setting"]["hmm_viterbi"]["top1_accuracy"]), f4(decoder["selected_student_setting"]["hmm_viterbi"]["route_macro_f1"]), f4(decoder["selected_student_setting"]["hmm_viterbi"]["route_continuity"]), f4(decoder["selected_student_setting"]["hmm_viterbi"]["candidate_miss_rate"]), f4(decoder["selected_student_setting"]["hmm_viterbi"]["sequence_exact"])],
        ["Evaluator 10 trips", "Classifier only", f4(final["means_across_trips"]["classifier_only"]["top1_accuracy"]), f4(final["means_across_trips"]["classifier_only"]["route_macro_f1"]), f4(final["means_across_trips"]["classifier_only"]["route_continuity"]), f4(final["means_across_trips"]["classifier_only"]["candidate_miss_rate"]), f4(final["means_across_trips"]["classifier_only"]["sequence_exact"])],
        ["Evaluator 10 trips", "HMM/Viterbi", f4(final["means_across_trips"]["hmm_viterbi"]["top1_accuracy"]), f4(final["means_across_trips"]["hmm_viterbi"]["route_macro_f1"]), f4(final["means_across_trips"]["hmm_viterbi"]["route_continuity"]), f4(final["means_across_trips"]["hmm_viterbi"]["candidate_miss_rate"]), f4(final["means_across_trips"]["hmm_viterbi"]["sequence_exact"])],
    ], widths=[1.03, 1.05, 0.7, 0.9, 0.8, 0.86, 0.83], font=7.1)
    doc.add_paragraph(
        "On the evaluator set, HMM/Viterbi raised mean trip Top-1 from 0.8081 to 0.8736 and route Macro F1 from 0.6195 to 0.7280. Continuity rose from 0.8873 to 0.9429. Candidate miss rate stayed 0.0202 and no trip achieved exact full-sequence match. The decoder therefore improved aggregate point and route metrics but did not eliminate missed candidates or all point errors."
    )
    doc.add_paragraph(
        "The decoder did not help every student trip and metric. For trip 00000007, Top-1 decreased from 0.9018 to 0.8752 while route Macro F1 rose from 0.8741 to 0.9307. For trip 00000056, Top-1 fell from 0.7943 to 0.7812 and route Macro F1 from 0.7272 to 0.7255, although continuity improved. No full sequence was an exact match in CV, student test or evaluator data."
    )

    doc.add_heading("6 Geographical Visualization Using Folium", level=1)
    doc.add_paragraph(
        "The interactive Folium map was generated from student-test trip 00000007 using the actual trip graph, GPS track, supplied route and the frozen HGB plus HMM/Viterbi prediction. It has independent layer controls for GPS observations, nearby road network, reference route and predicted route, plus start and end markers. The same geometry is used in the static report figure."
    )
    add_image(doc, ROOT / "visualizations/representative_trip.png", width=6.65,
              caption="Figure 6. Student-test trip 00000007. Blue shows the observed GPS trace, gray the nearby road network, green the supplied reference route, and red the decoded prediction. Interactive layer-controlled map: visualizations/representative_trip.html.")
    doc.add_paragraph(
        "For this trip, the API-compatible inference pipeline returned pointwise accuracy 0.8796, route Macro F1 0.9309 and candidate miss rate 0.0000. This map is an illustrative student-test visualization, not a final evaluator score or proof that every route segment is physically correct."
    )

    doc.add_heading("7 API UI and Deployment", level=1)
    doc.add_paragraph(
        "The FastAPI service loads the artifact and evaluator trip list once at startup. GET /health reports readiness; GET /trips returns trip IDs; POST /predict accepts a trip ID. The backend reads the selected trip, creates a route-free Trip for feature extraction and decoding, then reads the reference route separately for server-side evaluation. The response contains the predicted arc IDs, GPS and predicted geometry, and permitted summary metrics. It does not return reference arc IDs."
    )
    doc.add_paragraph(
        "The UI requests trip IDs, lets the evaluator select any returned ID, posts that ID, and displays the API response with a GPS trace, decoded path, pointwise accuracy, route Macro F1 and candidate miss rate. A working local run used API port 8001 and UI port 8502. The local command and split-laptop launch procedure are in docs/deployment.md and README.md."
    )
    if screenshot.exists():
        add_image(doc, screenshot, width=6.65,
                  caption="Figure 7. Submitted UI run for evaluator trip 00000067. The visible result shows 916 GPS observations, pointwise accuracy 80.5%, route Macro F1 0.740 and candidate miss rate 0.0%.")
    doc.add_paragraph("API terminal log excerpt supplied with the UI run:")
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.2)
    p.paragraph_format.right_indent = Inches(0.2)
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    run = p.add_run(api_log)
    run.font.name = "Consolas"
    run.font.size = Pt(7.5)
    run.font.color.rgb = RGBColor.from_string(GRAY)
    doc.add_paragraph(
        "The API/UI code shown here is in src/api.py, src/inference.py and ui/. The model artifact, candidate configuration, feature schema, random seed, HMM settings, Python version and scikit-learn version are recorded in artifacts/final_metadata.json. Scikit-learn is pinned to 1.6.1 because the serialized HGB artifact was created with that version."
    )

    doc.add_heading("8 Limitations Reproducibility and Conclusions", level=1)
    add_bullet(doc, "Candidate generation limits the achievable score. An omitted reference arc cannot be recovered by the classifier or decoder; miss rate was 0.0353 in grouped CV, 0.0092 on student test, and 0.0202 on the evaluator set.")
    add_bullet(doc, "The route file is treated as a reference label. A small number of consecutive reference arcs are disconnected in the supplied graphs, which may reflect label or graph limitations.")
    add_bullet(doc, "Trip-local graphs prevent global road-arc classes. The shared binary candidate-label task ranks arcs only within each trip.")
    add_bullet(doc, "The exact-sequence rate remained zero for the final route on all three evaluation stages, even when pointwise scores improved.")
    add_bullet(doc, "Student-test labels were used to choose among decoder settings. Those scores are development results; only the separate evaluator run is reserved final evaluation.")
    add_bullet(doc, "The final evaluator set contains 10 trips. Its aggregate is a small sample and should not be generalized beyond the provided partition without additional validation.")
    doc.add_paragraph(
        "The selected HGB model is a practical choice because it was close to Random Forest in grouped CV, scored slightly better on the student route metric, produced lower candidate log loss, and fit much faster in the recorded comparison. Bayesian tuning made a small further improvement. HMM/Viterbi produced the clearest aggregate gain, but it did not fix candidate omissions or every route label."
    )
    doc.add_paragraph(
        "Reproduction commands, model scope and data contract are in README.md and docs/. Machine-readable evidence is in results/model_comparison.csv, results/experiment_results.json, results/bayesian_tuning.json, results/decoder_comparison.json and results/final_evaluator.json. The dataset archives are external and must not be committed. The local Git repository has no GitHub remote yet; public publication remains a team action after review."
    )
    doc.add_paragraph(
        "AI tools assisted with parts of implementation and report drafting. Team members should verify the figures and interpretation, revise the report in their own voice where needed, and follow the event's AI-assistance disclosure rules before final submission."
    )

    out = REPORTS / "PathSnap_Technical_Report.docx"
    REPORTS.mkdir(exist_ok=True)
    doc.core_properties.title = "PathSnap Technical Report"
    doc.core_properties.subject = "Trip-level GPS map matching experiments and evaluation"
    doc.save(out)
    print(out)


def short_config(name: str, config: dict) -> str:
    if name == "logistic_regression":
        return f"C={config['C']}; iter={config['max_iter']}"
    if name == "distance_weighted_knn":
        return f"k={config['n_neighbors']}; cap={config['train_row_cap']}"
    if name == "random_forest":
        return f"trees={config['n_estimators']}; leaf={config['min_samples_leaf']}"
    if name == "extra_trees":
        return f"trees={config['n_estimators']}; leaf={config['min_samples_leaf']}"
    if name == "rbf_svm":
        return f"C={config['C']}; cap={config['train_row_cap']}"
    if name == "mlp_shallow":
        return "64 units; alpha=.001; max 150"
    if name == "mlp_deep":
        return "128-64-32; alpha=.001; max 150"
    return str(config)


if __name__ == "__main__":
    main()
