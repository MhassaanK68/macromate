"""Generate figures and a readable report from the three experiment files."""

import argparse
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import FancyBboxPatch

from .config import ROOT, EVIDENCE, read_record, read_predictions, sha256
from .evaluate import subgroup_metrics
from .metrics import classification_metrics

FIGURES = ROOT / "figures" / "model"

def architecture_figure(figures):
    """Draw the actual frozen/trainable paths without requiring model loading."""
    fig, ax = plt.subplots(figsize=(11.6, 5.5), layout="constrained")
    ax.set(xlim=(0, 12), ylim=(0, 6))
    ax.axis("off")

    def box(x, y, width, height, title, detail, trainable=False):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.04",
                                   facecolor="#e4efea" if trainable else "#f1f3f5",
                                   edgecolor="#36705a" if trainable else "#59616b", linewidth=1.2))
        ax.text(x + width / 2, y + height * 0.72, title, ha="center", va="center",
                fontsize=10, weight="bold")
        ax.text(x + width / 2, y + height * 0.32, detail, ha="center", va="center", fontsize=9)

    def arrow(start, end):
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", color="#404750", lw=1.4))

    box(0.1, 3.75, 2.0, 1.25, "Meal photograph", "EXIF / RGB / crop\n224 x 224 x 3")
    box(2.65, 3.75, 2.2, 1.25, "Patch embedding", "32 x 32 patches\n49 patches + CLS; d=768")
    box(5.4, 3.75, 2.6, 1.25, "Frozen vision blocks", "Blocks 0-9; 12 heads\nHidden width 768", False)
    box(8.6, 3.75, 3.1, 1.25, "Adapted vision path", "Blocks 10-11 + final LN\n768 -> 512 projection", True)
    arrow((2.15, 4.38), (2.6, 4.38))
    arrow((4.9, 4.38), (5.35, 4.38))
    arrow((8.05, 4.38), (8.55, 4.38))
    box(0.1, 1.55, 2.75, 1.25, "Eight fixed prompts", '"a photo of Pakistani\n[dish]"')
    box(3.4, 1.55, 3.0, 1.25, "Frozen text encoder", "12 blocks; d=512; 8 heads\nPrecompute 8 x 512 vectors")
    box(7.0, 1.55, 2.05, 1.25, "CLIP comparison", "L2-normalize both\nScaled dot products")
    box(9.6, 1.55, 2.1, 1.25, "Eight dish logits", "Train: cross-entropy\nInfer: softmax + rank")
    arrow((2.9, 2.18), (3.35, 2.18))
    arrow((6.45, 2.18), (6.95, 2.18))
    arrow((9.1, 2.18), (9.55, 2.18))
    arrow((10.15, 3.7), (8.0, 2.85))
    ax.text(9.35, 3.18, "512-d image vector", ha="center", fontsize=9)
    ax.text(6, 5.65, "Macromate: partially fine-tuned CLIP ViT-B/32", ha="center",
            fontsize=15, weight="bold")
    ax.text(6, 0.65, "Green: 14,570,496 trainable parameters. Gray: fixed operations / frozen weights.\n"
            "The text features and CLIP logit scale stay fixed; no new classification head is added.",
            ha="center", va="center", fontsize=10)
    fig.savefig(figures / "architecture.png", dpi=190)
    plt.close(fig)


def load_results(evidence):
    evidence = Path(evidence)
    training = read_record(evidence / "training.json")
    evaluation = read_record(evidence / "evaluation.json")
    if training["status"] != "completed" or evaluation["status"] != "completed":
        raise ValueError("Training and evaluation must be completed before reporting")
    if sha256(evidence / "predictions.csv") != evaluation["predictions_sha256"]:
        raise ValueError("Saved predictions changed after evaluation")
    protocol, labels = evaluation["protocol"], training["labels"]
    verification = training.get("verification") or {}
    if (protocol["checkpoint_sha256"] != training["checkpoint"]["sha256"]
            or protocol["checkpoint_path"] != training["checkpoint"]["path"]
            or protocol["selected_epoch"] != training["result"]["best_epoch"]
            or protocol["lineage"] != training["lineage"] or protocol["labels"] != labels
            or verification.get("status") != "passed"
            or verification.get("checkpoint_sha256") != training["checkpoint"]["sha256"]):
        raise ValueError("Training and evaluation identify different models or datasets")
    rows = read_predictions(evidence / "predictions.csv")
    if (len(rows) != protocol["test_count"] or len({r["image_id"] for r in rows}) != len(rows)
            or {r["image_id"] for r in rows} != set(protocol["test_pixels_sha256"])):
        raise ValueError("Predictions must cover every held-out image exactly once")
    for row in rows:
        truth, predicted = row["true_label_id"], row["predicted_label_id"]
        if not 0 <= truth < len(labels) or not 0 <= predicted < len(labels):
            raise ValueError("Prediction contains an invalid class ID")
        scores, logits = np.asarray(row["scores"], dtype=float), np.asarray(row["logits"], dtype=float)
        if (scores.shape != (len(labels),) or logits.shape != scores.shape
                or not np.isfinite(logits).all()):
            raise ValueError("Prediction must contain finite class-ordered scores and logits")
        ranked = np.argsort(-scores, kind="stable").tolist()
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        if (not np.allclose(probabilities, scores, atol=1e-12, rtol=0)
                or row["ranked_label_ids"] != ranked
                or row["ranked_dishes"] != [labels[i] for i in ranked]
                or row["true_dish"] != labels[truth] or row["predicted_dish"] != labels[predicted]
                or predicted != ranked[0] or row["correct"] != (truth == predicted)
                or row["true_rank"] != ranked.index(truth) + 1):
            raise ValueError("Prediction labels, ranking, scores and logits disagree")
    metrics = classification_metrics([r["true_label_id"] for r in rows], [r["scores"] for r in rows], labels)
    if metrics != evaluation["metrics"] or subgroup_metrics(rows) != evaluation["subgroups"]:
        raise ValueError("Metrics or subgroups do not reproduce saved predictions")
    return training, evaluation, rows


def make_figures(evidence, rows, metrics, figures):
    training = read_record(evidence / "training.json")
    figures.mkdir(parents=True, exist_ok=True)
    architecture_figure(figures)
    history = training["history"]
    result = training["result"]
    epochs = [h["epoch"] for h in history]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    for subset, label in (("train", "Training (280 photos)"), ("validation", "Validation (64 photos)")):
        axes[0].plot(epochs, [h[subset]["loss"] for h in history], marker="o", label=label)
        axes[1].plot(epochs, [100 * h[subset]["top1_accuracy"] for h in history], marker="o", label=label)
    for ax in axes:
        ax.axvline(result["best_epoch"], color="black", linestyle="--", linewidth=1, label="Selected epoch")
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)
    axes[0].set(title="Training and validation loss", ylabel="Cross-entropy (mean per photo)")
    axes[1].set(title="Training and validation top-1", ylabel="Correct first choice (%)", ylim=(0, 102))
    fig.savefig(figures / "learning_curves.png", dpi=180)
    plt.close(fig)

    matrix = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8.5, 6.8), layout="constrained")
    chart = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(7, int(matrix.max())))
    labels = metrics["labels"]
    ax.set(xticks=range(8), yticks=range(8), xticklabels=labels, yticklabels=labels,
           xlabel="Predicted dish", ylabel="True dish", title="Held-out confusion matrix (56 photos)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(8):
        for j in range(8):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] >= 4 else "black")
    fig.colorbar(chart, ax=ax, label="Number of test photos", ticks=range(8))
    fig.savefig(figures / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.3), layout="constrained")
    for curve in metrics["roc_curves"]:
        if curve["auc"] is not None:
            ax.plot(curve["fpr"], curve["tpr"], label=f"{curve['dish']} (AUC {curve['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Chance ranking")
    ax.set(xlabel="False positive rate (fraction)", ylabel="True positive rate (fraction)",
           title="Held-out one-vs-rest ROC (7 positives per dish)", xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="lower right")
    fig.savefig(figures / "roc_curves.png", dpi=180)
    plt.close(fig)

    errors = [r for r in rows if not r["correct"]]
    if errors:
        columns = min(3, len(errors))
        fig, axes = plt.subplots(math.ceil(len(errors) / columns), columns,
                                 figsize=(4.2 * columns, 4.6 * math.ceil(len(errors) / columns)),
                                 squeeze=False, layout="constrained")
        for i, ax in enumerate(axes.flat):
            if i >= len(errors):
                ax.set_visible(False)
                continue
            row = errors[i]
            with Image.open(ROOT / row["local_path"]) as photo:
                ax.imshow(photo)
            ax.set_title(f"E{i + 1}: {row['true_dish']} -> {row['predicted_dish']}\n"
                         f"True rank {row['true_rank']}; {row['source_id']}", fontsize=10)
            ax.set_xlabel(f"{row['image_id']}\n{row['author']} / {row['license']}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("All held-out top-1 errors (original exported photos)", fontsize=13)
        fig.savefig(figures / "errors.png", dpi=180)
        plt.close(fig)
    else:
        (figures / "errors.png").unlink(missing_ok=True)


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
                    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(c, c) for c in str(value))


class Report:
    """Render the same report to Markdown and optional LaTeX source."""

    def __init__(self, output, figures):
        self.output, self.figures = Path(output), Path(figures)
        self.md = ["# Macromate: Deliverable 2", "", "Model Architecture and Empirical Evaluation", ""]
        self.tex = [r"\documentclass[10pt,a4paper]{article}", r"\usepackage[margin=0.78in]{geometry}",
                    r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern,graphicx,booktabs,tabularx,xurl}",
                    r"\usepackage{array,microtype,needspace}", r"\usepackage[hidelinks]{hyperref}",
                    r"\setlength{\parindent}{0pt}",
                    r"\setlength{\parskip}{6pt}", r"\setlength{\emergencystretch}{3em}",
                    r"\begin{document}",
                    r"{\LARGE\bfseries Macromate: Deliverable 2\par}",
                    r"{\large Model Architecture and Empirical Evaluation\par}",
                    r"Hassan Mehmood (24L-2559), Mohid Israr (24L-2510), Adnan Zaka (24i-2537)\par",
                    r"\smallskip\hrule\medskip"]

    def section(self, title):
        self.md.extend(["## " + title, ""])
        self.tex.append(r"\Needspace{5\baselineskip}\section*{" + tex_escape(title) + "}")

    def paragraph(self, text):
        self.md.extend([text, ""])
        self.tex.append(tex_escape(text) + "\n")

    def path(self, title, value):
        self.md.extend([f"{title}: `{value}`", ""])
        self.tex.append(tex_escape(title) + r": \path{" + value + r"}\par")

    def link(self, title, value):
        self.md.extend([f"[{title}]({value})", ""])
        self.tex.append(tex_escape(title) + r": \url{" + value + r"}\par")

    def table(self, headers, rows, text_columns=False):
        self.md.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
        self.md.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
        self.md.append("")
        columns = (r">{\raggedright\arraybackslash}X" * len(headers)
                   if text_columns else "X" + "r" * (len(headers) - 1))
        self.tex.extend([r"\begin{center}\footnotesize\setlength{\tabcolsep}{4pt}",
                         r"\begin{tabularx}{\linewidth}{" + columns + "}",
                         r"\toprule", " & ".join(tex_escape(x) for x in headers) + r" \\", r"\midrule"])
        self.tex.extend(" & ".join(tex_escape(v) for v in row) + r" \\" for row in rows)
        self.tex.append(r"\bottomrule\end{tabularx}\end{center}")

    def figure(self, name, caption):
        path = (self.figures / (name + ".png")).resolve()
        relative = Path(os.path.relpath(path, self.output)).as_posix()
        self.md.extend([f"![{caption}]({relative})", ""])
        tex_path = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
        self.tex.extend([r"\Needspace{0.36\textheight}\begin{center}",
                         r"\includegraphics[width=0.94\linewidth,height=0.43\textheight,keepaspectratio]{"
                         + tex_path + "}", r"\par\small " + tex_escape(caption), r"\end{center}"])

    def save(self):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "results.md").write_text("\n".join(self.md).rstrip() + "\n", encoding="utf-8", newline="\n")
        (self.output / "model_report.tex").write_text("\n".join(self.tex + [r"\end{document}"]) + "\n",
                                                     encoding="utf-8", newline="\n")


def make_report(training, evaluation, rows, output, figures):
    metrics, result = evaluation["metrics"], training["result"]
    report = Report(output, figures)
    report.paragraph("Macromate recognizes eight Pakistani dish categories from photos using PyTorch and "
                     "a partially fine-tuned pretrained CLIP ViT-B/32. The 400-photo dataset has 50 photos "
                     "per dish. The selected model correctly recognizes 54 of 56 held-out photos "
                     "(96.43%); this small, source-confounded sample does not establish real-world performance.")
    report.paragraph("Evidence consists of training.json (settings, learning history and verification), "
                     "predictions.csv (every held-out prediction), and evaluation.json (protocol, metrics, reviews and model details). "
                     "This report recomputes metrics from the CSV and does not run test inference.")
    report.section("Current implementation")
    report.paragraph("The maintained implementation is in models/; model/ is a compatibility package for "
                     "historical commands. The models package contains nine Python files, including "
                     "__init__.py as its package marker. "
                     "config.py owns settings and evidence-file reading/writing; data.py owns dataset validation, "
                     "image preprocessing and loaders; clip_classifier.py owns CLIP and checkpoint saving/loading; "
                     "train.py owns training, epoch selection and checkpoint verification.")
    report.paragraph("evaluate.py runs held-out evaluation, metrics.py implements numerical measurements, "
                     "infer.py predicts dishes for new photos, and report.py builds this report from saved results. "
                     "Grouping these responsibilities preserved the architecture, checkpoint format and measured results.")
    report.paragraph("The development workflow is train, evaluate, then report. Training includes checkpoint verification. "
                     "Standalone verification uses python -m models.train --verify-only with the experiment's --evidence "
                     "directory; it does not retrain or score held-out photos. Inference needs best.pt and the pinned "
                     "runs/hub base cache, without reading the evidence files or importing training.")
    report.paragraph("Historical evidence retains the source-file identities and hashes recorded for the original run. "
                     "The consolidated training and evaluation records retain pilot selection and evaluation-correction "
                     "provenance. Active model commands do not require the earlier phase-specific files or an archive.")
    report.section("Phase 3: model architecture")
    report.figure("architecture", "Actual CLIP inference and fine-tuning paths; green modules receive gradients.")
    report.paragraph("The vision encoder embeds 32-by-32 patches from a 224-by-224 RGB input into "
                     "49 patch tokens plus a class token (hidden width 768). Its 12 Transformer blocks "
                     "use 12 attention heads and a 3072-wide MLP. The class-token output passes through "
                     "final layer normalization and a 768-to-512 visual projection. The frozen text encoder "
                     "uses 12 blocks, hidden width 512 and eight attention heads; its projected prompt "
                     "vectors are computed once and stored in the checkpoint.")
    report.paragraph("CLIP ViT-B/32 compares each photo with eight fixed 'a photo of Pakistani [dish]' text vectors. "
                     "Only vision blocks 10 and 11, final vision normalization and visual projection are trained. "
                     "Training uses cross-entropy, AdamW, warmup/cosine decay and gradient clipping. The pinned base revision is "
                     + training["base_model"]["revision"] + ".")
    report.paragraph("For each class c, the logit is exp(logit_scale) times the dot product of the "
                     "unit-length image and prompt vectors. Categorical cross-entropy compares all eight "
                     "logits with the integer dish label; there are no class weights or label smoothing. "
                     "Only 14,570,496 parameters are trainable. AdamW uses separate block and "
                     "normalization/projection learning rates; biases and normalization parameters "
                     "receive zero weight decay. Gradients are clipped to norm 1.0; learning rates use "
                     "10% warmup followed by cosine decay. Dropout remains at the pretrained value zero.")
    report.section("Hyperparameter tuning")
    pilots = training.get("provenance", {}).get("pilot_results", [])
    if pilots:
        report.table(["Candidate", "Block LR", "Proj. LR", "Batch", "Decay", "Dropout",
                      "Val. correct", "Val. loss"],
                     [[p["run_id"], f'{p["settings"]["block_lr"]:.0e}',
                       f'{p["settings"]["projection_lr"]:.0e}', p["settings"]["batch_size"],
                       p["settings"]["weight_decay"], p["settings"]["dropout"],
                       f'{p["validation"]["top1_correct"]}/{p["validation"]["total"]}',
                       f'{p["validation"]["loss"]:.6f}'] for p in pilots])
        selection = training["provenance"]["pilot_selection"]
        report.paragraph("Six exploratory candidates were trained for two epochs each on the same "
                         "280 training and 64 validation photos; test photos were excluded. Selection "
                         "prioritized validation correct count, then loss, then runtime for exact ties. "
                         "The selected candidate was " + selection["run_id"] + ": "
                         "58/64 correct with the lowest tied-count validation loss (0.282456). "
                         "This favors block LR 2e-5, projection LR 5e-5, batch 8 and decay 0.01. "
                         "Batch 16 and smaller learning rates did not improve this pilot.")
        report.paragraph("Dropout 0, maximum 20 epochs, patience 5, seed 42, fixed prompts, "
                         "optimizer and scheduler are design choices, not demonstrated tuning winners. "
                         "The pilot uses one seed and a short budget; one validation photo changes accuracy "
                         "by 1.5625 percentage points. The six candidates do not establish a global optimum.")
    else:
        report.paragraph("No pilot comparison is present for this experiment; the settings below "
                         "must be treated as configured choices rather than empirically tuned winners.")
    report.section("Phase 4: training and validation")
    report.table(["Setting", "Value"], [[key, value] for key, value in training["settings"].items()])
    counts = training["counts"]
    report.paragraph(f"Fixed stratified, group-aware split: {counts['train']} training, {counts['val']} validation, "
                     f"{counts['test']} test photos. Training updates weights; validation selects the epoch; test images "
                     "measure the selected model. Whole-image allocation gives 70%/16%/14% "
                     "(35/8/7 photos per dish). Seed 42 fixes membership; exact/perceptual duplicate "
                     "groups stay in one partition. The held-out results were not used for tuning.")
    report.paragraph(f"Completed {result['epochs_completed']} epochs; selected epoch {result['best_epoch']} by validation "
                     f"top-1 count, then lower loss, then earlier epoch. Validation top-1: "
                     f"{result['validation']['top1_correct']}/{result['validation']['total']}; "
                     f"loss {result['validation']['loss']:.6f}. Training time: {result['seconds']:.1f} seconds.")
    verified = training["verification"]
    report.paragraph(f"Checkpoint reconstruction verified {verified['validation_images']} validation rows; "
                     f"maximum absolute logit difference {verified['reload_max_error']}. Original-photo preprocessing "
                     "and the app-facing prediction were also checked. Verification status: " + verified["status"] + ".")
    env = training["environment"]
    report.paragraph(f"Recorded training environment: Python {env['python']}, {env['device_name']}, "
                     f"PyTorch {env['packages']['torch']}, precision {env['precision']}.")
    report.figure("learning_curves", "Training and validation loss/accuracy; dashed line marks the selected epoch.")
    report.section("Held-out evaluation")
    report.paragraph(f"Top-1: {metrics['top1_correct']}/{metrics['total']} ({100 * metrics['top1_accuracy']:.2f}%). "
                     f"Top-3: {metrics['top3_correct']}/{metrics['total']} ({100 * metrics['top3_accuracy']:.2f}%). "
                     f"Macro F1: {metrics['macro']['f1']:.4f}.")
    frozen = evaluation["protocol"]["frozen_at_utc"]
    frozen_display = frozen[:10] + " at " + frozen[11:19] + " UTC"
    report.paragraph("The checkpoint, data hashes and evaluation rules were recorded before inference on "
                     + frozen_display + ". Evaluation attempts, including failures, are retained; "
                     "repeat evaluation in the same directory is refused. Figures and metrics reuse saved predictions.")
    decimal = lambda x: "undefined" if x is None else f"{x:.4f}"
    report.table(["Dish", "Support", "Precision", "Recall", "F1", "ROC-AUC"],
                 [[r["dish"], r["support"], decimal(r["precision"]), decimal(r["recall"]), decimal(r["f1"]),
                   decimal(r["roc_auc"])] for r in metrics["per_class"]] +
                 [[name, metrics["total"], decimal(metrics[key]["precision"]), decimal(metrics[key]["recall"]),
                   decimal(metrics[key]["f1"]), decimal(metrics[key]["roc_auc"])]
                  for name, key in (("Macro", "macro"), ("Weighted", "weighted"))])
    report.paragraph("Precision is TP/(TP+FP), recall is TP/(TP+FN), and F1 is their harmonic mean. "
                     "Macro averages weight dishes equally; weighted averages use class support. ROC-AUC measures "
                     "one-vs-rest score ranking with tied scores grouped. Scores are uncalibrated. Confusion rows are true labels.")
    report.paragraph("Precision measures incorrect dish suggestions, recall measures missed instances of a dish, "
                     "and F1 summarizes that tradeoff. Macro averaging gives each dish equal importance and "
                     "would expose weak classes even if support became imbalanced. Here every class has seven "
                     "test photos, so macro and support-weighted means coincide. One-vs-rest ROC-AUC checks "
                     "score ranking across thresholds and complements the single top-1 decision; high AUC "
                     "does not imply calibrated probabilities, reliable unfamiliar-food rejection, or fairness.")
    report.figure("confusion_matrix", "Held-out confusion counts: rows=true dish, columns=predicted dish.")
    report.figure("roc_curves", "One-vs-rest ROC curves from saved full-precision scores.")
    report.section("Error analysis")
    errors = [r for r in rows if not r["correct"]]
    reviews = {r["image_id"]: r for r in evaluation.get("error_reviews", [])}
    if errors:
        report.figure("errors", "Every incorrect prediction, with source attribution.")
    else:
        report.paragraph("No top-1 errors occurred on this small held-out set; broader generalization is untested.")
    for row in errors:
        review = reviews.get(row["image_id"], {})
        observation = (review.get("report_summary") or "Visual review pending.").replace(
            "error_review.json", "evaluation.json")
        report.paragraph(f"{row['image_id']}: {row['true_dish']} predicted as {row['predicted_dish']}; "
                         f"true rank {row['true_rank']}. {observation} Source: {row['source_url']}")
    if evaluation.get("provenance", {}).get("rerun"):
        report.paragraph("Historical evaluation correction: " + evaluation["provenance"]["rerun"]["reason"])
    report.section("Bias Check and limitations")
    for field, title in (("source_id", "Source"), ("original_size", "Original image size")):
        report.table([title, "Correct / total", "Top-1"],
                     [[r["group"], f"{r['correct']}/{r['total']}", f"{100 * r['accuracy']:.2f}%"]
                      for r in evaluation["subgroups"][field]])
    report.paragraph("The test set contains only seven photos per dish. Source, dish and provider image size are "
                     "confounded, so subgroup differences do not establish causes. The sources largely represent Indian "
                     "and Bangladeshi appearances; external Pakistani meals remain untested. Second human label review "
                     "is pending. Demographic data are unavailable. Mixed plates and unfamiliar foods are unsupported.")
    report.section("Final model and application use")
    report.path("Checkpoint", training["checkpoint"]["path"])
    report.path("Checkpoint SHA-256", training["checkpoint"]["sha256"])
    report.paragraph("The adapted-weight checkpoint requires the pinned runs/hub base cache. "
                     "Recognizer(checkpoint, device).predict(photo) returns eight ranked dishes. Ask the user to confirm "
                     "the dish and enter serving grams before applying the separate per-100-g nutrition reference.")
    report.paragraph("The working prototype includes preparation, training, evaluation, reports and "
                     "CLI/Python inference. A graphical interface, serving-size workflow, nutrition "
                     "integration and daily log are future work. The recognizer returns ranked dishes only.")
    report.section("Contribution Table: verified Git history")
    contribution_path = ROOT / "docs" / "contributions.json"
    if contribution_path.is_file():
        import json
        contributions = json.loads(contribution_path.read_text(encoding="utf-8"))
        report.paragraph("The table maps actual commit authors to changed files in the repository history "
                         "at " + contributions["head"][:12] + ". It records evidence of authorship, "
                         "not an independently verified division of effort.")
        report.table(["Student / Git author", "Commit evidence", "Modules / coding status"],
                     [[m["student"] + " / " + m["git_author"], m["commit_evidence"],
                       m["modules"] + ". " + m["coding_status"]] for m in contributions["members"]],
                     text_columns=True)
        missing = [m["student"] for m in contributions["members"] if not m["coding_commits"]]
        if missing:
            report.paragraph("No coding commit was found for " + ", ".join(missing) + ". "
                             "The visible contribution changes source photos and CSV manifests. This "
                             "does not satisfy the guide's individual coding requirement. Each member "
                             "must commit their own real implementation work through Git CLI or a "
                             "desktop client. Existing shared history and author identities are preserved.")
        else:
            report.paragraph("Every listed member has a coding commit in the audited history. "
                             "The recorded modules and commit authors provide the contribution evidence.")
        if contributions.get("manual_upload_indicators"):
            report.paragraph("Historical commit(s) "
                             + ", ".join(contributions["manual_upload_indicators"])
                             + " are titled 'Add files via upload'. Their presence is disclosed because "
                             "the guide prohibits manual GitHub uploads. The historical workflow needs "
                             "clarification with the instructor; a new CLI commit does not erase a prior "
                             "policy issue.")
    else:
        report.paragraph("Git contribution snapshot is unavailable. Run scripts/collect_contributions.py "
                         "inside the repository before preparing a submission report.")
    report.section("Reproduction and submission assets")
    report.paragraph("Submission checks identified a PNG-encoding dependency: preprocessing with Pillow "
                     "11.3.0 changes the compressed bytes of some files even though all 400 decoded RGB "
                     "arrays are identical. Pillow 13.0.0 recreates every original prepared PNG hash "
                     "exactly. The submission requirements therefore pin Pillow 13.0.0; the original "
                     "training environment recorded above remains unchanged in the historical evidence.")
    report.paragraph("Install Python 3.12, the appropriate torch 2.9.0 CPU/CUDA wheel, and "
                     "requirements.txt. Download the pinned pretrained base model using the README "
                     "command, restore runs/phase4/best.pt, and verify its SHA-256 above. "
                     "Run python -m models.infer CHECKPOINT PHOTO --device cpu for a new photo; "
                     "python -m models.report regenerates this report from saved predictions.")
    release_path = ROOT / "docs/weights_release.json"
    if release_path.is_file():
        import json
        release = json.loads(release_path.read_text(encoding="utf-8"))
        report.link("Download evaluated model weights", release["assets"][0]["url"])
        report.paragraph("The 58,351,801-byte checkpoint is distributed in a verified ZIP through "
                         "GitHub Releases, outside Git history. Its SHA-256 is listed above; the ZIP "
                         "also includes a manifest, installation instructions and separate checksums.")
    else:
        report.paragraph("The local weights are 58,351,801 bytes and remain outside Git. A reviewer-accessible "
                         "external download URL must be added to README.md before submission.")
    validation_path = ROOT / "docs/validation_results.json"
    if validation_path.is_file():
        import json
        checks = json.loads(validation_path.read_text(encoding="utf-8"))
        report.section("Software verification")
        report.paragraph(f"{checks['unit_tests']['passed']} automated tests passed. The complete data pipeline "
                         "ran twice with 424 identical output files, and all 400 prepared image hashes match "
                         "the recorded experiment. Dependency checks pass. A separate one-epoch GPU training "
                         "smoke test completed, saved a checkpoint and independently reproduced all 64 "
                         "validation rows with zero logit difference. This smoke run did not replace the "
                         "evaluated model or use held-out inference.")
        report.paragraph("A fresh reconstruction of the original checkpoint on the recorded GPU also "
                         "reproduced every validation logit exactly. CPU CLI inference returned all eight "
                         "ranked dishes and matched the recorded single-image GPU output within 1e-4.")
        comparison = checks["cpu_validation_comparison"]
        report.paragraph(f"Portability limit: full batched CPU replay agrees with the GPU top-1 label "
                         f"on {comparison['top1_agreement']}/{comparison['images']} validation photos, "
                         f"but its maximum logit difference is {comparison['max_logit_difference']:.6f}, "
                         "exceeding the existing strict cross-device threshold of 1e-4. The strict "
                         "verifier correctly rejects that numerical replay. Its tolerance has not been "
                         "relaxed; use the recorded GPU device for exact reproduction. CPU inference "
                         "remains operational. Original evaluation evidence and weights are unchanged.")
    report.paragraph("The submission checklist records software checks and unresolved collaboration "
                     "requirements separately from the measured experiment.")
    report.section("References")
    report.link("Radford et al. (2021), Learning Transferable Visual Models From Natural Language Supervision",
                "https://arxiv.org/abs/2103.00020")
    report.paragraph("Architecture settings are taken from the pinned CLIP config.json and the implementation "
                     "in models/clip_classifier.py. All reported experiment measurements come from the "
                     "three local evidence files; the pretrained CLIP architecture is adapted rather than "
                     "introduced as a new model in this project.")
    report.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--output", type=Path, default=ROOT / "docs")
    parser.add_argument("--figures", type=Path, default=FIGURES)
    args = parser.parse_args()
    training, evaluation, rows = load_results(args.evidence)
    make_figures(args.evidence, rows, evaluation["metrics"], args.figures)
    make_report(training, evaluation, rows, args.output, args.figures)
    print("Verified saved metrics; generated figures, results.md and model_report.tex. No inference repeated.")


if __name__ == "__main__":
    main()
