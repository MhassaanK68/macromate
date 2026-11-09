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

from .config import ROOT, EVIDENCE, read_record, read_predictions, sha256
from .evaluate import subgroup_metrics
from .metrics import classification_metrics

FIGURES = ROOT / "figures" / "model"


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
        self.md = ["# Macromate: training and evaluation", ""]
        self.tex = [r"\documentclass[11pt]{article}", r"\usepackage[margin=0.85in]{geometry}",
                    r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern,graphicx,booktabs,tabularx,xurl}",
                    r"\usepackage[hidelinks]{hyperref}", r"\setlength{\parindent}{0pt}",
                    r"\setlength{\parskip}{6pt}", r"\setlength{\emergencystretch}{3em}",
                    r"\begin{document}", r"\section*{Macromate: training and evaluation}"]

    def section(self, title):
        self.md.extend(["## " + title, ""])
        self.tex.append(r"\section*{" + tex_escape(title) + "}")

    def paragraph(self, text):
        self.md.extend([text, ""])
        self.tex.append(tex_escape(text) + "\n")

    def path(self, title, value):
        self.md.extend([f"{title}: `{value}`", ""])
        self.tex.append(tex_escape(title) + r": \path{" + value + r"}\par")

    def table(self, headers, rows):
        self.md.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
        self.md.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
        self.md.append("")
        self.tex.extend([r"\begin{center}\small", r"\begin{tabularx}{\linewidth}{X" + "r" * (len(headers) - 1) + "}",
                         r"\toprule", " & ".join(tex_escape(x) for x in headers) + r" \\", r"\midrule"])
        self.tex.extend(" & ".join(tex_escape(v) for v in row) + r" \\" for row in rows)
        self.tex.append(r"\bottomrule\end{tabularx}\end{center}")

    def figure(self, name, caption):
        path = (self.figures / (name + ".png")).resolve()
        relative = Path(os.path.relpath(path, self.output)).as_posix()
        self.md.extend([f"![{caption}]({relative})", ""])
        tex_path = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.as_posix()
        self.tex.extend([r"\begin{center}", r"\includegraphics[width=0.94\linewidth,height=0.48\textheight,keepaspectratio]{"
                         + tex_path + "}", r"\par\small " + tex_escape(caption), r"\end{center}"])

    def save(self):
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "results.md").write_text("\n".join(self.md) + "\n", encoding="utf-8", newline="\n")
        (self.output / "model_report.tex").write_text("\n".join(self.tex + [r"\end{document}"]) + "\n",
                                                     encoding="utf-8", newline="\n")


def make_report(training, evaluation, rows, output, figures):
    metrics, result = evaluation["metrics"], training["result"]
    report = Report(output, figures)
    report.paragraph("Evidence consists of training.json (settings, learning history and verification), "
                     "predictions.csv (every held-out prediction), and evaluation.json (protocol, metrics, reviews and model details). "
                     "This report recomputes metrics from the CSV and does not run test inference.")
    report.section("Current implementation")
    report.paragraph("The model package contains nine Python files, including __init__.py as its package marker. "
                     "config.py owns settings and evidence-file reading/writing; data.py owns dataset validation, "
                     "image preprocessing and loaders; clip_classifier.py owns CLIP and checkpoint saving/loading; "
                     "train.py owns training, epoch selection and checkpoint verification.")
    report.paragraph("evaluate.py runs held-out evaluation, metrics.py implements numerical measurements, "
                     "infer.py predicts dishes for new photos, and report.py builds this report from saved results. "
                     "Grouping these responsibilities preserved the architecture, checkpoint format and measured results.")
    report.paragraph("The development workflow is train, evaluate, then report. Training includes checkpoint verification. "
                     "Standalone verification uses python -m model.train --verify-only with the experiment's --evidence "
                     "directory; it does not retrain or score held-out photos. Inference needs best.pt and the pinned "
                     "runs/hub base cache, without reading the evidence files or importing training.")
    report.paragraph("Historical evidence retains the source-file identities and hashes recorded for the original run. "
                     "The consolidated training and evaluation records retain pilot selection and evaluation-correction "
                     "provenance. Active model commands do not require the earlier phase-specific files or an archive.")
    report.section("Model and training")
    report.paragraph("CLIP ViT-B/32 compares each photo with eight fixed 'a photo of Pakistani [dish]' text vectors. "
                     "Only vision blocks 10 and 11, final vision normalization and visual projection are trained. "
                     "Training uses cross-entropy, AdamW, warmup/cosine decay and gradient clipping. The pinned base revision is "
                     + training["base_model"]["revision"] + ".")
    report.table(["Setting", "Value"], [[key, value] for key, value in training["settings"].items()])
    counts = training["counts"]
    report.paragraph(f"Fixed stratified, group-aware split: {counts['train']} training, {counts['val']} validation, "
                     f"{counts['test']} test photos. Training updates weights; validation selects the epoch; test images "
                     "measure the selected model. The held-out results were not used for tuning.")
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
    report.paragraph("The checkpoint, data hashes and evaluation rules were recorded before inference at "
                     + evaluation["protocol"]["frozen_at_utc"] + ". Evaluation attempts, including failures, are retained; "
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
        observation = review.get("report_summary") or "Visual review pending."
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
