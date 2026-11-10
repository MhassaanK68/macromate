"""Classification metrics for training, evaluation, and saved-result checks."""

import torch


def summarize(logits, labels, loss_sum):
    if not len(labels) or not torch.isfinite(logits).all():
        raise ValueError("Metrics require nonempty, finite logits")
    ranked = logits.argsort(dim=1, descending=True, stable=True)
    correct = ranked[:, 0].eq(labels)
    top3 = ranked[:, :3].eq(labels[:, None]).any(dim=1)
    return {
        "loss": loss_sum / len(labels), "total": len(labels),
        "top1_correct": int(correct.sum()), "top3_correct": int(top3.sum()),
        "top1_accuracy": float(correct.double().mean()),
        "top3_accuracy": float(top3.double().mean()),
        "per_class": [{"label_id": i, "total": int(labels.eq(i).sum()),
                       "correct": int((correct & labels.eq(i)).sum())} for i in range(8)],
    }


def selection_key(metrics):
    return metrics["top1_correct"], -metrics["loss"]


def binary_roc(targets, scores):
    """Empirical ROC; equal scores share one threshold and receive half credit."""
    import numpy as np

    targets = np.asarray(targets, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    if targets.ndim != 1 or scores.shape != targets.shape or not np.isfinite(scores).all():
        raise ValueError("ROC needs equal-length, finite one-dimensional inputs")
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if not positives or not negatives:
        return {"auc": None, "reason": "No positives or no negatives", "positives": positives,
                "negatives": negatives, "fpr": [], "tpr": [], "thresholds": []}
    order = np.argsort(-scores, kind="stable")
    sorted_scores, truth = scores[order], targets[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    true_counts = np.cumsum(truth)[ends]
    false_counts = 1 + ends - true_counts
    tpr = np.r_[0.0, true_counts / positives]
    fpr = np.r_[0.0, false_counts / negatives]
    return {"auc": float(np.trapezoid(tpr, fpr)), "reason": None,
            "positives": positives, "negatives": negatives,
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "thresholds": [None] + sorted_scores[ends].tolist()}


def classification_metrics(targets, scores, labels):
    """Metrics from saved softmax scores in the checkpoint's class order."""
    import numpy as np

    targets = np.asarray(targets)
    scores = np.asarray(scores, dtype=float)
    count = len(labels)
    if (targets.ndim != 1 or not len(targets) or targets.dtype.kind not in "iu"
            or scores.shape != (len(targets), count) or count < 3
            or len(set(labels)) != count or not np.isfinite(scores).all()
            or np.any(scores < 0) or np.any(scores > 1)
            or not np.allclose(scores.sum(axis=1), 1, atol=1e-6, rtol=0)
            or np.any(targets < 0) or np.any(targets >= count)):
        raise ValueError("Invalid targets, class order or softmax scores")
    ranked = np.argsort(-scores, axis=1, kind="stable")
    predicted = ranked[:, 0]
    confusion = np.zeros((count, count), dtype=int)
    for truth, guess in zip(targets, predicted):
        confusion[truth, guess] += 1
    per_class, curves = [], []
    for i, label in enumerate(labels):
        tp = int(confusion[i, i])
        support = int(confusion[i].sum())
        predicted_count = int(confusion[:, i].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        curve = binary_roc(targets == i, scores[:, i])
        curves.append(dict(label_id=i, dish=label, **curve))
        per_class.append({"label_id": i, "dish": label, "true_positive": tp,
                          "false_positive": predicted_count - tp, "false_negative": support - tp,
                          "support": support, "predicted_count": predicted_count,
                          "precision": precision, "recall": recall, "f1": f1, "roc_auc": curve["auc"]})
    macro = {key: sum(row[key] for row in per_class) / count for key in ("precision", "recall", "f1")}
    weighted = {key: sum(row[key] * row["support"] for row in per_class) / len(targets)
                for key in ("precision", "recall", "f1")}
    aucs = [row["roc_auc"] for row in per_class]
    # Never silently average only the defined classes and call that full macro AUC.
    macro["roc_auc"] = sum(aucs) / count if all(x is not None for x in aucs) else None
    weighted["roc_auc"] = (sum(row["roc_auc"] * row["support"] for row in per_class) / len(targets)
                           if all(x is not None for x in aucs) else None)
    correct = int((predicted == targets).sum())
    top3 = int((ranked[:, :3] == targets[:, None]).any(axis=1).sum())
    return {"total": len(targets), "top1_correct": correct, "top3_correct": top3,
            "top1_accuracy": correct / len(targets), "top3_accuracy": top3 / len(targets),
            "labels": list(labels), "confusion_matrix": confusion.tolist(),
            "per_class": per_class, "macro": macro, "weighted": weighted, "roc_curves": curves}
