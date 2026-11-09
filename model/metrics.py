"""Count-based metrics shared by training and validation."""

import torch


def summarize(logits, labels, loss_sum):
    if not len(labels) or not torch.isfinite(logits).all():
        raise ValueError("Metrics require nonempty, finite logits")
    ranked = logits.argsort(dim=1, descending=True)
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
