"""One locked held-out pass; reports and visual inspection reuse saved results."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from .clip_classifier import load_checkpoint, read_checkpoint
from .config import (ROOT, EVIDENCE, FORMAT_VERSION, DEFINITIONS, sha256, write_json,
                     code_hashes, read_record, resolve_checkpoint, reserve_record,
                     test_hashes, timestamp, write_predictions)
from .data import load_records, data_lineage, read_unique, image_tensor
from .metrics import classification_metrics


class HeldOutDataset(Dataset):
    """Only the evaluator can request test pixels; training datasets stay restricted."""

    def __init__(self, records):
        self.records = [r for r in records if r["split"] == "test"]
        if not self.records:
            raise ValueError("No held-out records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return image_tensor(row["path"]), row["label_id"], row["image_id"]


def normalized_label(value):
    """Match the Phase 2 raw-to-canonical label conversion exactly."""
    return (value or "").strip().lower().replace("_", " ")


def validate_metadata(records, labels, metadata, sources):
    for row in records:
        meta = metadata[row["image_id"]]
        if normalized_label(meta["dish_label"]) != labels[row["label_id"]]:
            raise ValueError(f"Metadata label mismatch for {row['image_id']}")
        if meta["source_id"] not in sources:
            raise ValueError(f"Unknown source for {row['image_id']}")


def evaluation_protocol(training, records, labels):
    checkpoint = resolve_checkpoint(training)
    verification = training.get("verification") or {}
    if (training["status"] != "completed" or verification.get("status") != "passed"
            or labels != training["labels"] or data_lineage(records) != training["lineage"]
            or sha256(checkpoint) != training["checkpoint"]["sha256"]
            or verification.get("checkpoint_sha256") != training["checkpoint"]["sha256"]):
        raise ValueError("Completed training, verification, checkpoint and current data must agree")
    payload = read_checkpoint(checkpoint)
    if (payload["labels"] != labels or payload["metadata"]["lineage"] != training["lineage"]
            or payload["metadata"]["settings"] != training["settings"]
            or payload["metadata"]["epoch"] != training["result"]["best_epoch"]):
        raise ValueError("Checkpoint metadata differs from training record")
    return {
        "frozen_at_utc": timestamp(), "checkpoint_path": training["checkpoint"]["path"],
        "checkpoint_sha256": training["checkpoint"]["sha256"],
        "revision": training["base_model"]["revision"], "selected_epoch": training["result"]["best_epoch"],
        "labels": labels, "lineage": training["lineage"],
        "metadata_sha256": sha256(ROOT / "data" / "metadata.csv"),
        "sources_sha256": sha256(ROOT / "data" / "sources.csv"),
        "test_pixels_sha256": test_hashes(records), "code_sha256": code_hashes(),
        "definitions": DEFINITIONS, "test_count": sum(r["split"] == "test" for r in records),
        "use": "One held-out pass; reuse predictions for analysis; no test-based tuning",
    }


def model_handoff(training, metrics, rows):
    confusions = defaultdict(int)
    for row in rows:
        if not row["correct"]:
            confusions[(row["true_dish"], row["predicted_dish"])] += 1
    return {
        "checkpoint_path": training["checkpoint"]["path"], "checkpoint_sha256": training["checkpoint"]["sha256"],
        "base_revision": training["base_model"]["revision"], "base_cache_required": True,
        "labels": training["labels"], "selected_epoch": training["result"]["best_epoch"],
        "preprocessing": training["preprocessing"], "api": "model.infer.Recognizer(checkpoint, device).predict(photo)",
        "output": "ranked_dishes: label_id, dish, logit, score", "scores_are_calibrated": False,
        "user_confirmation_required": True, "mixed_or_unfamiliar_meals_supported": False,
        "nutrition": "User-confirmed dish; user-entered grams; separate per-100g lookup",
        "test_metrics": {k: metrics[k] for k in ("total", "top1_correct", "top3_correct", "macro")},
        "observed_confusions": [{"true_dish": a, "predicted_dish": b, "count": n}
                                for (a, b), n in sorted(confusions.items())],
        "limits": ["Small held-out set", "Second human label review pending", "Source and label confounded",
                   "External Pakistani meal domain untested", "Demographic data unavailable", "Softmax uncalibrated"],
    }


def attach_metadata(image_ids, targets, logits, labels, metadata, sources):
    scores = logits.double().softmax(dim=1)
    rankings = scores.argsort(dim=1, descending=True, stable=True)
    rows = []
    for index, image_id in enumerate(image_ids):
        meta = metadata[image_id]
        if labels[targets[index]] != normalized_label(meta["dish_label"]):
            raise ValueError(f"Metadata label mismatch for {image_id}")
        source = sources[meta["source_id"]]
        ranked = rankings[index].tolist()
        predicted = ranked[0]
        rows.append({"image_id": image_id, "true_label_id": targets[index], "true_dish": labels[targets[index]],
                     "predicted_label_id": predicted, "predicted_dish": labels[predicted],
                     "correct": predicted == targets[index], "true_rank": ranked.index(targets[index]) + 1,
                     "ranked_label_ids": ranked, "ranked_dishes": [labels[i] for i in ranked],
                     "logits": logits[index].tolist(), "scores": scores[index].tolist(),
                     "source_id": meta["source_id"], "source_name": source["source_name"],
                     "original_size": meta["width"] + "x" + meta["height"],
                     "preprocessing_style": source["transformations"],
                     "image_transformations": meta["transformations"],
                     "original_path": meta["original_path"], "local_path": meta["local_path"],
                     "author": meta["author"], "license": meta["license"], "source_url": meta["source_url"]})
    return rows


def subgroup_metrics(rows):
    result = {}
    for field in ("true_dish", "source_id", "source_name", "original_size", "preprocessing_style"):
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        result[field] = [{"group": key, "total": len(group),
                          "correct": sum(row["correct"] for row in group),
                          "accuracy": sum(row["correct"] for row in group) / len(group)}
                         for key, group in sorted(groups.items())]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["source_id"], row["true_dish"])].append(row)
    result["source_by_dish"] = [{"source_id": source, "dish": dish, "total": len(group),
                                 "correct": sum(r["correct"] for r in group),
                                 "accuracy": sum(r["correct"] for r in group) / len(group)}
                                for (source, dish), group in sorted(groups.items())]
    return result


def evaluate(evidence=EVIDENCE, device="cpu"):
    evidence = Path(evidence)
    record_path = evidence / "evaluation.json"
    if record_path.exists() or (evidence / "predictions.csv").exists():
        raise ValueError("Evaluation already attempted; reuse saved predictions or choose a fresh experiment")
    training = read_record(evidence / "training.json")
    records, labels = load_records()
    protocol = evaluation_protocol(training, records, labels)
    metadata = read_unique(ROOT / "data" / "metadata.csv", ["image_id", "source_id", "dish_label"])
    with (ROOT / "data" / "sources.csv").open(newline="", encoding="utf-8-sig") as handle:
        sources = {row["source_id"]: row for row in csv.DictReader(handle)}
    validate_metadata(records, labels, metadata, sources)
    evaluation = {"format_version": FORMAT_VERSION, "status": "started", "protocol": protocol,
                  "started_at_utc": timestamp(), "device": device, "inference_passes": 0,
                  "test_based_tuning": False}
    # The same file contains the pre-inference lock, success or failure, and final results.
    reserve_record(record_path, evaluation)
    try:
        model, _ = load_checkpoint(resolve_checkpoint(training), device)
        torch.set_num_threads(4)
        loader = DataLoader(HeldOutDataset(records), batch_size=training["settings"]["batch_size"], num_workers=0)
        logits, targets, image_ids = [], [], []
        with torch.inference_mode():
            for pixels, truth, ids in loader:
                batch_logits = model(pixels.to(device)).cpu()
                if not torch.isfinite(batch_logits).all():
                    raise ValueError("Nonfinite held-out logits")
                logits.append(batch_logits)
                targets.extend(truth.tolist())
                image_ids.extend(ids)
        if len(image_ids) != protocol["test_count"] or len(set(image_ids)) != protocol["test_count"]:
            raise ValueError("Incorrect or duplicated test IDs")
        rows = attach_metadata(image_ids, targets, torch.cat(logits), labels, metadata, sources)
        metrics = classification_metrics(targets, [row["scores"] for row in rows], labels)
        write_predictions(evidence / "predictions.csv", rows)
        evaluation.update(
            status="completed", finished_at_utc=timestamp(), inference_passes=1,
            predictions_sha256=sha256(evidence / "predictions.csv"),
            metrics=metrics, subgroups=subgroup_metrics(rows), model=model_handoff(training, metrics, rows),
            error_reviews=[{"image_id": r["image_id"], "reviewed": False, "report_summary": ""}
                           for r in rows if not r["correct"]],
        )
        write_json(record_path, evaluation)
        print(f"Held-out top-1 {metrics['top1_correct']}/{metrics['total']}; top-3 {metrics['top3_correct']}/{metrics['total']}")
        return evaluation
    except Exception as error:
        evaluation.update(status="failed", failure={"at_utc": timestamp(), "error": repr(error)})
        write_json(record_path, evaluation)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate(args.evidence, args.device)


if __name__ == "__main__":
    main()
