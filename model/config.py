"""Model settings and serialization of the three experiment evidence files."""

import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RUNS = ROOT / "runs"
CHECKPOINT = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
SIZE = 224
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULTS = {
    "block_lr": 2e-5, "projection_lr": 5e-5, "batch_size": 8,
    "weight_decay": 0.01, "epochs": 20, "patience": 5, "seed": 42,
    "warmup_fraction": 0.1, "gradient_clip": 1.0,
    "betas": [0.9, 0.999], "epsilon": 1e-8, "dropout": 0.0,
}

EVIDENCE = ROOT / "evidence"
FORMAT_VERSION = 1
CORE_FILES = ("config.py", "data.py", "clip_classifier.py", "infer.py",
              "metrics.py", "train.py", "evaluate.py", "report.py")
DEFINITIONS = {
    "top1": "First ranked class correct / total",
    "top3": "True class among first three ranked classes / total",
    "tie_order": "Descending score; ascending class ID for exact ties",
    "precision": "TP / (TP + FP); zero for zero denominator",
    "recall": "TP / (TP + FN); zero for zero denominator",
    "f1": "Harmonic mean of precision and recall; zero for zero denominator",
    "macro": "Unweighted mean over all eight classes",
    "weighted": "Mean weighted by true class support",
    "roc_auc": "One-vs-rest softmax ranking; equal scores grouped; empirical ROC area",
    "undefined_auc": "null when no positives or negatives; full macro AUC then null",
    "confusion_matrix": "Rows=true class; columns=predicted class",
    "scores": "Uncalibrated softmax ranking over eight supported dishes",
    "subgroups": "Descriptive accuracy by source, dish, size, transformation, source x dish",
}
LIST_FIELDS = {"logits", "scores", "ranked_label_ids", "ranked_dishes"}
INT_FIELDS = {"true_label_id", "predicted_label_id", "true_rank"}


def read_labels(path=PROCESSED / "label_map.json"):
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = mapping["id_to_label"]
    if set(ids) != {str(i) for i in range(8)}:
        raise ValueError("label_map must contain exactly class IDs 0 through 7")
    labels = [ids[str(i)] for i in range(8)]
    if len(set(labels)) != 8 or any(not isinstance(x, str) or not x for x in labels):
        raise ValueError("label_map needs eight unique nonempty dish names")
    if mapping["label_to_id"] != {label: i for i, label in enumerate(labels)}:
        raise ValueError("label_map directions disagree")
    return labels


def prompts_for(labels):
    return ["a photo of Pakistani " + label for label in labels]


def preprocessing():
    return {"size": SIZE, "mean": list(MEAN), "std": list(STD),
            "prepared": "RGB PNG; scale /255 and normalize once; no resize/crop",
            "upload": "EXIF transpose, RGB, bicubic shorter side, centre crop"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, allow_nan=False) + "\n"
    # Replace complete records atomically so an interrupted write cannot truncate them.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                     dir=path.parent, suffix=".temporary", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_record(path):
    record = read_json(path)
    if not isinstance(record, dict) or record.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported experiment record: {path}")
    return record


def reserve_record(path, record):
    """Exclusive creation protects prior runs, including incomplete attempts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")


def portable_path(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def resolve_checkpoint(training):
    return ROOT / training["checkpoint"]["path"]


def code_hashes():
    return {"model/" + name: sha256(ROOT / "model" / name) for name in CORE_FILES}


def test_hashes(records):
    return {r["image_id"]: sha256(r["path"]) for r in records if r["split"] == "test"}


def write_predictions(path, rows):
    """JSON cells preserve full-precision class scores and source attribution."""
    if not rows:
        raise ValueError("No predictions to save")
    path = Path(path)
    temporary = path.with_suffix(".temporary")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, allow_nan=False) if k in LIST_FIELDS
                                 else str(v).lower() if k == "correct" else v
                                 for k, v in row.items()})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_predictions(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = LIST_FIELDS | INT_FIELDS | {"image_id", "true_dish", "predicted_dish", "correct"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Prediction CSV is missing required columns")
        rows = []
        for raw in reader:
            row = dict(raw)
            if row["correct"] not in ("true", "false"):
                raise ValueError("Prediction correctness must be true or false")
            row["correct"] = row["correct"] == "true"
            for name in INT_FIELDS:
                row[name] = int(row[name])
            for name in LIST_FIELDS:
                row[name] = json.loads(row[name])
            rows.append(row)
    return rows
