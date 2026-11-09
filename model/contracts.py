"""Manifest checks that run with the Python standard library alone."""

import csv
from collections import Counter
from pathlib import Path

from .config import ROOT, PROCESSED, read_labels, sha256


def read_unique(path, fields):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not set(fields).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns in {path}: {fields}")
        rows = list(reader)
    result = {}
    for row in rows:
        image_id = row["image_id"]
        if not image_id or image_id in result:
            raise ValueError(f"Empty or duplicate image_id in {path}: {image_id}")
        result[image_id] = row
    return result


def load_records(processed=PROCESSED, root=ROOT, expected_counts=True):
    processed, root = Path(processed), Path(root).resolve()
    labels = read_labels(processed / "label_map.json")
    manifest = read_unique(processed / "clip_manifest.csv",
                           ["image_id", "dish_label", "label_id", "clip_path"])
    splits = read_unique(processed / "split.csv",
                         ["image_id", "dish_label", "label_id", "split", "cluster_id"])
    if set(manifest) != set(splits):
        raise ValueError("Manifest and split image IDs do not match")
    records, groups, paths = [], {}, set()
    for image_id in sorted(manifest):
        row, split = manifest[image_id], splits[image_id]
        if row["dish_label"] != split["dish_label"] or row["label_id"] != split["label_id"]:
            raise ValueError(f"Label mismatch for {image_id}")
        label_id = int(row["label_id"])
        if not 0 <= label_id < 8 or labels[label_id] != row["dish_label"]:
            raise ValueError(f"Label ID disagrees with label_map for {image_id}")
        subset = split["split"]
        if subset not in ("train", "val", "test"):
            raise ValueError(f"Unknown split for {image_id}: {subset}")
        group = split["cluster_id"]
        if not group or groups.get(group, subset) != subset:
            raise ValueError(f"Empty group or group crosses splits: {group}")
        groups[group] = subset
        path = (root / row["clip_path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path in paths:
            raise ValueError(f"Invalid, missing or reused prepared path: {path}")
        paths.add(path)
        records.append(dict(row, split=subset, path=path, label_id=label_id))
    counts = Counter((r["split"], r["label_id"]) for r in records)
    if expected_counts:
        expected = {(s, i): count for s, count in [("train", 35), ("val", 8), ("test", 7)]
                    for i in range(8)}
        if dict(counts) != expected:
            raise ValueError(f"Expected 35/8/7 per dish; found {dict(counts)}")
    return records, labels


def data_lineage(records, processed=PROCESSED):
    files = {name: sha256(Path(processed) / name) for name in
             ("clip_manifest.csv", "split.csv", "label_map.json")}
    # Hash only training/validation pixels. Test paths are checked, never decoded.
    files["train_val_pixels"] = {r["image_id"]: sha256(r["path"])
                                  for r in records if r["split"] != "test"}
    return files
