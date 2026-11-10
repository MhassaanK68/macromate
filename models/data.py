"""Dataset validation, image preprocessing, and train/validation loaders."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, DataLoader

from .config import ROOT, PROCESSED, SIZE, MEAN, STD, read_labels, sha256


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


def image_tensor(path, prepared=True):
    with Image.open(path) as image:
        if prepared:
            if image.format != "PNG" or image.mode != "RGB" or image.size != (SIZE, SIZE):
                raise ValueError("Prepared input must be an RGB 224x224 PNG")
        else:
            # Match data scripts/preprocess.py exactly, including rounding.
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            scale = SIZE / min(width, height)
            width, height = max(SIZE, round(width * scale)), max(SIZE, round(height * scale))
            image = image.resize((width, height), Image.Resampling.BICUBIC)
            left, top = (width - SIZE) // 2, (height - SIZE) // 2
            image = image.crop((left, top, left + SIZE, top + SIZE))
        pixels = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.array(MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(STD, dtype=np.float32).reshape(3, 1, 1)
    return torch.from_numpy((pixels - mean) / std)


class DishDataset(Dataset):
    def __init__(self, records, split):
        if split not in ("train", "val"):
            raise ValueError("Training datasets permit train or val only; test is reserved")
        self.records = [r for r in records if r["split"] == split]
        if not self.records:
            raise ValueError(f"No records for {split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return image_tensor(row["path"]), row["label_id"], row["image_id"]


def make_loaders(records, batch_size, seed):
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(DishDataset(records, "train"), batch_size=batch_size,
                       shuffle=True, generator=generator, num_workers=0)
    val = DataLoader(DishDataset(records, "val"), batch_size=batch_size, num_workers=0)
    return train, val
