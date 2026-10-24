"""Shared constants and helpers for the Macromate Phase 2 data scripts.

Every script imports the dish list, paths and CLIP constants from here so that
they cannot drift apart.  Change LABELS here if the final dish list changes
after the data audit.
"""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
IMAGES = DATA / "images"          # Section 3 package stores images here
RAW_DIRS = [RAW, IMAGES]          # images may live in either folder (both git-ignored)
PROCESSED = DATA / "processed"
FIGURES = ROOT / "figures"

# Canonical dish names.  The ORDER defines the integer label IDs (0..7), so it
# stays fixed even if a class temporarily has no images.
LABELS = [
    "chicken biryani", "chicken pulao", "chicken karahi", "daal",
    "nihari", "haleem", "samosa", "chapati",
]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

METADATA_FIELDS = ["image_id", "local_path", "dish_label", "source_id", "notes"]
NUTRITION_FIELDS = [
    "kcal_per_100g", "protein_g_per_100g", "carbs_g_per_100g", "fat_g_per_100g",
]

# Official preprocessing of the OpenAI CLIP ViT-B/32 (and ViT-B/16) checkpoints:
# RGB -> resize shorter side to 224 (bicubic) -> centre crop 224 -> scale to
# [0, 1] -> normalise per channel with the mean/std below.
CLIP_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

SPLIT_SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# A perceptual-hash distance at or below this (out of 64 bits) is treated as a
# "likely near-duplicate" that a human must review.
NEAR_DUP_MAX_DISTANCE = 6


def norm_label(value: str | None) -> str:
    """'Chicken_Biryani ' -> 'chicken biryani' (accepts underscores or spaces)."""
    return (value or "").strip().lower().replace("_", " ")


def sources_path() -> Path:
    """data/sources.csv if it has rows, otherwise the Section 3 source_log.csv."""
    main = DATA / "sources.csv"
    log = DATA / "source_log.csv"
    if main.exists() and len(read_csv(main, ["source_id"])) > 0:
        return main
    return log if log.exists() else main


def read_csv(path: Path, required: list[str]) -> list[dict[str, str]]:
    """Read a CSV and fail loudly if required columns are missing."""
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, obj: object) -> None:
    """Deterministic JSON: sorted keys, no timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image, hash_size: int = 8) -> int:
    """64-bit difference hash (robust to resize, re-compression, small crops)."""
    from PIL import Image

    small = image.convert("L").resize((hash_size + 1, hash_size), Image.LANCZOS)
    pixels = list(small.getdata())
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")
