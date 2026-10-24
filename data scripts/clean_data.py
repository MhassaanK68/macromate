"""Initial Phase 2 checks for Macromate image and nutrition metadata.

This is a starter pipeline. Review its flags and extend it as real sources arrive.
It does not download or redistribute image files.
"""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
LABELS = {
    "chicken biryani", "chicken pulao", "chicken karahi", "daal",
    "nihari", "haleem", "samosa", "chapati",
}
METADATA_FIELDS = ["image_id", "local_path", "dish_label", "source_id", "notes"]
NUTRITION_FIELDS = [
    "kcal_per_100g", "protein_g_per_100g", "carbs_g_per_100g",
    "fat_g_per_100g",
]


def read_csv(path: Path, required: list[str]) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_nutrition() -> dict[str, object]:
    rows = read_csv(DATA / "nutrition.csv", ["dish_label", *NUTRITION_FIELDS, "source"])
    problems = []
    seen_labels = set()
    for line, row in enumerate(rows, start=2):
        label = (row["dish_label"] or "").strip().lower()
        if label not in LABELS:
            problems.append(f"line {line}: unknown dish label")
        if label in seen_labels:
            problems.append(f"line {line}: duplicate dish label")
        seen_labels.add(label)
        if not (row["source"] or "").strip():
            problems.append(f"line {line}: missing nutrition source")
        for field in NUTRITION_FIELDS:
            try:
                value = float(row[field])
            except (TypeError, ValueError):
                problems.append(f"line {line}: invalid {field}")
                continue
            if not math.isfinite(value):
                problems.append(f"line {line}: non-finite {field}")
                continue
            if value < 0:
                problems.append(f"line {line}: negative {field}")
            if field != "kcal_per_100g" and value > 100:
                problems.append(f"line {line}: review {field} above 100 g per 100 g")
            if field == "kcal_per_100g" and value > 900:
                problems.append(f"line {line}: review energy above 900 kcal per 100 g")
    return {"rows": len(rows), "issues_to_review": problems}


def main() -> None:
    rows = read_csv(DATA / "metadata.csv", METADATA_FIELDS)
    sources = read_csv(DATA / "sources.csv", ["source_id", "url", "license_or_terms"])
    source_ids = {r["source_id"].strip() for r in sources if r["source_id"]}
    nutrition = check_nutrition()

    if not rows:
        print("No image records yet. Add licensed images and metadata before cleaning.")
        print(f"Nutrition rows: {nutrition['rows']}")
        return

    cleaned = []
    exclusions = []
    flags = []
    seen_ids = set()
    seen_hashes = set()

    for line, row in enumerate(rows, start=2):
        image_id = (row["image_id"] or "").strip()
        label = (row["dish_label"] or "").strip().lower()
        source_id = (row["source_id"] or "").strip()
        local_path = (row["local_path"] or "").strip()
        reason = None

        if not image_id or not local_path or not label or not source_id:
            reason = "missing required value"
        elif label not in LABELS:
            reason = "unsupported dish label"
        elif source_id not in source_ids:
            reason = "source_id absent from data/sources.csv"
        elif image_id in seen_ids:
            reason = "duplicate image_id"
        else:
            path = (ROOT / local_path).resolve()
            if not path.is_relative_to(RAW.resolve()):
                reason = "local_path must be inside data/raw"
            elif not path.is_file():
                reason = "missing image file"
            else:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        width, height = image.size
                    digest = hash_file(path)
                    if digest in seen_hashes:
                        reason = "duplicate image bytes"
                    else:
                        seen_hashes.add(digest)
                        if width < 224 or height < 224:
                            flags.append({"image_id": image_id, "flag": "small image", "size": [width, height]})
                        if max(width, height) / max(1, min(width, height)) > 3:
                            flags.append({"image_id": image_id, "flag": "unusual aspect ratio", "size": [width, height]})
                except (OSError, UnidentifiedImageError, ValueError):
                    reason = "unreadable image file"

        if reason:
            exclusions.append({"line": line, "image_id": image_id, "reason": reason})
            continue
        seen_ids.add(image_id)
        cleaned.append({**row, "image_id": image_id, "local_path": local_path,
                        "dish_label": label, "source_id": source_id})

    PROCESSED.mkdir(parents=True, exist_ok=True)
    with (PROCESSED / "metadata_clean.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(cleaned)

    summary = {
        "raw_images": len(rows),
        "usable_images": len(cleaned),
        "raw_class_counts": dict(Counter((r["dish_label"] or "").strip().lower() for r in rows)),
        "usable_class_counts": dict(Counter(r["dish_label"] for r in cleaned)),
        "exclusions": exclusions,
        "flags_for_manual_review": flags,
        "nutrition": nutrition,
    }
    (PROCESSED / "cleaning_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Raw images: {len(rows)} | usable: {len(cleaned)} | excluded: {len(exclusions)}")
    print(f"Review flags: {len(flags)} | nutrition issues: {len(nutrition['issues_to_review'])}")
    print("Wrote data/processed/metadata_clean.csv and cleaning_summary.json")


if __name__ == "__main__":
    main()
