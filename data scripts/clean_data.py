"""Phase 2, steps 4.1 and 4.2: validate raw inputs, find duplicates, flag outliers.

Reads (never modifies):
    data/metadata.csv, data/sources.csv, data/nutrition.csv
    data/review_decisions.csv  (optional: image_id,decision,reason; decision=drop)
    optional metadata column `group_id`: photos of the same physical plate share
    one value so the split keeps them together

Writes to data/processed/:
    metadata_clean.csv        records that passed every hard check
    image_features.csv        per-image size, hash and colour features (used by EDA/split)
    exclusions.csv            every removed record with the reason
    near_duplicate_pairs.csv  likely near-duplicates that need human review
    outlier_flags.csv         unusual images (flagged, NOT removed)
    nutrition_clean.csv       validated nutrition table with label IDs
    cleaning_summary.json     machine-readable counts (identical on every run)

Run from the repository root:  python "data scripts/clean_data.py"
"""

import math
import os
import sys
from collections import Counter, defaultdict

from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    DATA, LABEL_TO_ID, LABELS, METADATA_FIELDS, NEAR_DUP_MAX_DISTANCE,
    NUTRITION_FIELDS, PROCESSED, RAW_DIRS, ROOT, dhash, hamming, norm_label,
    read_csv, sha256_file, sources_path, write_csv, write_json,
)

RIGHTS_OK = "per_file_license_recorded"
CLEAN_FIELDS = METADATA_FIELDS + ["group_id", "rights_status", "rights_cleared"]
FEATURE_FIELDS = [
    "image_id", "dish_label", "source_id", "group_id", "rights_cleared", "width", "height", "aspect_ratio",
    "file_bytes", "mean_r", "mean_g", "mean_b", "brightness", "sha256", "dhash",
]


# --------------------------------------------------------------------------
# Nutrition checks (section 4.2: missing, negative, impossible, implausible)
# --------------------------------------------------------------------------
def check_nutrition() -> tuple[list[dict], dict]:
    path = DATA / "nutrition.csv"
    rows = read_csv(path, ["dish_label", *NUTRITION_FIELDS, "source"])
    issues, clean, seen = [], [], set()
    for line, row in enumerate(rows, start=2):
        label = norm_label(row["dish_label"])
        ok = True
        if label not in LABEL_TO_ID:
            issues.append(f"line {line}: unknown dish label '{label}'")
            continue
        if label in seen:
            issues.append(f"line {line}: duplicate dish label '{label}'")
            continue
        seen.add(label)
        if not (row["source"] or "").strip():
            issues.append(f"line {line} ({label}): missing nutrition source")
            ok = False
        recipe = (row.get("recipe_assumptions") or row.get("recipe_assumption") or "").strip()
        if not recipe:
            issues.append(f"line {line} ({label}): no recipe assumption recorded")
        values = {}
        for field in NUTRITION_FIELDS:
            try:
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError
            except (TypeError, ValueError):
                issues.append(f"line {line} ({label}): missing/invalid {field}")
                ok = False
                continue
            if value < 0:
                issues.append(f"line {line} ({label}): negative {field}")
                ok = False
            values[field] = value
        if ok and len(values) == 4:
            p, c, f = (values["protein_g_per_100g"], values["carbs_g_per_100g"],
                       values["fat_g_per_100g"])
            kcal = values["kcal_per_100g"]
            if p + c + f > 100:
                issues.append(f"line {line} ({label}): macros sum to {p + c + f:.1f} g "
                              "per 100 g (impossible, >100)")
                ok = False
            if kcal > 900:
                issues.append(f"line {line} ({label}): {kcal:.0f} kcal/100 g exceeds "
                              "pure fat (900); impossible")
                ok = False
            atwater = 4 * p + 4 * c + 9 * f
            if kcal > 0 and abs(atwater - kcal) / kcal > 0.25:
                issues.append(f"line {line} ({label}): reported {kcal:.0f} kcal differs "
                              f"from macro-based estimate {atwater:.0f} kcal by >25% "
                              "(review units / serving conversion)")
            if ok:
                clean.append({
                    "dish_label": label, "label_id": LABEL_TO_ID[label],
                    **{k: round(v, 2) for k, v in values.items()},
                    "source": row["source"].strip(),
                    "recipe_assumptions": recipe,
                })
    missing_dishes = [l for l in LABELS if l not in seen]
    clean.sort(key=lambda r: r["label_id"])
    return clean, {
        "rows": len(rows), "valid_rows": len(clean),
        "dishes_without_nutrition_row": missing_dishes,
        "issues_to_review": issues,
    }


# --------------------------------------------------------------------------
# Image reading
# --------------------------------------------------------------------------
def read_image_info(path):
    """Return size and colour features; raise on unreadable files."""
    with Image.open(path) as probe:
        probe.verify()
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        thumb = image.resize((64, 64), Image.LANCZOS)
        r, g, b = ImageStat.Stat(thumb).mean
        return {
            "width": width, "height": height,
            "aspect_ratio": round(max(width, height) / max(1, min(width, height)), 3),
            "file_bytes": path.stat().st_size,
            "mean_r": round(r, 2), "mean_g": round(g, 2), "mean_b": round(b, 2),
            "brightness": round((r + g + b) / 3, 2),
            "dhash": f"{dhash(image):016x}",
            "sha256": sha256_file(path),
        }


def quartile_fences(values: list[float]) -> tuple[float, float] | None:
    """Tukey fences at 3*IQR (extreme outliers only); None when too few images."""
    if len(values) < 8:
        return None
    ordered = sorted(values)

    def pct(p):
        k = (len(ordered) - 1) * p
        lo, hi = math.floor(k), math.ceil(k)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)

    q1, q3 = pct(0.25), pct(0.75)
    iqr = q3 - q1
    return q1 - 3 * iqr, q3 + 3 * iqr


def main() -> None:
    rows = read_csv(DATA / "metadata.csv", METADATA_FIELDS)
    sources = read_csv(sources_path(), ["source_id", "url"])
    source_ids = {r["source_id"].strip() for r in sources if r["source_id"]}
    nutrition_clean, nutrition_report = check_nutrition()
    if nutrition_clean:
        write_csv(PROCESSED / "nutrition_clean.csv",
                  list(nutrition_clean[0].keys()), nutrition_clean)

    decisions_path = DATA / "review_decisions.csv"
    decisions = {}
    if decisions_path.exists():
        for r in read_csv(decisions_path, ["image_id", "decision", "reason"]):
            if r["decision"].strip().lower() == "drop":
                decisions[r["image_id"].strip()] = (r["reason"] or "").strip()

    if not rows:
        print("No image records yet. Add licensed images and metadata before cleaning.")
        print(f"Nutrition rows: {nutrition_report['rows']} "
              f"(valid: {nutrition_report['valid_rows']})")
        return

    # ---- 4.1 validate raw inputs ------------------------------------------
    missing_values = {f: sum(1 for r in rows if not (r.get(f) or "").strip())
                      for f in ["image_id", "local_path", "dish_label", "source_id"]}
    exclusions, kept, seen_ids = [], [], set()
    for line, row in enumerate(rows, start=2):
        image_id = (row["image_id"] or "").strip()
        label = norm_label(row["dish_label"])
        source_id = (row["source_id"] or "").strip()
        local_path = (row["local_path"] or "").strip().replace("\\", "/")
        reason, info = None, None
        if image_id and local_path and source_id and not label:
            reason = ("no canonical label (ambiguous/excluded during source screening)"
                      if (row.get("proposed_dish_label") or "").strip() or row.get("quality_flag")
                      else "missing required value")
        elif not (image_id and local_path and label and source_id):
            reason = "missing required value"
        elif label not in LABEL_TO_ID:
            reason = "unknown dish name"
        elif source_id not in source_ids:
            reason = "source_id absent from data/sources.csv"
        elif image_id in seen_ids:
            reason = "duplicate image_id"
        else:
            path = (ROOT / local_path).resolve()
            if not any(path.is_relative_to(d.resolve()) for d in RAW_DIRS):
                reason = "local_path must be inside data/raw or data/images"
            elif not path.is_file():
                reason = "missing image file"
            else:
                try:
                    info = read_image_info(path)
                except (OSError, UnidentifiedImageError, ValueError, SyntaxError):
                    reason = "unreadable image file"
        if image_id:
            seen_ids.add(image_id)
        if reason:
            exclusions.append({"line": line, "image_id": image_id,
                               "dish_label": label, "reason": reason})
            continue
        kept.append({"image_id": image_id, "local_path": local_path, "dish_label": label,
                     "source_id": source_id, "notes": (row.get("notes") or "").strip(),
                     "group_id": (row.get("group_id") or "").strip(),
                     "rights_status": (row.get("rights_status") or "").strip(),
                     "rights_cleared": (row.get("rights_status") or "").strip() == RIGHTS_OK,
                     **info})

    # ---- 4.2a exact duplicates (identical bytes) ---------------------------
    by_hash = defaultdict(list)
    for rec in kept:
        by_hash[rec["sha256"]].append(rec)
    drop = {}
    for group in by_hash.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda r: r["image_id"])
        if len({r["dish_label"] for r in group}) > 1:
            for rec in group:  # same picture, different labels -> ambiguous
                drop[rec["image_id"]] = "identical image under conflicting labels"
        else:
            for rec in group[1:]:
                drop[rec["image_id"]] = f"exact duplicate of {group[0]['image_id']}"
    exact_dups = sum(1 for v in drop.values() if v.startswith("exact"))
    label_conflicts = sum(1 for v in drop.values() if "conflicting" in v)

    # ---- 4.2b manual decisions recorded in data/review_decisions.csv -------
    manual_drops = 0
    for rec in kept:
        if rec["image_id"] in decisions and rec["image_id"] not in drop:
            drop[rec["image_id"]] = "manual review: " + (decisions[rec["image_id"]] or "no reason given")
            manual_drops += 1
    for rec in kept:
        if rec["image_id"] in drop:
            exclusions.append({"line": "", "image_id": rec["image_id"],
                               "dish_label": rec["dish_label"], "reason": drop[rec["image_id"]]})
    survivors = [r for r in kept if r["image_id"] not in drop]
    survivors.sort(key=lambda r: r["image_id"])

    # ---- 4.2c near duplicates (perceptual hash), flagged for review --------
    pairs = []
    hashes = [(r, int(r["dhash"], 16)) for r in survivors]
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            dist = hamming(hashes[i][1], hashes[j][1])
            if dist <= NEAR_DUP_MAX_DISTANCE:
                a, b = hashes[i][0], hashes[j][0]
                pairs.append({
                    "image_a": a["image_id"], "image_b": b["image_id"],
                    "label_a": a["dish_label"], "label_b": b["dish_label"],
                    "hamming_distance": dist,
                    "same_label": a["dish_label"] == b["dish_label"],
                    "action": "review: keep one, add the other to data/review_decisions.csv",
                })
    cross_label_pairs = sum(1 for p in pairs if not p["same_label"])

    # ---- 4.2d image outliers (flagged, never auto-removed) -----------------
    flags = []
    fences = {f: quartile_fences([r[f] for r in survivors])
              for f in ("width", "height", "file_bytes")}   # aspect ratio uses the absolute 3:1 rule
    for rec in survivors:
        found = []
        if min(rec["width"], rec["height"]) < 224:
            found.append("shorter side below 224 px (CLIP input size)")
        if rec["aspect_ratio"] > 3:
            found.append("extreme aspect ratio (>3:1)")
        if rec["file_bytes"] < 5 * 1024:
            found.append("very small file (<5 KB)")
        for field, fence in fences.items():
            if fence and not (fence[0] <= rec[field] <= fence[1]):
                found.append(f"{field} outside IQR fences [{fence[0]:.0f}, {fence[1]:.0f}]")
        for text in found:
            flags.append({"image_id": rec["image_id"], "dish_label": rec["dish_label"],
                          "flag": text})

    # ---- write outputs ------------------------------------------------------
    write_csv(PROCESSED / "metadata_clean.csv", CLEAN_FIELDS, [
        {k: r[k] for k in CLEAN_FIELDS} for r in survivors])
    write_csv(PROCESSED / "image_features.csv", FEATURE_FIELDS, [
        {k: r[k] for k in FEATURE_FIELDS} for r in survivors])
    exclusions.sort(key=lambda e: (e["image_id"], e["reason"]))
    write_csv(PROCESSED / "exclusions.csv", ["line", "image_id", "dish_label", "reason"],
              exclusions)
    pairs.sort(key=lambda p: (p["image_a"], p["image_b"]))
    write_csv(PROCESSED / "near_duplicate_pairs.csv",
              ["image_a", "image_b", "label_a", "label_b", "hamming_distance",
               "same_label", "action"], pairs)
    write_csv(PROCESSED / "outlier_flags.csv", ["image_id", "dish_label", "flag"], flags)

    flagged_images = {f["image_id"] for f in flags}
    summary = {
        "input_records": len(rows),
        "missing_values_by_column": missing_values,
        "invalid_records": len(exclusions) - exact_dups - label_conflicts - manual_drops,
        "exact_duplicates_removed": exact_dups,
        "conflicting_label_copies_removed": label_conflicts,
        "manual_review_drops": manual_drops,
        "final_clean_records": len(survivors),
        "exclusion_reasons": dict(Counter(
            e["reason"].split(" of ")[0] if e["reason"].startswith("exact") else
            ("manual review" if e["reason"].startswith("manual") else e["reason"])
            for e in exclusions)),
        "near_duplicate_pairs_to_review": len(pairs),
        "near_duplicate_pairs_with_different_labels": cross_label_pairs,
        "near_duplicate_hash_threshold_bits": NEAR_DUP_MAX_DISTANCE,
        "outlier_flags": len(flags),
        "images_with_outlier_flags": len(flagged_images),
        "raw_class_counts": {l: sum(1 for r in rows if norm_label(r["dish_label"]) == l)
                             for l in LABELS},
        "raw_records_with_blank_label": sum(1 for r in rows if not norm_label(r["dish_label"])),
        "clean_class_counts": {l: sum(1 for r in survivors if r["dish_label"] == l)
                               for l in LABELS},
        "unknown_label_records": sum(1 for r in rows if norm_label(r["dish_label"])
                                     and norm_label(r["dish_label"]) not in LABEL_TO_ID),
        "clean_records_with_documented_rights": sum(1 for r in survivors if r["rights_cleared"]),
        "clean_records_with_rights_held_or_unknown": sum(1 for r in survivors if not r["rights_cleared"]),
        "clean_rights_cleared_class_counts": {l: sum(1 for r in survivors if r["dish_label"] == l and r["rights_cleared"])
                                              for l in LABELS},
        "nutrition": nutrition_report,
    }
    write_json(PROCESSED / "cleaning_summary.json", summary)

    print(f"Input records: {len(rows)} | final clean: {len(survivors)}")
    print(f"  invalid: {summary['invalid_records']} | exact duplicates: {exact_dups} | "
          f"label conflicts: {label_conflicts} | manual drops: {manual_drops}")
    print(f"  near-duplicate pairs to review: {len(pairs)} "
          f"({cross_label_pairs} with different labels)")
    print(f"  images with outlier flags: {len(flagged_images)}")
    print(f"  clean images with documented per-file rights: {summary['clean_records_with_documented_rights']} "
          f"| rights held/unknown: {summary['clean_records_with_rights_held_or_unknown']}")
    print(f"  nutrition: {nutrition_report['valid_rows']}/{nutrition_report['rows']} valid rows, "
          f"{len(nutrition_report['issues_to_review'])} issue(s)")
    print("Wrote data/processed/ (metadata_clean, image_features, exclusions, "
          "near_duplicate_pairs, outlier_flags, nutrition_clean, cleaning_summary.json)")


if __name__ == "__main__":
    main()
