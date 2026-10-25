"""Phase 2, step 4.5: prove the pipeline gives identical outputs on a second run.

Runs source audit -> cleaning -> preprocessing -> split -> EDA twice and compares
SHA-256 hashes of all expected CSV, JSON, preprocessed PNG and chart outputs.
Writes data/processed/reproducibility_check.json.

    python "data scripts/verify_reproducible.py"
"""

import hashlib
import csv
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, FIGURES, GENERATED_FIGURES, PROCESSED, ROOT, write_json  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["audit_sources.py", "clean_data.py", "preprocess.py", "split_data.py", "eda.py"]
OUTPUTS = ["metadata_clean.csv", "image_features.csv", "exclusions.csv",
           "near_duplicate_pairs.csv", "outlier_flags.csv", "nutrition_clean.csv",
           "cleaning_summary.json", "label_map.json", "clip_manifest.csv",
           "preprocess_summary.json", "split.csv", "split_summary.json"]


def run_all() -> dict[str, str]:
    for step in STEPS:
        subprocess.run([sys.executable, os.path.join(HERE, step)], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
    paths = {f"data/processed/{name}": PROCESSED / name for name in OUTPUTS}
    paths.update({f"figures/{name}": FIGURES / name for name in GENERATED_FIGURES})
    with (DATA / "sources.csv").open(newline="", encoding="utf-8-sig") as handle:
        source_ids = {row["source_id"] for row in csv.DictReader(handle)}
    evidence_paths = {ROOT / "evidence" / f"{sid}_summary.json" for sid in source_ids}
    if set((ROOT / "evidence").glob("*_summary.json")) != evidence_paths:
        raise RuntimeError("Evidence summaries do not match the source manifest")
    paths.update({path.relative_to(ROOT).as_posix(): path for path in evidence_paths})
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Pipeline did not create required outputs: {', '.join(missing)}")

    with (PROCESSED / "clip_manifest.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    if not manifest:
        raise RuntimeError("No preprocessed images were produced")
    with (PROCESSED / "metadata_clean.csv").open(newline="", encoding="utf-8") as handle:
        clean_ids = {row["image_id"] for row in csv.DictReader(handle)}
    if len(manifest) != len(clean_ids) or {row["image_id"] for row in manifest} != clean_ids:
        raise RuntimeError("clip_manifest.csv does not cover every clean image")
    expected_pngs = {ROOT / row["clip_path"] for row in manifest}
    actual_pngs = set((PROCESSED / "clip_224").glob("*.png"))
    if len(expected_pngs) != len(manifest) or actual_pngs != expected_pngs:
        raise RuntimeError("Preprocessed PNGs do not match clip_manifest.csv")
    paths.update({path.relative_to(ROOT).as_posix(): path for path in expected_pngs})
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in sorted(paths.items())}


def main() -> None:
    first, second = run_all(), run_all()
    same = first == second
    write_json(PROCESSED / "reproducibility_check.json", {
        "files_compared": sorted(first), "identical_on_second_run": same,
        "sha256_first_run": first,
    })
    print(f"Compared {len(first)} output files across two full runs.")
    print("RESULT:", "IDENTICAL" if same else "DIFFERENT (see reproducibility_check.json)")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
