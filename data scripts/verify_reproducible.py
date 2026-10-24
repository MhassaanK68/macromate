"""Phase 2, step 4.5: prove the pipeline gives identical outputs on a second run.

Runs clean_data -> preprocess -> split_data twice and compares SHA-256 hashes of
every machine-readable output.  Writes data/processed/reproducibility_check.json.

    python "data scripts/verify_reproducible.py"
"""

import hashlib
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import PROCESSED, ROOT, write_json  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ["clean_data.py", "preprocess.py", "split_data.py"]
OUTPUTS = ["metadata_clean.csv", "image_features.csv", "exclusions.csv",
           "near_duplicate_pairs.csv", "outlier_flags.csv", "nutrition_clean.csv",
           "cleaning_summary.json", "label_map.json", "preprocess_summary.json",
           "split.csv", "split_summary.json"]


def run_all() -> dict[str, str]:
    for step in STEPS:
        subprocess.run([sys.executable, os.path.join(HERE, step)], cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL)
    return {name: hashlib.sha256((PROCESSED / name).read_bytes()).hexdigest()
            for name in OUTPUTS if (PROCESSED / name).exists()}


def main() -> None:
    first, second = run_all(), run_all()
    same = first == second and bool(first)
    write_json(PROCESSED / "reproducibility_check.json", {
        "files_compared": sorted(first), "identical_on_second_run": same,
        "sha256_first_run": first,
    })
    print(f"Compared {len(first)} output files across two full runs.")
    print("RESULT:", "IDENTICAL" if same else "DIFFERENT (see reproducibility_check.json)")
    if not first:
        print("(no outputs yet: add image records first)")
    sys.exit(0 if same or not first else 1)


if __name__ == "__main__":
    main()
