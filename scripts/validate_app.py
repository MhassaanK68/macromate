"""Check the real model adapter against CLI behavior and record CPU latency.

Run from the root: python scripts/validate_app.py --output runs/app_validation.json
This uses ten non-held-out source photos for software checks, not accuracy claims.
"""

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "app_validation.json")
    args = parser.parse_args()
    from app_core.inference import CHECKPOINT, get_recognizer, predict_dishes
    from scripts.prepare_app_assets import digest, prepare_assets

    prepare_assets()
    begin = time.perf_counter()
    recognizer = get_recognizer()
    cold_seconds = time.perf_counter() - begin
    with (ROOT / "evidence" / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        held_out = {row["image_id"] for row in csv.DictReader(handle)}
    candidates = []
    folders = sorted((ROOT / "data" / "images").iterdir())
    for index in range(2):
        for folder in folders:
            choices = [path for path in sorted(folder.glob("*.jpg")) if path.stem not in held_out]
            candidates.append(choices[index])
    samples = candidates[:10]
    # One untimed inference initializes CPU kernels before warm measurements.
    recognizer.predict(str(samples[0]), prepared=False)
    measurements = []
    for path in samples:
        content = path.read_bytes()
        begin = time.perf_counter()
        adapted = predict_dishes(content)
        elapsed = time.perf_counter() - begin
        direct = recognizer.predict(str(path), prepared=False)["ranked_dishes"][:3]
        if [item["dish"] for item in adapted] != [item["dish"] for item in direct]:
            raise AssertionError("Adapter and direct Recognizer rankings disagree")
        measurements.append({"photo": path.relative_to(ROOT).as_posix(), "seconds": elapsed,
                             "ranked_dishes": [item["dish"] for item in adapted]})
    cli = json.loads(subprocess.check_output(
        [sys.executable, "-m", "models.infer", str(CHECKPOINT), str(samples[0]), "--device", "cpu"],
        cwd=ROOT, text=True))
    if [item["dish"] for item in cli["ranked_dishes"][:3]] != measurements[0]["ranked_dishes"]:
        raise AssertionError("CLI and application rankings disagree")
    record = {
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "working_tree": "See associated validation notes for uncommitted application changes",
        "application_sha256": {name: digest(ROOT / name) for name in (
            "app.py", "app_core/inference.py", "app_core/nutrition.py", "app_core/meal_log.py",
            "app_core/workflow.py", "scripts/prepare_app_assets.py", "requirements.txt", "requirements-model.txt")},
        "platform": platform.platform(), "python": platform.python_version(),
        "processor": platform.processor(), "device": "cpu", "checkpoint_sha256": digest(CHECKPOINT),
        "cold_model_load_seconds": cold_seconds,
        "sample_count": len(samples), "held_out_photos_excluded": True,
        "untimed_warmup_count": 1,
        "warm_median_seconds": statistics.median(item["seconds"] for item in measurements),
        "warm_slowest_seconds": max(item["seconds"] for item in measurements),
        "cli_adapter_ranking_agreement": True, "measurements": measurements,
        "scope": "Software integration and latency only; not new accuracy or hosted UI latency",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in record.items() if key != "measurements"}, indent=2))


if __name__ == "__main__":
    main()
