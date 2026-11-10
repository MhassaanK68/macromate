"""Verify regenerated data against the recorded experiment, not just another run."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.config import read_record, sha256  # noqa: E402
from models.data import load_records, data_lineage  # noqa: E402


def verify_dataset():
    training = read_record(ROOT / "evidence/training.json")
    evaluation = read_record(ROOT / "evidence/evaluation.json")
    records, labels = load_records()
    if labels != training["labels"] or data_lineage(records) != training["lineage"]:
        raise ValueError("Regenerated training/validation data differ from the recorded experiment. "
                         "Use requirements.txt, including Pillow 13.0.0, then rerun data preparation.")
    expected = evaluation["protocol"]["test_pixels_sha256"]
    actual = {r["image_id"]: sha256(r["path"]) for r in records if r["split"] == "test"}
    if actual != expected:
        raise ValueError("Prepared test files differ from the recorded evaluation.")
    return {"status": "passed", "prepared_images": len(records),
            "train_validation_images": len(training["lineage"]["train_val_pixels"]),
            "test_files_hashed_without_inference": len(actual)}


if __name__ == "__main__":
    print(json.dumps(verify_dataset(), indent=2))
