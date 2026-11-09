"""Verify the selected pilot checkpoint in a fresh process without training imports."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import ROOT, write_json
from .data import DishDataset, load_records, data_lineage
from .infer import Recognizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, default=ROOT / "evidence" / "phase3")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    selection = json.loads((args.evidence / "selection.json").read_text(encoding="utf-8"))
    run_id = selection["run_id"]
    records, labels = load_records()
    recognizer = Recognizer(ROOT / selection["pilot_checkpoint"], args.device)
    if recognizer.model.labels != labels:
        raise ValueError("Current data label order differs from saved model")
    if recognizer.checkpoint["metadata"]["lineage"] != data_lineage(records):
        raise ValueError("Current manifests or training/validation pixels differ from checkpoint")
    saved = json.loads((args.evidence / run_id / "best_validation_predictions.json").read_text(encoding="utf-8"))
    batch_size = recognizer.checkpoint["metadata"]["settings"]["batch_size"]
    loader = DataLoader(DishDataset(records, "val"), batch_size=batch_size, num_workers=0)
    logits, ids = [], []
    with torch.inference_mode():
        for pixels, _, image_ids in loader:
            logits.append(recognizer.model(pixels.to(args.device)).cpu())
            ids.extend(image_ids)
    logits = torch.cat(logits)
    expected = torch.tensor(saved["logits"])
    max_error = float((logits - expected).abs().max())
    # CPU and CUDA can differ slightly; identical devices use the strict tolerance.
    same_device = recognizer.checkpoint["metadata"]["environment"]["device"] == args.device
    tolerance = 1e-6 if same_device else 1e-4
    if ids != saved["image_ids"] or not torch.allclose(logits, expected, atol=tolerance, rtol=0):
        raise ValueError(f"Fresh-process reload mismatch: {max_error}")
    sample = next(r for r in records if r["image_id"] == ids[0])
    prediction = recognizer.predict(sample["path"], prepared=True)
    if prediction["ranked_dishes"][0]["label_id"] != int(logits[0].argmax()):
        raise ValueError("App-facing inference ranking disagrees with validation")
    result = {"status": "passed", "run_id": run_id, "device": args.device,
              "validation_images": len(ids), "reload_max_error": max_error,
              "absolute_tolerance": tolerance, "test_pixels_read": False,
              "sample_image_id": ids[0], "sample_prediction": prediction,
              "training_module_imported": "model.train" in __import__("sys").modules}
    write_json(args.evidence / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
