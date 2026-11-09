"""Run the declared six-candidate Phase 3 pilot and export measured evidence."""

import argparse
import csv
import gc
import json
import traceback
from pathlib import Path

import torch

from .config import DEFAULTS, ROOT, RUNS, write_json
from .data import load_records, data_lineage
from .train import train_run
from .metrics import selection_key

CANDIDATES = [
    ("baseline", {}),
    ("block_low", {"block_lr": 5e-6}),
    ("block_high", {"block_lr": 2e-5}),
    ("projection_low", {"projection_lr": 2e-5}),
    ("batch_16", {"batch_size": 16}),
    ("no_decay", {"weight_decay": 0.0}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=RUNS / "phase3")
    parser.add_argument("--evidence", type=Path, default=ROOT / "evidence" / "phase3")
    args = parser.parse_args()
    if not 1 <= args.epochs <= 5:
        parser.error("Pilot budget must be 1..5 equal epochs per candidate")
    records, labels = load_records()
    lineage = data_lineage(records)
    protocol = {"candidate_changes": dict(CANDIDATES), "epochs_per_candidate": args.epochs,
                "fixed_defaults": DEFAULTS, "labels": labels, "lineage": lineage,
                "selection": "validation correct count, then lower loss, then lower measured runtime",
                "test_pixels_read": False}
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding="utf-8"))
        if old != protocol:
            raise ValueError("Existing pilot has a different protocol or data; use a new output directory")
    write_json(protocol_path, protocol)
    write_json(args.evidence / "protocol.json", protocol)
    results = []
    for name, change in CANDIDATES:
        settings = dict(DEFAULTS, **change)
        settings["epochs"] = args.epochs
        output = args.output / name
        try:
            if (output / "result.json").exists():
                result = json.loads((output / "result.json").read_text(encoding="utf-8"))
                if result["settings"] != settings or result["environment"]["device"] != args.device:
                    raise ValueError("Existing run settings/device differ; use a new output directory")
            else:
                result = train_run(settings, output, args.device, records, labels, lineage)
            result["run_id"] = name
            for filename in ("history.json", "setup.json", "best_validation_predictions.json"):
                value = json.loads((output / filename).read_text(encoding="utf-8"))
                write_json(args.evidence / name / filename, value)
        except (RuntimeError, ValueError, OSError) as error:
            result = {"run_id": name, "settings": settings, "status": "failed",
                      "error": str(error), "traceback": traceback.format_exc()}
            write_json(output / "failure.json", result)
            print(f"{name}: FAILED: {error}", flush=True)
        results.append(result)
        write_json(args.evidence / "results.json", results)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    completed = [r for r in results if r["status"] == "completed"]
    if not completed:
        raise RuntimeError("No measured pilot completed; inspect failure records")
    selected = max(completed, key=lambda r: (*selection_key(r["validation"]), -r["seconds"]))
    selected_settings = dict(selected["settings"], epochs=20)
    write_json(args.evidence / "selected_config.json", selected_settings)
    checkpoint_path = (args.output / selected["run_id"] / "best.pt").resolve()
    if checkpoint_path.is_relative_to(ROOT):
        checkpoint_path = checkpoint_path.relative_to(ROOT)
    write_json(args.evidence / "selection.json", {
        "run_id": selected["run_id"], "validation": selected["validation"],
        "pilot_checkpoint": checkpoint_path.as_posix(),
        "reason": "Highest validation correct count; lower validation loss breaks ties; runtime breaks exact ties.",
        "limits": f"{args.epochs}-epoch pilot is exploratory; 64 validation images; one image is 1.5625 percentage points. "
                  "Epoch cap 20, patience five, prompts and zero dropout are fixed design choices, not tuned findings.",
        "all_candidates_completed": len(completed) == len(CANDIDATES),
    })
    fields = ["run_id", "status", "block_lr", "projection_lr", "batch_size", "weight_decay",
              "dropout", "epochs_completed", "best_epoch", "val_correct", "val_total", "val_loss", "seconds"]
    with (args.evidence / "tuning.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            val = result.get("validation", {})
            writer.writerow({"run_id": result["run_id"], "status": result["status"],
                             **{k: result["settings"][k] for k in fields[2:7]},
                             **{k: result.get(k, "") for k in ("epochs_completed", "best_epoch", "seconds")},
                             "val_correct": val.get("top1_correct", ""), "val_total": val.get("total", ""),
                             "val_loss": val.get("loss", "")})
    print(f"Selected {selected['run_id']}: {selected['validation']['top1_correct']}/64", flush=True)
    if len(completed) != len(CANDIDATES):
        raise RuntimeError("Some pilot candidates failed; inspect evidence before claiming Phase 3 complete")


if __name__ == "__main__":
    main()
