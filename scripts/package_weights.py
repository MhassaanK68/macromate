"""Prepare a verified external-hosting asset; no weights enter the Git repository."""

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.config import read_record, sha256  # noqa: E402


def main():
    training = read_record(ROOT / "evidence/training.json")
    checkpoint = ROOT / training["checkpoint"]["path"]
    expected = training["checkpoint"]["sha256"]
    if sha256(checkpoint) != expected:
        raise ValueError("Checkpoint identity does not match the evaluated model.")
    destination = ROOT / "output/weights"
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "macromate-deliverable2-weights.zip"
    manifest = {
        "checkpoint_path": training["checkpoint"]["path"],
        "checkpoint_sha256": expected, "checkpoint_bytes": checkpoint.stat().st_size,
        "base_model": training["base_model"], "labels": training["labels"],
        "selected_epoch": training["result"]["best_epoch"],
        "base_cache_included": False,
    }
    instructions = (
        "Macromate Deliverable 2 - evaluated adapted weights\n\n"
        "Extract this ZIP at the repository root, retaining runs/phase4/best.pt.\n"
        "Install the torch wheel and requirements.txt as explained in README.md.\n"
        "Download the pinned base model once with:\n\n"
        "python -c \"from huggingface_hub import snapshot_download; "
        "snapshot_download('openai/clip-vit-base-patch32', "
        f"revision='{training['base_model']['revision']}', cache_dir='runs/hub', "
        "allow_patterns=['*.json', '*.txt', 'pytorch_model.bin'])\"\n\n"
        "The ~600 MB base cache is separate; it is not included in this ZIP.\n"
        "Verify best.pt SHA-256: " + expected + "\n"
        "Run: python -m models.infer runs/phase4/best.pt path/to/meal.jpg --device cpu\n"
        "Returns rankings for eight supported dishes; scores are uncalibrated.\n"
    )
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        handle.write(checkpoint, manifest["checkpoint_path"])
        handle.writestr("WEIGHTS_README.txt", instructions)
        handle.writestr("weights_manifest.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(archive) as handle:
        if handle.testzip():
            raise ValueError("Weights ZIP failed its CRC check.")
        import hashlib
        if hashlib.sha256(handle.read(manifest["checkpoint_path"])).hexdigest() != expected:
            raise ValueError("Archived weights differ from the evaluated checkpoint.")
    (destination / "SHA256SUMS.txt").write_text(
        f"{sha256(archive)}  {archive.name}\n{expected}  runs/phase4/best.pt\n",
        encoding="utf-8", newline="\n")
    print(f"Verified {archive.relative_to(ROOT)} ({archive.stat().st_size:,} bytes).")


if __name__ == "__main__":
    main()
