"""Shared model settings. Label order is read from Phase 2, never duplicated."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RUNS = ROOT / "runs"
CHECKPOINT = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
SIZE = 224
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
DEFAULTS = {
    "block_lr": 1e-5, "projection_lr": 5e-5, "batch_size": 8,
    "weight_decay": 0.01, "epochs": 20, "patience": 5, "seed": 42,
    "warmup_fraction": 0.1, "gradient_clip": 1.0,
    "betas": [0.9, 0.999], "epsilon": 1e-8, "dropout": 0.0,
}


def read_labels(path=PROCESSED / "label_map.json"):
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = mapping["id_to_label"]
    if set(ids) != {str(i) for i in range(8)}:
        raise ValueError("label_map must contain exactly class IDs 0 through 7")
    labels = [ids[str(i)] for i in range(8)]
    if len(set(labels)) != 8 or any(not isinstance(x, str) or not x for x in labels):
        raise ValueError("label_map needs eight unique nonempty dish names")
    if mapping["label_to_id"] != {label: i for i, label in enumerate(labels)}:
        raise ValueError("label_map directions disagree")
    return labels


def prompts_for(labels):
    return ["a photo of Pakistani " + label for label in labels]


def preprocessing():
    return {"size": SIZE, "mean": list(MEAN), "std": list(STD),
            "prepared": "RGB PNG; scale /255 and normalize once; no resize/crop",
            "upload": "EXIF transpose, RGB, bicubic shorter side, centre crop"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
