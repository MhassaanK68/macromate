"""Phase 2, step 4.3: normalise and encode the variables.

* Encodes dish names as stable integer IDs (order fixed in common.LABELS).
* Applies the official CLIP image preprocessing to every clean image:
  RGB -> resize shorter side to 224 (bicubic) -> centre crop -> scale to [0,1]
  -> normalise with CLIP's per-channel mean/std.
  The 224x224 RGB result is saved as PNG for inspection; the pixel statistics
  before/after normalisation are recorded as evidence that it worked.
* Nutrition stays in consistent per-100-g units.  It is deliberately NOT scaled:
  the calculation is value_per_100g * grams / 100, so scaling would add nothing.

Run after clean_data.py:  python "data scripts/preprocess.py"
"""

import os
import re
import sys

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    CLIP_MEAN, CLIP_SIZE, CLIP_STD, LABEL_TO_ID, LABELS, PROCESSED, ROOT,
    read_csv, write_json,
)

OUT_DIR = PROCESSED / "clip_224"


def clip_preprocess(path):
    """Return the 224x224 RGB PIL image CLIP's transform produces before normalisation."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        scale = CLIP_SIZE / min(width, height)
        new_w, new_h = max(CLIP_SIZE, round(width * scale)), max(CLIP_SIZE, round(height * scale))
        image = image.resize((new_w, new_h), Image.BICUBIC)
        left, top = (new_w - CLIP_SIZE) // 2, (new_h - CLIP_SIZE) // 2
        image = image.crop((left, top, left + CLIP_SIZE, top + CLIP_SIZE))
    return image


def main() -> None:
    clean_path = PROCESSED / "metadata_clean.csv"
    if not clean_path.exists():
        print("data/processed/metadata_clean.csv not found. Run clean_data.py first "
              "(it needs image records in data/metadata.csv).")
        return
    rows = read_csv(clean_path, ["image_id", "local_path", "dish_label"])
    if not rows:
        print("No clean images to preprocess.")
        return
    rows.sort(key=lambda r: r["image_id"])

    write_json(PROCESSED / "label_map.json", {
        "label_to_id": LABEL_TO_ID, "id_to_label": {str(i): l for i, l in enumerate(LABELS)}})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mean = np.array(CLIP_MEAN, dtype=np.float64).reshape(3, 1, 1)
    std = np.array(CLIP_STD, dtype=np.float64).reshape(3, 1, 1)
    n_pix = 0
    s_raw = np.zeros(3); ss_raw = np.zeros(3)
    s_norm = np.zeros(3); ss_norm = np.zeros(3)
    lo, hi = np.inf, -np.inf
    for row in rows:
        image = clip_preprocess(ROOT / row["local_path"])
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", row["image_id"])
        image.save(OUT_DIR / f"{safe}.png")
        x = np.asarray(image, dtype=np.float64).transpose(2, 0, 1) / 255.0   # (3,H,W) in [0,1]
        z = (x - mean) / std
        n_pix += x.shape[1] * x.shape[2]
        s_raw += x.sum(axis=(1, 2)); ss_raw += (x ** 2).sum(axis=(1, 2))
        s_norm += z.sum(axis=(1, 2)); ss_norm += (z ** 2).sum(axis=(1, 2))
        lo, hi = min(lo, z.min()), max(hi, z.max())

    def stats(s, ss):
        m = s / n_pix
        return m, np.sqrt(np.maximum(ss / n_pix - m ** 2, 0))

    m_raw, sd_raw = stats(s_raw, ss_raw)
    m_norm, sd_norm = stats(s_norm, ss_norm)
    r4 = lambda a: [round(float(v), 4) for v in a]  # noqa: E731
    write_json(PROCESSED / "preprocess_summary.json", {
        "images_processed": len(rows),
        "clip_input_size": CLIP_SIZE,
        "steps": ["convert to RGB (after EXIF rotation)",
                  "resize shorter side to 224, bicubic",
                  "centre crop 224x224",
                  "scale pixels to [0,1]",
                  "normalise per channel with CLIP mean/std"],
        "clip_mean_rgb": list(CLIP_MEAN), "clip_std_rgb": list(CLIP_STD),
        "dataset_pixel_mean_before_normalisation_rgb": r4(m_raw),
        "dataset_pixel_std_before_normalisation_rgb": r4(sd_raw),
        "dataset_pixel_mean_after_normalisation_rgb": r4(m_norm),
        "dataset_pixel_std_after_normalisation_rgb": r4(sd_norm),
        "normalised_value_range": [round(float(lo), 3), round(float(hi), 3)],
        "label_encoding": LABEL_TO_ID,
        "nutrition_units": "per 100 g throughout; intentionally not scaled",
    })
    print(f"Preprocessed {len(rows)} images to {OUT_DIR.relative_to(ROOT)}/")
    print("  mean before:", r4(m_raw), "-> after:", r4(m_norm))
    print("  std  before:", r4(sd_raw), "-> after:", r4(sd_norm))
    print("Wrote label_map.json and preprocess_summary.json")


if __name__ == "__main__":
    main()
