"""Exploratory data analysis charts for the Deliverable 1 report.

Uses only real data.  Each chart is skipped (with a message) if its inputs do
not exist yet, so nothing is ever fabricated.  Output goes to figures/.

    figures/raw_class_counts.png      raw vs clean images per dish
    figures/missing_data.png          missing values per metadata column
    figures/sample_images.png         example photos per dish
    figures/image_size_distribution.png  width/height, aspect ratio, file size
    figures/class_by_source.png       which source supplied each dish
    figures/feature_correlation.png   correlations of image-level features
    figures/split_distribution.png    train/val/test counts per dish
    figures/nutrition_macros.png      reference nutrition per 100 g

Run after clean_data.py and split_data.py:  python "data scripts/eda.py"
"""

import os
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    DATA, FIGURES, LABELS, METADATA_FIELDS, PROCESSED, ROOT, norm_label, read_csv,
)

plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})
SHORT = {l: l.replace("chicken ", "ch. ") for l in LABELS}


def save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name)
    plt.close(fig)
    print(f"Wrote figures/{name}")


def maybe(path, required):
    return read_csv(path, required) if path.exists() else None


def main() -> None:
    raw = read_csv(DATA / "metadata.csv", METADATA_FIELDS)
    if not raw:
        print("No image records yet. Charts will be made after data collection.")
        return
    clean = maybe(PROCESSED / "metadata_clean.csv", ["image_id", "dish_label"]) or []
    feats = maybe(PROCESSED / "image_features.csv", ["image_id", "width"]) or []

    # 1. raw vs clean class counts -----------------------------------------
    raw_counts = Counter(norm_label(r["dish_label"]) for r in raw)
    blank = raw_counts.pop("", 0)
    clean_counts = Counter(r["dish_label"] for r in clean)
    x = np.arange(len(LABELS)); w = 0.4
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - w / 2, [raw_counts[l] for l in LABELS], w, label="raw (before cleaning)", color="#9aa5b1")
    if clean:
        b2 = ax.bar(x + w / 2, [clean_counts[l] for l in LABELS], w, label="clean (after cleaning)", color="#2f6f9f")
        ax.bar_label(b2, fontsize=8)
    ax.bar_label(b1, fontsize=8)
    ax.set_xticks(x, [SHORT[l] for l in LABELS], rotation=20)
    ax.set_ylabel("images"); ax.set_title(f"Images per dish (raw n={len(raw)}; {blank} with blank/ambiguous label not shown)")
    ax.legend(); save(fig, "raw_class_counts.png")

    # 2. missing data -------------------------------------------------------
    cols = ["image_id", "local_path", "dish_label", "source_id", "notes"]
    miss = [sum(1 for r in raw if not (r.get(c) or "").strip()) for c in cols]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bars = ax.bar(cols, miss, color="#c0504d")
    ax.bar_label(bars); ax.set_ylabel("rows with missing value")
    ax.set_title("Missing values per metadata column (raw)")
    ax.tick_params(axis="x", rotation=20); save(fig, "missing_data.png")

    if not clean:
        print("Run clean_data.py to enable the remaining charts.")
        return

    # 3. sample images ------------------------------------------------------
    per = 4
    fig, axes = plt.subplots(len(LABELS), per, figsize=(per * 1.8, len(LABELS) * 1.8))
    for i, label in enumerate(LABELS):
        items = sorted((r for r in clean if r["dish_label"] == label), key=lambda r: r["image_id"])
        picks = [items[int(k * len(items) / per)] for k in range(per)] if len(items) >= per else items
        for j in range(per):
            ax = axes[i][j]; ax.axis("off")
            if j < len(picks):
                with Image.open(ROOT / picks[j]["local_path"]) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im = ImageOps.fit(im, (200, 200))
                    ax.imshow(im)
            if j == 0:
                ax.set_title(label, fontsize=8, loc="left")
    fig.suptitle("Sample photos per dish (evenly spaced picks)", y=1.0, fontsize=10)
    save(fig, "sample_images.png")

    # 4. size distributions -------------------------------------------------
    if feats:
        wd = np.array([float(f["width"]) for f in feats]); ht = np.array([float(f["height"]) for f in feats])
        asp = np.array([float(f["aspect_ratio"]) for f in feats])
        kb = np.array([float(f["file_bytes"]) for f in feats]) / 1024
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
        axs[0].scatter(wd, ht, s=8, alpha=0.5, color="#2f6f9f")
        axs[0].axvline(224, ls="--", c="gray", lw=0.8); axs[0].axhline(224, ls="--", c="gray", lw=0.8)
        axs[0].set_xlabel("width (px)"); axs[0].set_ylabel("height (px)"); axs[0].set_title("Image dimensions")
        axs[1].hist(asp, bins=30, color="#2f6f9f"); axs[1].set_xlabel("long side / short side")
        axs[1].set_title("Aspect ratio")
        axs[2].hist(kb, bins=30, color="#2f6f9f"); axs[2].set_xlabel("file size (KB)")
        axs[2].set_title("File size")
        save(fig, "image_size_distribution.png")

        # 6. feature correlation --------------------------------------------
        names = ["width", "height", "aspect_ratio", "file_bytes", "brightness", "mean_r", "mean_g", "mean_b"]
        mat = np.array([[float(f[n]) for n in names] for f in feats])
        if len(feats) > 2 and (mat.std(axis=0) > 0).all():
            corr = np.corrcoef(mat, rowvar=False)
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(range(len(names)), names, rotation=45, ha="right"); ax.set_yticks(range(len(names)), names)
            for a in range(len(names)):
                for b in range(len(names)):
                    ax.text(b, a, f"{corr[a, b]:.2f}", ha="center", va="center", fontsize=7)
            fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Correlation of image-level features")
            save(fig, "feature_correlation.png")

    # 5. class by source ----------------------------------------------------
    sources = sorted({r["source_id"] for r in clean})
    fig, ax = plt.subplots(figsize=(9, 4.5)); bottom = np.zeros(len(LABELS))
    for s in sources:
        vals = np.array([sum(1 for r in clean if r["dish_label"] == l and r["source_id"] == s) for l in LABELS])
        ax.bar(range(len(LABELS)), vals, bottom=bottom, label=s); bottom += vals
    ax.set_xticks(range(len(LABELS)), [SHORT[l] for l in LABELS], rotation=20)
    ax.set_ylabel("clean images"); ax.set_title("Image source per dish"); ax.legend(fontsize=8)
    save(fig, "class_by_source.png")

    # 7. split distribution -------------------------------------------------
    split = maybe(PROCESSED / "split.csv", ["image_id", "dish_label", "split"])
    if split:
        fig, ax = plt.subplots(figsize=(9, 4.5)); bottom = np.zeros(len(LABELS))
        for s, colour in (("train", "#2f6f9f"), ("val", "#e0a030"), ("test", "#c0504d")):
            vals = np.array([sum(1 for r in split if r["dish_label"] == l and r["split"] == s) for l in LABELS])
            ax.bar(range(len(LABELS)), vals, bottom=bottom, label=s, color=colour); bottom += vals
        ax.set_xticks(range(len(LABELS)), [SHORT[l] for l in LABELS], rotation=20)
        ax.set_ylabel("images"); ax.set_title("Train / validation / test split per dish"); ax.legend()
        save(fig, "split_distribution.png")

    # 8. nutrition ----------------------------------------------------------
    nut = maybe(PROCESSED / "nutrition_clean.csv", ["dish_label", "kcal_per_100g"])
    if nut:
        labels = [r["dish_label"] for r in nut]
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].bar(range(len(labels)), [float(r["kcal_per_100g"]) for r in nut], color="#6a9955")
        axs[0].set_xticks(range(len(labels)), [SHORT[l] for l in labels], rotation=25)
        axs[0].set_ylabel("kcal per 100 g"); axs[0].set_title("Energy")
        width = 0.27
        for k, (col, name) in enumerate((("protein_g_per_100g", "protein"), ("carbs_g_per_100g", "carbs"),
                                         ("fat_g_per_100g", "fat"))):
            axs[1].bar(np.arange(len(labels)) + (k - 1) * width, [float(r[col]) for r in nut], width, label=name)
        axs[1].set_xticks(range(len(labels)), [SHORT[l] for l in labels], rotation=25)
        axs[1].set_ylabel("g per 100 g"); axs[1].set_title("Macronutrients"); axs[1].legend()
        save(fig, "nutrition_macros.png")


if __name__ == "__main__":
    main()
