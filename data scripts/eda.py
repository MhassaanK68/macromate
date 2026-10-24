"""Plot the raw class distribution from data/metadata.csv."""

import csv
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "metadata.csv"
OUTPUT = ROOT / "figures" / "raw_class_counts.png"
LABELS = [
    "chicken biryani", "chicken pulao", "chicken karahi", "daal",
    "nihari", "haleem", "samosa", "chapati",
]


def main() -> None:
    with METADATA.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "dish_label" not in (reader.fieldnames or []):
            raise ValueError("data/metadata.csv needs a dish_label column")
        rows = list(reader)
    if not rows:
        print("No image records yet. Raw class chart will be made after data collection.")
        return

    counts = Counter((row["dish_label"] or "").strip().lower() for row in rows)
    width, height = 1400, 680
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arial.ttf", 30)
        body_font = ImageFont.truetype("arial.ttf", 21)
    except OSError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    draw.text((70, 35), f"Raw images by dish (n={len(rows)})", font=title_font, fill="black")
    left, right, top, bottom = 90, width - 40, 115, height - 170
    draw.line((left, top, left, bottom), fill="black", width=3)
    draw.line((left, bottom, right, bottom), fill="black", width=3)
    max_count = max(counts.values(), default=1)
    bar_slot = (right - left) / len(LABELS)
    for index, label in enumerate(LABELS):
        count = counts[label]
        x0 = left + index * bar_slot + 22
        x1 = left + (index + 1) * bar_slot - 22
        bar_height = (bottom - top - 30) * count / max_count
        draw.rectangle((x0, bottom - bar_height, x1, bottom), fill="#555555")
        draw.text((x0, bottom - bar_height - 30), str(count), font=body_font, fill="black")
        words = label.split()
        if len(words) > 1:
            midpoint = (len(words) + 1) // 2
            lines = [" ".join(words[:midpoint]), " ".join(words[midpoint:])]
        else:
            lines = words
        for line_number, line in enumerate(lines):
            draw.text((x0 - 10, bottom + 12 + 28 * line_number), line, font=body_font, fill="black")
    draw.text((90, height - 35), "Counted before cleaning; labels and files may still need review.",
              font=body_font, fill="#444444")
    OUTPUT.parent.mkdir(exist_ok=True)
    image.save(OUTPUT)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
