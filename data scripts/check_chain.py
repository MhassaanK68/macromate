"""Check your whole backend on ONE photo: model -> top 3 -> nutrition.

Run from the repo root:
    python check_chain.py                  (uses a sample samosa photo)
    python check_chain.py my_photo.jpg (uses your own photo)
"""

import os
import sys

from app_core.inference import ImageError, predict_dishes
from app_core.nutrition import calculate_nutrition


def default_photo():
    folder = os.path.join("data", "images", "samosa")
    return os.path.join(folder, sorted(os.listdir(folder))[0])


photo = sys.argv[1] if len(sys.argv) > 1 else default_photo()
print("Photo:", photo)

# 1. Real model: top 3 dishes
with open(photo, "rb") as file:
    top = predict_dishes(file.read())
for rank, item in enumerate(top, start=1):
    print(f"  {rank}. {item['dish']}  (ranking score {item['score']:.3f})")

# 2. Nutrition for the top dish at 150 g
result = calculate_nutrition(top[0]["label"], 150)
print(f"\nNutrition for 150 g of {result['dish']}:")
print(f"  kcal {result['kcal']:.2f} | protein {result['protein_g']:.2f} g | "
      f"carbs {result['carbs_g']:.2f} g | fat {result['fat_g']:.2f} g")

# 3. Bad uploads must give a friendly error, not a crash
print("\nBad upload checks:")
for bad in (b"", b"this is not an image"):
    try:
        predict_dishes(bad)
        print("  PROBLEM: bad upload was accepted")
    except ImageError as error:
        print("  OK:", error)
