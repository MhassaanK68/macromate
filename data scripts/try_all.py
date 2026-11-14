"""Smoke test: run the model on 2 photos of every dish.

This only proves the code runs for all 8 dishes. These photos were used in
training, so do NOT report this as an accuracy result.

Run from the repo root:  python try_all.py
"""

import os

from app_core.inference import predict_dishes

root = os.path.join("data", "images")
for dish in sorted(os.listdir(root)):
    folder = os.path.join(root, dish)
    for name in sorted(os.listdir(folder))[:2]:
        with open(os.path.join(folder, name), "rb") as file:
            top = predict_dishes(file.read())
        guess = top[0]["label"]
        print(f"{dish:16} -> {guess:16} score {top[0]['score']:.3f}  "
              f"{'OK' if guess == dish else 'CHECK'}")
