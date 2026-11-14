"""Measure how fast the model starts and predicts (for the report).

Run from the repo root:  python measure.py
Write down the numbers together with your laptop's CPU, RAM and the date.
"""

import os
import statistics
import time

start = time.perf_counter()
from app_core.inference import get_recognizer, predict_dishes

get_recognizer()
print("Cold start (import + model load):", round(time.perf_counter() - start, 2), "seconds")

photos = []
root = os.path.join("data", "images")
for dish in sorted(os.listdir(root)):
    folder = os.path.join(root, dish)
    photos += [os.path.join(folder, f) for f in sorted(os.listdir(folder))[:2]]
photos = photos[:10]

times = []
for path in photos:
    with open(path, "rb") as file:
        data = file.read()
    begin = time.perf_counter()
    predict_dishes(data)
    times.append(time.perf_counter() - begin)

print("Photos timed:", len(times))
print("Median:", round(statistics.median(times), 3), "seconds")
print("Slowest:", round(max(times), 3), "seconds")
