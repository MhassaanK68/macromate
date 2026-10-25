"""Phase 2, step 4.4: reproducible, leakage-aware train/validation/test split.

* Stratified by dish: each class is divided ~70/15/15 (see common.SPLIT_RATIOS).
* Seeded (common.SPLIT_SEED) so the same input always gives the same split.
* Images that belong together are placed in ONE split, so near-identical photos
  and photos of the same physical plate cannot leak between train and test.
  "Together" = near-duplicate pairs from clean_data.py, plus any images that
  share a value in the optional `group_id` column of data/metadata.csv.
* The test split is reserved for the final zero-shot vs fine-tuned comparison.

Run after clean_data.py:  python "data scripts/split_data.py"
"""

import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    LABEL_TO_ID, LABELS, NEAR_DUP_MAX_DISTANCE, PROCESSED, SPLIT_RATIOS,
    SPLIT_SEED, hamming, read_csv, write_csv, write_json,
)

SPLITS = ["train", "val", "test"]


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)   # smallest id is the root: deterministic


def main() -> None:
    clean_path = PROCESSED / "metadata_clean.csv"
    if not clean_path.exists():
        for name in ("split.csv", "split_summary.json"):
            (PROCESSED / name).unlink(missing_ok=True)
        print("data/processed/metadata_clean.csv not found. Run clean_data.py first.")
        return
    rows = read_csv(clean_path, ["image_id", "local_path", "dish_label"])
    if not rows:
        for name in ("split.csv", "split_summary.json"):
            (PROCESSED / name).unlink(missing_ok=True)
        print("No clean images to split.")
        return
    rows.sort(key=lambda r: r["image_id"])
    by_id = {r["image_id"]: r for r in rows}
    features = {r["image_id"]: r for r in read_csv(PROCESSED / "image_features.csv",
                                                   ["image_id", "dhash"])}

    # ---- group images that must stay together ------------------------------
    uf = UnionFind(by_id)
    pairs = read_csv(PROCESSED / "near_duplicate_pairs.csv", ["image_a", "image_b"])
    for p in pairs:
        if p["image_a"] in by_id and p["image_b"] in by_id:
            uf.union(p["image_a"], p["image_b"])
    first_in_group = {}
    for r in rows:
        g = (r.get("group_id") or "").strip()
        if g:
            if g in first_in_group:
                uf.union(first_in_group[g], r["image_id"])
            else:
                first_in_group[g] = r["image_id"]
    clusters = defaultdict(list)
    for r in rows:
        clusters[uf.find(r["image_id"])].append(r["image_id"])

    # ---- stratified, cluster-aware assignment ------------------------------
    rng = random.Random(SPLIT_SEED)
    class_total = Counter(r["dish_label"] for r in rows)
    target = {l: {s: class_total[l] * SPLIT_RATIOS[s] for s in SPLITS} for l in LABELS}
    current = {l: {s: 0 for s in SPLITS} for l in LABELS}
    assignment = {}
    clusters_by_label = defaultdict(list)
    for cid, members in clusters.items():
        majority = Counter(by_id[m]["dish_label"] for m in members).most_common()
        top = sorted(l for l, c in majority if c == majority[0][1])[0]
        clusters_by_label[top].append((cid, sorted(members)))
    for label in LABELS:
        items = sorted(clusters_by_label[label])             # deterministic base order
        rng.shuffle(items)                                   # seeded randomness
        items.sort(key=lambda it: -len(it[1]))               # place big clusters first
        for cid, members in items:
            deficit = {s: target[label][s] - current[label][s] for s in SPLITS}
            chosen = max(SPLITS, key=lambda s: (deficit[s], -SPLITS.index(s)))
            for m in members:
                assignment[m] = chosen
                current[by_id[m]["dish_label"]][chosen] += 1

    out = [{
        "image_id": r["image_id"], "local_path": r["local_path"],
        "dish_label": r["dish_label"], "label_id": LABEL_TO_ID[r["dish_label"]],
        "split": assignment[r["image_id"]], "cluster_id": uf.find(r["image_id"]),
        "rights_cleared": r.get("rights_cleared", ""),
    } for r in rows]
    write_csv(PROCESSED / "split.csv",
              ["image_id", "local_path", "dish_label", "label_id", "split", "cluster_id", "rights_cleared"], out)

    # ---- leakage checks -----------------------------------------------------
    cluster_splits = defaultdict(set)
    for r in out:
        cluster_splits[r["cluster_id"]].add(r["split"])
    clusters_spanning = sum(1 for s in cluster_splits.values() if len(s) > 1)
    close_cross_split = 0
    min_dist = None
    ids = [r["image_id"] for r in out]
    h = {i: int(features[i]["dhash"], 16) for i in ids}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            if assignment[ids[a]] != assignment[ids[b]]:
                d = hamming(h[ids[a]], h[ids[b]])
                min_dist = d if min_dist is None else min(min_dist, d)
                if d <= NEAR_DUP_MAX_DISTANCE:
                    close_cross_split += 1
    counts = {l: {s: current[l][s] for s in SPLITS} for l in LABELS}
    empty = [f"{l}/{s}" for l in LABELS for s in SPLITS if class_total[l] and counts[l][s] == 0]
    summary = {
        "seed": SPLIT_SEED, "ratios": SPLIT_RATIOS, "total_images": len(out),
        "split_totals": {s: sum(counts[l][s] for l in LABELS) for s in SPLITS},
        "per_class_counts": counts,
        "rights_cleared_images_per_split": {s: sum(1 for r in out if r["split"] == s and r["rights_cleared"] == "True") for s in SPLITS},
        "grouped_clusters_with_more_than_one_image": sum(1 for m in clusters.values() if len(m) > 1),
        "leakage_checks": {
            "clusters_spanning_multiple_splits": clusters_spanning,
            "near_duplicate_pairs_across_splits": close_cross_split,
            "min_hash_distance_across_splits_bits": min_dist,
            "class_split_cells_that_are_empty": empty,
        },
        "note": "Test split is reserved for the final zero-shot vs fine-tuned CLIP comparison.",
    }
    write_json(PROCESSED / "split_summary.json", summary)
    t = summary["split_totals"]
    print(f"Split {len(out)} images (seed {SPLIT_SEED}): train {t['train']} | val {t['val']} | test {t['test']}")
    print(f"  leakage: {clusters_spanning} clusters span splits, "
          f"{close_cross_split} near-duplicate pairs across splits")
    if empty:
        print("  WARNING: empty class/split cells (class too small):", ", ".join(empty))
    print("Wrote split.csv and split_summary.json")


if __name__ == "__main__":
    main()
