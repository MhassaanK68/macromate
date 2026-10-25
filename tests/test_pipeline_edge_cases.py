"""Regression checks for data validation and generated outputs."""

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "data scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class PipelineEdgeCases(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.images = self.data / "images"
        self.images.mkdir(parents=True)
        self.processed = self.data / "processed"
        for name, colour in (("one", "red"), ("two", "blue")):
            Image.new("RGB", (240, 240), colour).save(self.images / f"{name}.png")

    def prepare_cleaner(self, recorded_hash=""):
        write_csv(self.data / "sources.csv", ["source_id"], [{"source_id": "test"}])
        self.nutrition_fields = ["dish_label", "kcal_per_100g", "protein_g_per_100g",
                                 "carbs_g_per_100g", "fat_g_per_100g", "source",
                                 "recipe_assumptions"]
        self.nutrition_row = {"dish_label": "daal", "kcal_per_100g": "98",
                              "protein_g_per_100g": "10", "carbs_g_per_100g": "10",
                              "fat_g_per_100g": "2", "source": "test",
                              "recipe_assumptions": "test"}
        write_csv(self.data / "nutrition.csv", self.nutrition_fields, [self.nutrition_row])
        write_csv(self.data / "metadata.csv",
                  ["image_id", "local_path", "dish_label", "source_id", "notes", "sha256"],
                  [{"image_id": "one", "local_path": "data/images/one.png",
                    "dish_label": "daal", "source_id": "test", "notes": "",
                    "sha256": recorded_hash}])
        clean = load_script("clean_data")
        clean.DATA = self.data
        clean.ROOT = self.root
        clean.RAW_DIRS = [self.data / "raw", self.images]
        clean.PROCESSED = self.processed
        clean.sources_path = lambda: self.data / "sources.csv"
        return clean

    def test_wrong_recorded_hash_is_excluded(self):
        clean = self.prepare_cleaner("0" * 64)
        clean.main()
        summary = json.loads((self.processed / "cleaning_summary.json").read_text())
        self.assertEqual(summary["final_clean_records"], 0)
        with (self.processed / "exclusions.csv").open(newline="") as handle:
            exclusions = list(csv.DictReader(handle))
        self.assertEqual(exclusions[0]["reason"], "image SHA-256 does not match metadata")

    def test_invalid_nutrition_replaces_old_clean_output(self):
        clean = self.prepare_cleaner()
        clean.main()
        self.nutrition_row["kcal_per_100g"] = "-1"
        write_csv(self.data / "nutrition.csv", self.nutrition_fields, [self.nutrition_row])
        clean.main()
        with (self.processed / "nutrition_clean.csv").open(newline="") as handle:
            self.assertEqual(list(csv.DictReader(handle)), [])

    def test_preprocessed_names_are_unique_and_stale_pngs_are_removed(self):
        prep = load_script("preprocess")
        prep.ROOT = self.root
        prep.PROCESSED = self.processed
        prep.OUT_DIR = self.processed / "clip_224"
        fields = ["image_id", "local_path", "dish_label"]
        write_csv(self.processed / "metadata_clean.csv", fields, [
            {"image_id": "a/b", "local_path": "data/images/one.png", "dish_label": "daal"},
            {"image_id": "a_b", "local_path": "data/images/two.png", "dish_label": "daal"},
        ])
        prep.main()
        self.assertEqual(len(list(prep.OUT_DIR.glob("*.png"))), 2)
        with (self.processed / "clip_manifest.csv").open(newline="") as handle:
            manifest = list(csv.DictReader(handle))
        self.assertEqual(len({row["clip_path"] for row in manifest}), 2)

        write_csv(self.processed / "metadata_clean.csv", fields, [
            {"image_id": "other", "local_path": "data/images/one.png", "dish_label": "daal"},
        ])
        prep.main()
        self.assertEqual(len(list(prep.OUT_DIR.glob("*.png"))), 1)
        (self.processed / "metadata_clean.csv").unlink()
        prep.main()
        self.assertEqual(list(prep.OUT_DIR.glob("*.png")), [])
        self.assertFalse((self.processed / "clip_manifest.csv").exists())

    def test_eda_removes_stale_charts_when_data_is_empty(self):
        eda = load_script("eda")
        eda.DATA = self.data
        eda.FIGURES = self.root / "figures"
        eda.FIGURES.mkdir()
        for name in eda.GENERATED_FIGURES:
            (eda.FIGURES / name).write_bytes(b"old chart")
        write_csv(self.data / "metadata.csv",
                  ["image_id", "local_path", "dish_label", "source_id", "notes"], [])
        eda.main()
        self.assertEqual(list(eda.FIGURES.glob("*.png")), [])


if __name__ == "__main__":
    unittest.main()
