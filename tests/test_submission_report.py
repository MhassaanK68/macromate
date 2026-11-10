"""Submission report checks using the real saved experiment, without inference."""

import tempfile
import unittest
from pathlib import Path

from models.config import ROOT, code_hashes
from models.report import load_results, make_report
from scripts.verify_dataset import verify_dataset


class SubmissionReportTests(unittest.TestCase):
    def test_prepared_files_reproduce_the_recorded_experiment(self):
        result = verify_dataset()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["prepared_images"], 400)
        self.assertEqual(result["train_validation_images"], 344)
        self.assertEqual(result["test_files_hashed_without_inference"], 56)

    def test_canonical_package_hashes_existing_implementation(self):
        hashes = code_hashes()
        self.assertEqual(len(hashes), 8)
        self.assertTrue(all(name.startswith("models/") for name in hashes))
        self.assertTrue(all((ROOT / name).is_file() for name in hashes))

    def test_both_report_formats_cover_required_evidence(self):
        training, evaluation, rows = load_results(ROOT / "evidence")
        scratch = ROOT / "tmp"
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            output = Path(temporary)
            make_report(training, evaluation, rows, output, ROOT / "figures/model")
            markdown = (output / "results.md").read_text(encoding="utf-8")
            tex = (output / "model_report.tex").read_text(encoding="utf-8")
            for required in ("architecture.png", "Hyperparameter tuning", "block_high",
                             "Bias Check", "Contribution Table",
                             "0.9641", "0.9985"):
                with self.subTest(required=required):
                    self.assertIn(required, markdown)
                    self.assertIn(required.replace("_", r"\_") if required == "block_high"
                                  else required, tex)
            self.assertNotIn("error_review.json", markdown)
            self.assertIn("Hassan Mehmood", markdown)
            self.assertIn("Mohid Israr", markdown)
            self.assertIn("Adnan Zaka", markdown)
            self.assertIn(r"\end{document}", tex)


if __name__ == "__main__":
    unittest.main()
