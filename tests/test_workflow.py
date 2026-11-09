"""Regression checks for the simplified training/evaluation artifact workflow."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from model import evaluate as evaluator
from model import train as trainer
from model.config import (DEFAULTS, ROOT, sha256, write_json, read_json, read_predictions,
                          reserve_record, write_predictions)
from model.data import load_records, data_lineage, DishDataset
from model.metrics import binary_roc, classification_metrics, selection_key
from model.report import load_results
from model.train import train_run


class ExperimentCase(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / 'tmp'
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='test_artifacts_', dir=scratch)
        self.folder = Path(self.temporary.name).resolve()
        self.assertTrue(self.folder.is_relative_to(scratch.resolve()))
        self.addCleanup(self.temporary.cleanup)

    def copy_evidence(self):
        for name in ('training.json', 'predictions.csv', 'evaluation.json'):
            shutil.copyfile(ROOT / 'evidence' / name, self.folder / name)
        return self.folder


class SavedResultTests(ExperimentCase):
    def test_three_files_preserve_recorded_predictions_and_metrics(self):
        training, evaluation, rows = load_results(ROOT / 'evidence')
        # The original Windows exporter wrote CRLF. Reconstruct its bytes from
        # the CSV to verify every prediction against the retained original hash.
        original_json = (json.dumps(rows, indent=2, allow_nan=False) + '\n').replace('\n', '\r\n')
        self.assertEqual(hashlib.sha256(original_json.encode('utf-8')).hexdigest(),
                         evaluation['provenance']['original_predictions_sha256'])
        self.assertEqual(training['settings'], DEFAULTS)
        selected = max(training['history'], key=lambda h: (h['validation']['top1_correct'], -h['validation']['loss']))
        self.assertEqual(selected['epoch'], training['result']['best_epoch'])
        self.assertEqual(selected['validation'], training['result']['validation'])
        self.assertEqual(evaluation['metrics']['top1_correct'], 54)
        self.assertEqual(len(rows), 56)
        self.assertEqual({p.name for p in (ROOT / 'evidence').iterdir() if p.is_file()},
                         {'training.json', 'predictions.csv', 'evaluation.json'})

    def test_historical_provenance_preserves_evaluation_correction(self):
        evaluation = read_json(ROOT / 'evidence/evaluation.json')
        provenance = evaluation['provenance']
        correction = provenance['rerun']
        self.assertIn('underscores', correction['reason'])
        self.assertFalse(correction['checkpoint_changed'])
        self.assertFalse(correction['training_or_data_changed'])
        self.assertFalse(provenance['inference_repeated'])
        self.assertIn('prior_attempt', correction)
        self.assertEqual(provenance['historical_quality_checks']['status'], 'passed')
        self.assertTrue(evaluation['protocol']['code_sha256'])

    def test_csv_roundtrip_retains_nested_fields_and_float_precision(self):
        rows = read_predictions(ROOT / 'evidence/predictions.csv')
        path = self.folder / 'predictions.csv'
        write_predictions(path, rows)
        self.assertEqual(read_predictions(path), rows)
        self.assertFalse(path.with_suffix('.temporary').exists())

    def test_modified_csv_is_rejected(self):
        folder = self.copy_evidence()
        path = folder / 'predictions.csv'
        path.write_text(path.read_text() + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'changed'):
            load_results(folder)

    def test_modified_metrics_are_rejected(self):
        folder = self.copy_evidence()
        evaluation = read_json(folder / 'evaluation.json')
        evaluation['metrics']['top1_correct'] = 55
        write_json(folder / 'evaluation.json', evaluation)
        with self.assertRaisesRegex(ValueError, 'Metrics'):
            load_results(folder)

    def test_duplicate_or_inconsistent_prediction_is_rejected(self):
        folder = self.copy_evidence()
        rows = read_predictions(folder / 'predictions.csv')
        rows[1]['image_id'] = rows[0]['image_id']
        write_predictions(folder / 'predictions.csv', rows)
        evaluation = read_json(folder / 'evaluation.json')
        evaluation['predictions_sha256'] = sha256(folder / 'predictions.csv')
        write_json(folder / 'evaluation.json', evaluation)
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            load_results(folder)

    def test_invalid_ranking_is_rejected_even_with_updated_csv_hash(self):
        folder = self.copy_evidence()
        rows = read_predictions(folder / 'predictions.csv')
        rows[0]['ranked_label_ids'].reverse()
        write_predictions(folder / 'predictions.csv', rows)
        evaluation = read_json(folder / 'evaluation.json')
        evaluation['predictions_sha256'] = sha256(folder / 'predictions.csv')
        write_json(folder / 'evaluation.json', evaluation)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            load_results(folder)

    def test_report_rejects_different_checkpoint_identity(self):
        folder = self.copy_evidence()
        training = read_json(folder / 'training.json')
        training['checkpoint']['sha256'] = '0' * 64
        write_json(folder / 'training.json', training)
        with self.assertRaisesRegex(ValueError, 'different models'):
            load_results(folder)


class BoundaryTests(ExperimentCase):
    def test_verify_only_command_never_starts_training(self):
        with patch('sys.argv', ['model.train', '--verify-only', '--evidence', str(self.folder), '--device', 'cpu']), \
                patch.object(trainer, 'verify') as verify, patch.object(trainer, 'train_run') as train:
            trainer.main()
            verify.assert_called_once_with(self.folder, 'cpu')
            train.assert_not_called()

    def test_exclusive_record_does_not_overwrite_previous_attempt(self):
        path = self.folder / 'evaluation.json'
        reserve_record(path, {'status': 'started'})
        with self.assertRaises(FileExistsError):
            reserve_record(path, {'status': 'completed'})
        self.assertEqual(read_json(path)['status'], 'started')

    def test_repeat_evaluation_is_rejected_before_loading_data_or_model(self):
        for status in ('started', 'failed', 'completed'):
            with self.subTest(status=status):
                write_json(self.folder / 'evaluation.json', {'status': status})
                with patch.object(evaluator, 'load_records') as records, patch.object(evaluator, 'load_checkpoint') as model:
                    with self.assertRaisesRegex(ValueError, 'already attempted'):
                        evaluator.evaluate(self.folder, 'cpu')
                    records.assert_not_called()
                    model.assert_not_called()

    def test_existing_training_is_not_overwritten(self):
        before = (ROOT / 'evidence/training.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'Experiment already exists'):
            train_run({}, self.folder / 'weights', 'cpu', ROOT / 'evidence')
        self.assertEqual(before, (ROOT / 'evidence/training.json').read_bytes())

    def test_training_dataset_cannot_open_test_split(self):
        records, _ = load_records()
        with self.assertRaisesRegex(ValueError, 'test is reserved'):
            DishDataset(records, 'test')

    def test_raw_label_normalization_and_source_validation(self):
        records, labels = load_records()
        row = records[0]
        metadata = {row['image_id']: {'dish_label': labels[row['label_id']].upper().replace(' ', '_'), 'source_id': 'known'}}
        evaluator.validate_metadata([row], labels, metadata, {'known': {}})
        with self.assertRaisesRegex(ValueError, 'Unknown source'):
            evaluator.validate_metadata([row], labels, metadata, {})

    def test_inference_failure_is_recorded_in_evaluation_json(self):
        training = read_json(ROOT / 'evidence/training.json')
        write_json(self.folder / 'training.json', training)
        protocol = read_json(ROOT / 'evidence/evaluation.json')['protocol']
        with patch.object(evaluator, 'evaluation_protocol', return_value=protocol), \
                patch.object(evaluator, 'load_checkpoint', side_effect=RuntimeError('injected load failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                evaluator.evaluate(self.folder, 'cpu')
        failed = read_json(self.folder / 'evaluation.json')
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('injected load failure', failed['failure']['error'])
        self.assertEqual({p.name for p in self.folder.iterdir()}, {'training.json', 'evaluation.json'})

    def test_evaluator_writes_only_three_files_and_report_can_read_them(self):
        training = read_json(ROOT / 'evidence/training.json')
        records, labels = load_records()
        training['lineage'] = data_lineage(records)
        write_json(self.folder / 'training.json', training)
        reserved = [r for r in records if r['split'] == 'test']
        payload = {'labels': labels, 'metadata': {'lineage': training['lineage'],
                   'settings': training['settings'], 'epoch': training['result']['best_epoch']}}

        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, unused):
                pass
            def __len__(self):
                return len(reserved)
            def __getitem__(self, index):
                row = reserved[index]
                # Numerical inputs only; no held-out image pixels are read.
                return torch.tensor(row['label_id']), row['label_id'], row['image_id']

        class SyntheticModel:
            def __call__(self, ids):
                return torch.nn.functional.one_hot(ids, len(labels)).float() * 4

        with patch.object(evaluator, 'HeldOutDataset', SyntheticDataset), \
                patch.object(evaluator, 'read_checkpoint', return_value=payload), \
                patch.object(evaluator, 'load_checkpoint', return_value=(SyntheticModel(), {})):
            evaluation = evaluator.evaluate(self.folder, 'cpu')
        self.assertEqual(evaluation['status'], 'completed')
        self.assertEqual(evaluation['metrics']['top1_correct'], 56)
        self.assertEqual(evaluation['error_reviews'], [])
        self.assertEqual({p.name for p in self.folder.iterdir()},
                         {'training.json', 'predictions.csv', 'evaluation.json'})
        self.assertEqual(len(load_results(self.folder)[2]), 56)

    def test_changed_training_data_is_rejected_before_checkpoint_loading(self):
        training = read_json(ROOT / 'evidence/training.json')
        training['lineage'] = {'changed': True}
        records, labels = load_records()
        with patch.object(evaluator, 'read_checkpoint') as checkpoint:
            with self.assertRaisesRegex(ValueError, 'must agree'):
                evaluator.evaluation_protocol(training, records, labels)
            checkpoint.assert_not_called()


class MetricTests(unittest.TestCase):
    def test_auc_ties_and_independent_pairwise_definition(self):
        truth, scores = np.array([True, False, True, False]), np.array([0.8, 0.8, 0.6, 0.2])
        expected = np.mean([(a > b) + 0.5 * (a == b) for a in scores[truth] for b in scores[~truth]])
        self.assertEqual(binary_roc(truth, scores)['auc'], expected)
        self.assertIsNone(binary_roc([True, True], [0.9, 0.4])['auc'])

    def test_classification_and_selection_are_count_first(self):
        metrics = classification_metrics([0, 1, 2], [[0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.1, 0.1, 0.8]], ['a', 'b', 'c'])
        self.assertEqual(metrics['top1_correct'], 2)
        self.assertEqual(metrics['confusion_matrix'], [[1, 0, 0], [1, 0, 0], [0, 0, 1]])
        self.assertGreater(selection_key({'top1_correct': 3, 'loss': 0.8}),
                           selection_key({'top1_correct': 2, 'loss': 0.1}))


if __name__ == '__main__':
    unittest.main()
