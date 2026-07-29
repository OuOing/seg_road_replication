import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from threshold_sweep import (
    best_threshold,
    evaluate_thresholds,
    parse_thresholds,
    rows_from_results,
    write_results_csv,
)


class ParseThresholdsTest(unittest.TestCase):
    def test_parses_comma_separated_thresholds(self):
        self.assertEqual(parse_thresholds("0.3, 0.5,0.7"), [0.3, 0.5, 0.7])

    def test_rejects_empty_thresholds(self):
        with self.assertRaises(Exception):
            parse_thresholds(" , ")

    def test_rejects_out_of_range_threshold(self):
        with self.assertRaises(Exception):
            parse_thresholds("0.5,1.2")


class ThresholdDecisionTest(unittest.TestCase):
    def test_higher_threshold_predicts_fewer_positive_pixels(self):
        probabilities = torch.tensor([[[[0.2, 0.4], [0.6, 0.8]]]]).numpy()

        low_count = np.count_nonzero(probabilities >= 0.3)
        high_count = np.count_nonzero(probabilities >= 0.7)

        self.assertGreater(low_count, high_count)

    def test_evaluates_multiple_thresholds_in_one_loader_pass(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, images):
                self.calls += 1
                logits = torch.tensor([[[[-2.0, 0.0], [2.0, 4.0]]]])
                return logits, logits

        images = torch.zeros((1, 3, 2, 2))
        masks = torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]])
        pcs = torch.zeros((1, 8, 2, 2))
        loader = DataLoader(TensorDataset(images, masks, pcs), batch_size=1)
        model = ToyModel()

        results = evaluate_thresholds(model, loader, [0.5, 0.9], torch.device("cpu"))

        self.assertEqual(model.calls, 1)
        self.assertIn(0.5, results)
        self.assertIn(0.9, results)

    def test_selects_best_threshold_by_metric(self):
        results = {
            0.5: {"iou": 0.2, "f1": 0.3},
            0.7: {"iou": 0.4, "f1": 0.5},
        }

        threshold, metrics = best_threshold(results, "iou")

        self.assertEqual(threshold, 0.7)
        self.assertEqual(metrics["iou"], 0.4)

    def test_rows_are_sorted_by_threshold(self):
        results = {
            0.7: {
                "iou": 0.4,
                "f1": 0.5,
                "precision": 0.6,
                "recall": 0.7,
                "accuracy": 0.8,
                "predicted_positive_ratio": 0.1,
                "target_positive_ratio": 0.2,
            },
            0.5: {
                "iou": 0.2,
                "f1": 0.3,
                "precision": 0.4,
                "recall": 0.5,
                "accuracy": 0.6,
                "predicted_positive_ratio": 0.3,
                "target_positive_ratio": 0.2,
            },
        }

        rows = rows_from_results(results)

        self.assertEqual([row["threshold"] for row in rows], [0.5, 0.7])

    def test_writes_results_csv(self):
        results = {
            0.5: {
                "iou": 0.2,
                "f1": 0.3,
                "precision": 0.4,
                "recall": 0.5,
                "accuracy": 0.6,
                "predicted_positive_ratio": 0.3,
                "target_positive_ratio": 0.2,
            }
        }

        with TemporaryDirectory() as tmpdir:
            output_csv = Path(tmpdir) / "sweep.csv"
            write_results_csv(results, output_csv)

            text = output_csv.read_text()

        self.assertIn("threshold,iou,f1,precision", text)
        self.assertIn("0.5,0.2,0.3", text)


if __name__ == "__main__":
    unittest.main()
