import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from summarize_experiment import load_threshold_csv, markdown_table


class LoadThresholdCsvTest(unittest.TestCase):
    def test_loads_threshold_metrics(self):
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "thresholds.csv"
            csv_path.write_text(
                "threshold,iou,f1,precision,recall,accuracy,"
                "predicted_positive_ratio,target_positive_ratio\n"
                "0.7,0.4,0.5,0.6,0.7,0.8,0.1,0.2\n"
            )

            results = load_threshold_csv(csv_path)

        self.assertEqual(results[0.7]["iou"], 0.4)
        self.assertEqual(results[0.7]["target_positive_ratio"], 0.2)

    def test_rejects_empty_csv(self):
        with TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "empty.csv"
            csv_path.write_text(
                "threshold,iou,f1,precision,recall,accuracy,"
                "predicted_positive_ratio,target_positive_ratio\n"
            )

            with self.assertRaises(ValueError):
                load_threshold_csv(csv_path)


class MarkdownTableTest(unittest.TestCase):
    def test_formats_experiment_row(self):
        table = markdown_table(
            [
                {
                    "name": "exp",
                    "epoch": 8,
                    "threshold": 0.5,
                    "iou": 0.37757,
                    "f1": 0.54816,
                    "precision": 0.41283,
                    "recall": 0.81549,
                    "predicted_positive_ratio": 0.07948,
                    "target_positive_ratio": 0.04024,
                    "parameters_m": "3.979M",
                }
            ]
        )

        self.assertIn("| exp | 8 | 0.5000 | 0.3776 |", table)
        self.assertIn("Params", table)


if __name__ == "__main__":
    unittest.main()
