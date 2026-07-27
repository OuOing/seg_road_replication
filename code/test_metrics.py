import unittest

from metrics import metrics_from_counts


class MetricsFromCountsTest(unittest.TestCase):
    def test_reports_predicted_and_target_positive_ratios(self):
        metrics = metrics_from_counts((2, 1, 2, 5))

        self.assertAlmostEqual(metrics["predicted_positive_ratio"], 0.3)
        self.assertAlmostEqual(metrics["target_positive_ratio"], 0.4)


if __name__ == "__main__":
    unittest.main()
