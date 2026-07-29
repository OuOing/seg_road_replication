import unittest

import torch

from model_stats import count_parameters, format_millions


class CountParametersTest(unittest.TestCase):
    def test_counts_total_and_trainable_parameters(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 3),
            torch.nn.Linear(3, 1),
        )
        model[1].weight.requires_grad = False

        counts = count_parameters(model)

        self.assertEqual(counts["total"], 13)
        self.assertEqual(counts["trainable"], 10)


class FormatMillionsTest(unittest.TestCase):
    def test_formats_parameter_count_in_millions(self):
        self.assertEqual(format_millions(1_234_567), "1.235M")


if __name__ == "__main__":
    unittest.main()
