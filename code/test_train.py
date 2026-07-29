import unittest

import torch

from train import parse_args, run_epoch, set_optimizer_learning_rate, split_pairs


class SplitPairsTest(unittest.TestCase):
    def test_split_is_reproducible(self):
        pairs = list(range(10))

        first = split_pairs(pairs, val_ratio=0.2, seed=42)
        second = split_pairs(pairs, val_ratio=0.2, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 8)
        self.assertEqual(len(first[1]), 2)


class OptimizerLearningRateTest(unittest.TestCase):
    def test_sets_learning_rate_for_all_parameter_groups(self):
        first = torch.nn.Parameter(torch.tensor([1.0]))
        second = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = torch.optim.AdamW(
            [
                {"params": [first], "lr": 1e-4},
                {"params": [second], "lr": 1e-5},
            ]
        )

        set_optimizer_learning_rate(optimizer, 3e-5)

        self.assertEqual([group["lr"] for group in optimizer.param_groups], [3e-5, 3e-5])


class TrainArgumentTest(unittest.TestCase):
    def test_reset_best_on_resume_defaults_to_false(self):
        args = parse_args(
            [
                "--image-dir",
                "images",
                "--mask-dir",
                "masks",
            ]
        )

        self.assertFalse(args.reset_best_on_resume)

    def test_reset_best_on_resume_can_be_enabled(self):
        args = parse_args(
            [
                "--image-dir",
                "images",
                "--mask-dir",
                "masks",
                "--reset-best-on-resume",
            ]
        )

        self.assertTrue(args.reset_best_on_resume)

    def test_eval_threshold_defaults_to_point_five(self):
        args = parse_args(["--image-dir", "images", "--mask-dir", "masks"])

        self.assertEqual(args.eval_threshold, 0.5)

    def test_eval_threshold_can_be_overridden(self):
        args = parse_args(
            [
                "--image-dir",
                "images",
                "--mask-dir",
                "masks",
                "--eval-threshold",
                "0.8",
            ]
        )

        self.assertEqual(args.eval_threshold, 0.8)


class RunEpochThresholdTest(unittest.TestCase):
    def test_threshold_changes_reported_metrics(self):
        class FixedModel(torch.nn.Module):
            def forward(self, images):
                logits = torch.tensor([[[[-2.0, 0.0], [2.0, 4.0]]]])
                return logits, logits

        loader = [
            (
                torch.zeros(1, 3, 2, 2),
                torch.tensor([[[[0.0, 0.0], [1.0, 1.0]]]]),
                torch.zeros(1, 8, 2, 2),
            )
        ]

        def loss_fn(seg_out, masks, pcs_out, pcs_targets):
            loss = torch.tensor(0.0)
            return loss, loss, loss

        low = run_epoch(
            FixedModel(),
            loader,
            loss_fn,
            optimizer=None,
            device=torch.device("cpu"),
            training=False,
            threshold=0.5,
        )
        high = run_epoch(
            FixedModel(),
            loader,
            loss_fn,
            optimizer=None,
            device=torch.device("cpu"),
            training=False,
            threshold=0.9,
        )

        self.assertGreater(low["recall"], high["recall"])
        self.assertLess(low["precision"], high["precision"])


if __name__ == "__main__":
    unittest.main()
