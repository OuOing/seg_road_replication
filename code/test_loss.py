import unittest

import torch

from loss import SegRoadLoss, soft_dice_loss


class SoftDiceLossTest(unittest.TestCase):
    def test_prefers_correct_logits(self):
        target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        correct = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])
        incorrect = -correct

        self.assertLess(
            soft_dice_loss(correct, target),
            soft_dice_loss(incorrect, target),
        )

    def test_dice_weight_is_added_to_segmentation_loss(self):
        seg_pred = torch.zeros(1, 1, 2, 2)
        seg_target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        pcs_pred = torch.zeros(1, 8, 2, 2)
        pcs_target = torch.zeros(1, 8, 2, 2)
        plain = SegRoadLoss(alpha=0.2, dice_weight=0.0)
        combined = SegRoadLoss(alpha=0.2, dice_weight=1.0)

        plain_total, _, _ = plain(seg_pred, seg_target, pcs_pred, pcs_target)
        combined_total, _, _ = combined(
            seg_pred, seg_target, pcs_pred, pcs_target
        )

        self.assertGreater(combined_total, plain_total)


if __name__ == "__main__":
    unittest.main()
