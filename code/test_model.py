import unittest

import torch
import torch.nn as nn

from model import SegRoad, SegRoadDecoder


class SegRoadDecoderTest(unittest.TestCase):
    def test_uses_batch_independent_normalization(self):
        decoder = SegRoadDecoder([32, 64, 160, 256])

        normalizations = [
            module for module in decoder.modules() if isinstance(module, nn.GroupNorm)
        ]

        self.assertEqual(len(normalizations), 1)

    def test_output_shapes(self):
        model = SegRoad(model_size="s")
        image = torch.randn(1, 3, 64, 64)

        seg_out, pcs_out = model(image)

        self.assertEqual(tuple(seg_out.shape), (1, 1, 64, 64))
        self.assertEqual(tuple(pcs_out.shape), (1, 8, 64, 64))


if __name__ == "__main__":
    unittest.main()
