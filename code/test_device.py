import unittest
from unittest.mock import patch

import torch

from device import select_device


class SelectDeviceTest(unittest.TestCase):
    @patch("device.torch.backends.mps.is_available", return_value=True)
    @patch("device.torch.cuda.is_available", return_value=False)
    def test_auto_selects_mps_before_cpu(self, _cuda_available, _mps_available):
        self.assertEqual(select_device().type, "mps")

    @patch("device.torch.backends.mps.is_available", return_value=False)
    @patch("device.torch.cuda.is_available", return_value=False)
    def test_auto_falls_back_to_cpu(self, _cuda_available, _mps_available):
        self.assertEqual(select_device().type, "cpu")

    @patch("device.torch.backends.mps.is_available", return_value=False)
    def test_rejects_unavailable_mps(self, _mps_available):
        with self.assertRaisesRegex(RuntimeError, "MPS was requested"):
            select_device("mps")


if __name__ == "__main__":
    unittest.main()
