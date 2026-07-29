import unittest

import numpy as np

from visualize_predictions import choose_pairs, error_overlay


class ChoosePairsTest(unittest.TestCase):
    def test_selection_is_reproducible_and_ordered(self):
        pairs = list(range(20))
        first = choose_pairs(pairs, sample_count=5, seed=42)
        second = choose_pairs(pairs, sample_count=5, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first))
        self.assertEqual(len(first), 5)

    def test_rejects_non_positive_sample_count(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            choose_pairs([1], sample_count=0, seed=42)


class ErrorOverlayTest(unittest.TestCase):
    def test_colors_true_false_and_missed_road_pixels(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        target = np.array([[1, 0], [1, 0]], dtype=bool)
        prediction = np.array([[1, 1], [0, 0]], dtype=bool)

        overlay = error_overlay(image, target, prediction, alpha=1.0)

        np.testing.assert_array_equal(overlay[0, 0], [46, 204, 113])
        np.testing.assert_array_equal(overlay[0, 1], [231, 76, 60])
        np.testing.assert_array_equal(overlay[1, 0], [52, 152, 219])
        np.testing.assert_array_equal(overlay[1, 1], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
