import unittest

from train import split_pairs


class SplitPairsTest(unittest.TestCase):
    def test_split_is_reproducible(self):
        pairs = list(range(10))

        first = split_pairs(pairs, val_ratio=0.2, seed=42)
        second = split_pairs(pairs, val_ratio=0.2, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first[0]), 8)
        self.assertEqual(len(first[1]), 2)


if __name__ == "__main__":
    unittest.main()
