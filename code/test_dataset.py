import tempfile
import unittest
from pathlib import Path

from dataset import select_pairs_from_list


class SelectPairsFromListTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.pairs = [
            (root / "images" / "area_001.jpg", root / "masks" / "area_001.png"),
            (root / "images" / "area_002.jpg", root / "masks" / "area_002.png"),
        ]
        self.split_file = root / "split.txt"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_selects_in_file_order_and_allows_comments(self):
        self.split_file.write_text("area_002 # validation\n\narea_001.jpg\n")

        selected = select_pairs_from_list(self.pairs, self.split_file)

        self.assertEqual([pair[0].stem for pair in selected], ["area_002", "area_001"])

    def test_rejects_unknown_sample(self):
        self.split_file.write_text("missing_sample\n")

        with self.assertRaisesRegex(ValueError, "no image/mask pair"):
            select_pairs_from_list(self.pairs, self.split_file)

    def test_rejects_duplicate_sample(self):
        self.split_file.write_text("area_001\narea_001.jpg\n")

        with self.assertRaisesRegex(ValueError, "Duplicate sample"):
            select_pairs_from_list(self.pairs, self.split_file)


if __name__ == "__main__":
    unittest.main()
