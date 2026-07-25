"""Dataset utilities for the Seg-Road reproduction project."""

from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from pcs import generate_pcs_labels_numpy


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def collect_image_mask_pairs(image_dir, mask_dir):
    """Match images and masks by filename stem."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    masks = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }

    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = masks.get(image_path.stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))

    if not pairs:
        raise FileNotFoundError(
            f"No matching image/mask pairs found in {image_dir} and {mask_dir}."
        )
    return pairs


def select_pairs_from_list(pairs, split_file):
    """Select image/mask pairs using one sample stem per line."""
    split_file = Path(split_file)
    sample_names = []
    seen = set()
    for line_number, raw_line in enumerate(split_file.read_text().splitlines(), 1):
        sample_name = raw_line.split("#", 1)[0].strip()
        if not sample_name:
            continue
        sample_name = Path(sample_name).stem
        if sample_name in seen:
            raise ValueError(
                f"Duplicate sample '{sample_name}' in {split_file}:{line_number}."
            )
        seen.add(sample_name)
        sample_names.append(sample_name)

    if not sample_names:
        raise ValueError(f"Split file is empty: {split_file}")

    pairs_by_stem = {image_path.stem: pair for pair in pairs for image_path in pair[:1]}
    missing = [name for name in sample_names if name not in pairs_by_stem]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(
            f"{len(missing)} samples from {split_file} have no image/mask pair: "
            f"{preview}{suffix}"
        )
    return [pairs_by_stem[name] for name in sample_names]


class RoadDataset(Dataset):
    """Load an RGB image, binary road mask, and its 8-channel PCS target."""

    def __init__(self, pairs, image_size=(512, 512), augment=False, pcs_radius=2):
        self.pairs = list(pairs)
        self.image_size = tuple(image_size)
        self.augment = augment
        self.pcs_radius = pcs_radius

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        mask = mask.resize((self.image_size[1], self.image_size[0]), Image.NEAREST)

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                image_array = np.fliplr(image_array).copy()
                mask_array = np.fliplr(mask_array).copy()
            if np.random.rand() < 0.5:
                image_array = np.flipud(image_array).copy()
                mask_array = np.flipud(mask_array).copy()

        pcs_target = generate_pcs_labels_numpy(
            mask_array.astype(np.uint8), r=self.pcs_radius
        ).astype(np.float32)

        image_array = np.transpose(image_array, (2, 0, 1))
        mask_array = mask_array[None, ...]
        return image_array, mask_array, pcs_target


if __name__ == "__main__":
    print("Dataset utilities loaded.")
