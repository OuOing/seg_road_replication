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
