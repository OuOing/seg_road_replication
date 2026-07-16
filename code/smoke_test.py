"""Run a tiny end-to-end check without downloading a real dataset."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import RoadDataset, collect_image_mask_pairs
from loss import SegRoadLoss
from model import SegRoad


def create_example_dataset(root, count=2, size=64):
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    for index in range(count):
        image = np.zeros((size, size, 3), dtype=np.uint8)
        image[:, :, 1] = 80
        mask = np.zeros((size, size), dtype=np.uint8)
        start = 12 + index * 4
        mask[start : start + 5, 8:-8] = 1
        image[mask == 1] = (180, 180, 180)
        Image.fromarray(image).save(image_dir / f"road_{index:03d}.png")
        Image.fromarray(mask).save(mask_dir / f"road_{index:03d}.png")
    return image_dir, mask_dir


def main():
    temporary_root = Path(tempfile.mkdtemp(prefix="segroad-smoke-"))
    try:
        image_dir, mask_dir = create_example_dataset(temporary_root)
        pairs = collect_image_mask_pairs(image_dir, mask_dir)
        dataset = RoadDataset(pairs, image_size=(64, 64), augment=False)
        loader = DataLoader(dataset, batch_size=2)
        images, masks, pcs_targets = next(iter(loader))

        model = SegRoad(model_size="s")
        loss_fn = SegRoadLoss(alpha=0.2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        seg_out, pcs_out = model(images)
        total_loss, _, _ = loss_fn(seg_out, masks, pcs_out, pcs_targets)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print(f"images: {tuple(images.shape)}")
        print(f"masks: {tuple(masks.shape)}")
        print(f"pcs_targets: {tuple(pcs_targets.shape)}")
        print(f"seg_out: {tuple(seg_out.shape)}")
        print(f"pcs_out: {tuple(pcs_out.shape)}")
        print(f"loss: {total_loss.item():.6f}")
        print("Smoke test passed.")
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


if __name__ == "__main__":
    main()
