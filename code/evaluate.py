"""Evaluate a Seg-Road checkpoint on an image/mask directory pair."""

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import RoadDataset, collect_image_mask_pairs, select_pairs_from_list
from device import DEVICE_CHOICES, select_device
from metrics import confusion_counts, merge_counts, metrics_from_counts
from model import SegRoad


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument(
        "--split-list",
        help="Text file containing one test sample stem per line.",
    )
    parser.add_argument("--model-size", choices=("s", "m", "l"), default="s")
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = SegRoad(model_size=args.model_size).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    pairs = collect_image_mask_pairs(args.image_dir, args.mask_dir)
    if args.split_list:
        pairs = select_pairs_from_list(pairs, args.split_list)
    dataset = RoadDataset(
        pairs, image_size=(args.image_height, args.image_width), augment=False
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    counts = []
    with torch.no_grad():
        for images, masks, _ in loader:
            seg_out, _ = model(images.to(device))
            predictions = torch.sigmoid(seg_out) >= args.threshold
            counts.append(
                confusion_counts(
                    predictions.cpu().numpy(), masks.numpy() >= 0.5
                )
            )

    metrics = metrics_from_counts(merge_counts(counts))
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
