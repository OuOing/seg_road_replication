"""Minimal training entry point for the Seg-Road v1 reproduction."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RoadDataset, collect_image_mask_pairs, select_pairs_from_list
from device import DEVICE_CHOICES, select_device
from loss import SegRoadLoss
from metrics import confusion_counts, merge_counts, metrics_from_counts
from model import SegRoad


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_pairs(pairs, val_ratio, seed):
    pairs = list(pairs)
    generator = np.random.default_rng(seed)
    generator.shuffle(pairs)
    val_count = max(1, int(len(pairs) * val_ratio))
    if len(pairs) <= val_count:
        raise ValueError("The dataset needs more samples than the validation split.")
    return pairs[val_count:], pairs[:val_count]


def run_epoch(model, loader, loss_fn, optimizer, device, training):
    model.train(training)
    total_loss = 0.0
    batch_count = 0
    counts = []

    for images, masks, pcs_targets in loader:
        images = images.to(device)
        masks = masks.to(device)
        pcs_targets = pcs_targets.to(device)

        with torch.set_grad_enabled(training):
            seg_out, pcs_out = model(images)
            total, _, _ = loss_fn(seg_out, masks, pcs_out, pcs_targets)
            if training:
                optimizer.zero_grad()
                total.backward()
                optimizer.step()

        probabilities = torch.sigmoid(seg_out)
        predictions = probabilities >= 0.5
        counts.append(
            confusion_counts(
                predictions.detach().cpu().numpy(),
                masks.detach().cpu().numpy() >= 0.5,
            )
        )
        total_loss += total.item()
        batch_count += 1

    metrics = metrics_from_counts(merge_counts(counts))
    metrics["loss"] = total_loss / max(batch_count, 1)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-dir", default="runs/segroad")
    parser.add_argument("--model-size", choices=("s", "m", "l"), default="s")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument(
        "--train-list",
        help="Text file containing one training sample stem per line.",
    )
    parser.add_argument(
        "--val-list",
        help="Text file containing one validation sample stem per line.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--pcs-radius", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    pairs = collect_image_mask_pairs(args.image_dir, args.mask_dir)
    if bool(args.train_list) != bool(args.val_list):
        raise ValueError("--train-list and --val-list must be provided together.")
    if args.train_list:
        train_pairs = select_pairs_from_list(pairs, args.train_list)
        val_pairs = select_pairs_from_list(pairs, args.val_list)
        overlap = {pair[0].stem for pair in train_pairs} & {
            pair[0].stem for pair in val_pairs
        }
        if overlap:
            preview = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"Training and validation splits overlap: {preview}")
    else:
        print(
            "Warning: using a random file-level validation split. "
            "Use --train-list and --val-list for formal experiments."
        )
        train_pairs, val_pairs = split_pairs(pairs, args.val_ratio, args.seed)
    image_size = (args.image_height, args.image_width)

    train_dataset = RoadDataset(
        train_pairs, image_size=image_size, augment=True, pcs_radius=args.pcs_radius
    )
    val_dataset = RoadDataset(
        val_pairs, image_size=image_size, augment=False, pcs_radius=args.pcs_radius
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SegRoad(model_size=args.model_size).to(device)
    loss_fn = SegRoadLoss(alpha=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0

    print(f"Using device: {device}")
    print(f"Training samples: {len(train_dataset)}; validation samples: {len(val_dataset)}")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, loss_fn, optimizer, device, training=True
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, loss_fn, optimizer, device, training=False
            )
        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f}, IoU {train_metrics['iou']:.4f} | "
            f"val loss {val_metrics['loss']:.4f}, IoU {val_metrics['iou']:.4f}, "
            f"F1 {val_metrics['f1']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
