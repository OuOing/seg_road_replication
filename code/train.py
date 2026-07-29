"""Minimal training entry point for the Seg-Road v1 reproduction."""

import argparse
import random
import time
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


def set_optimizer_learning_rate(optimizer, learning_rate):
    """Set all optimizer parameter groups to the same learning rate."""
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def run_epoch(
    model,
    loader,
    loss_fn,
    optimizer,
    device,
    training,
    threshold=0.5,
    log_interval=0,
    phase="",
):
    model.train(training)
    started_at = time.perf_counter()
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
        predictions = probabilities >= threshold
        counts.append(
            confusion_counts(
                predictions.detach().cpu().numpy(),
                masks.detach().cpu().numpy() >= 0.5,
            )
        )
        total_loss += total.item()
        batch_count += 1
        if log_interval and batch_count % log_interval == 0:
            elapsed = time.perf_counter() - started_at
            print(
                f"  {phase} batch {batch_count}/{len(loader)} | "
                f"loss {total_loss / batch_count:.4f} | {elapsed:.1f}s",
                flush=True,
            )

    metrics = metrics_from_counts(merge_counts(counts))
    metrics["loss"] = total_loss / max(batch_count, 1)
    metrics["elapsed_seconds"] = time.perf_counter() - started_at
    return metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-dir", default="runs/segroad")
    parser.add_argument("--resume", help="Checkpoint path used to resume training.")
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
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used for train/validation metrics.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--resume-learning-rate",
        type=float,
        help=(
            "Override the learning rate stored in a resumed optimizer checkpoint."
        ),
    )
    parser.add_argument(
        "--reset-best-on-resume",
        action="store_true",
        help="Start best checkpoint tracking fresh after resuming.",
    )
    parser.add_argument("--seg-pos-weight", type=float, default=1.0)
    parser.add_argument("--pcs-pos-weight", type=float, default=1.0)
    parser.add_argument(
        "--pcs-alpha",
        type=float,
        default=0.2,
        help="Weight applied to the PCS connectivity loss; use 0 for ablation.",
    )
    parser.add_argument("--dice-weight", type=float, default=0.0)
    parser.add_argument("--pcs-radius", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    return parser.parse_args(argv)


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
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    model = SegRoad(model_size=args.model_size).to(device)
    seg_pos_weight = torch.tensor([args.seg_pos_weight], device=device)
    pcs_pos_weight = torch.tensor([args.pcs_pos_weight], device=device)
    loss_fn = SegRoadLoss(
        alpha=args.pcs_alpha,
        pos_weight_seg=seg_pos_weight,
        pos_weight_con=pcs_pos_weight,
        dice_weight=args.dice_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    start_epoch = 1

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if args.resume_learning_rate is not None:
            set_optimizer_learning_rate(optimizer, args.resume_learning_rate)
        start_epoch = checkpoint["epoch"] + 1
        if not args.reset_best_on_resume:
            best_iou = checkpoint.get("metrics", {}).get("iou", -1.0)
        print(f"Resuming from epoch {checkpoint['epoch']}: {args.resume}")
        if args.resume_learning_rate is not None:
            print(f"Overriding resumed learning rate: {args.resume_learning_rate}")
        if args.reset_best_on_resume:
            print("Resetting best checkpoint tracking for this resumed run.")

    print(f"Using device: {device}")
    print(f"Training samples: {len(train_dataset)}; validation samples: {len(val_dataset)}")
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            training=True,
            threshold=args.eval_threshold,
            log_interval=args.log_interval,
            phase="train",
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                loss_fn,
                optimizer,
                device,
                training=False,
                threshold=args.eval_threshold,
                log_interval=args.log_interval,
                phase="val",
            )
        print(
            f"Epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f}, IoU {train_metrics['iou']:.4f} | "
            f"val loss {val_metrics['loss']:.4f}, IoU {val_metrics['iou']:.4f}, "
            f"F1 {val_metrics['f1']:.4f}, "
            f"P {val_metrics['precision']:.4f}, R {val_metrics['recall']:.4f}, "
            f"pred+ {val_metrics['predicted_positive_ratio']:.4f}, "
            f"thr {args.eval_threshold:.2f} | "
            f"time {train_metrics['elapsed_seconds']:.0f}s/"
            f"{val_metrics['elapsed_seconds']:.0f}s"
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
