"""Evaluate one checkpoint at multiple segmentation thresholds in one pass."""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import RoadDataset, collect_image_mask_pairs, select_pairs_from_list
from device import DEVICE_CHOICES, select_device
from metrics import confusion_counts, merge_counts, metrics_from_counts
from model import SegRoad


CSV_FIELDS = (
    "threshold",
    "iou",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "predicted_positive_ratio",
    "target_positive_ratio",
)


def parse_thresholds(value):
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds:
        raise argparse.ArgumentTypeError("At least one threshold is required.")
    for threshold in thresholds:
        if threshold < 0.0 or threshold > 1.0:
            raise argparse.ArgumentTypeError(
                f"Threshold must be in [0, 1], got {threshold}."
            )
    return thresholds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument(
        "--split-list",
        help="Text file containing one validation sample stem per line.",
    )
    parser.add_argument("--model-size", choices=("s", "m", "l"), default="s")
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--thresholds",
        type=parse_thresholds,
        default=parse_thresholds("0.30,0.35,0.40,0.45,0.50,0.55,0.60"),
        help="Comma-separated probability thresholds.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Evaluate only the first N selected samples for a quick probe.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=0,
        help="Print progress every N batches.",
    )
    parser.add_argument(
        "--output-csv",
        help="Write threshold metrics to a CSV file.",
    )
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    return parser.parse_args()


def best_threshold(results, metric_name):
    if metric_name not in CSV_FIELDS:
        raise ValueError(f"Unknown metric: {metric_name}")
    return max(sorted(results.items()), key=lambda item: item[1][metric_name])


def rows_from_results(results):
    rows = []
    for threshold in sorted(results):
        metrics = results[threshold]
        row = {"threshold": threshold}
        row.update(metrics)
        rows.append(row)
    return rows


def write_results_csv(results, output_csv):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows_from_results(results):
            writer.writerow({field: row[field] for field in CSV_FIELDS})


def print_results(results):
    print(",".join(CSV_FIELDS))
    for row in rows_from_results(results):
        print(
            f"{row['threshold']:.4f},"
            f"{row['iou']:.6f},"
            f"{row['f1']:.6f},"
            f"{row['precision']:.6f},"
            f"{row['recall']:.6f},"
            f"{row['accuracy']:.6f},"
            f"{row['predicted_positive_ratio']:.6f},"
            f"{row['target_positive_ratio']:.6f}"
        )


def evaluate_thresholds(model, loader, thresholds, device, log_interval=0):
    counts_by_threshold = {threshold: [] for threshold in thresholds}
    started_at = time.perf_counter()
    with torch.no_grad():
        for batch_index, (images, masks, _) in enumerate(loader, 1):
            seg_out, _ = model(images.to(device))
            probabilities = torch.sigmoid(seg_out).cpu().numpy()
            targets = masks.numpy() >= 0.5
            for threshold in thresholds:
                counts_by_threshold[threshold].append(
                    confusion_counts(probabilities >= threshold, targets)
                )
            if log_interval and batch_index % log_interval == 0:
                elapsed = time.perf_counter() - started_at
                print(
                    f"batch {batch_index}/{len(loader)} | {elapsed:.1f}s",
                    flush=True,
                )

    return {
        threshold: metrics_from_counts(merge_counts(counts))
        for threshold, counts in counts_by_threshold.items()
    }


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
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive.")
        pairs = pairs[: args.max_samples]

    dataset = RoadDataset(
        pairs, image_size=(args.image_height, args.image_width), augment=False
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    results = evaluate_thresholds(
        model, loader, args.thresholds, device, log_interval=args.log_interval
    )
    print_results(results)

    best_iou_threshold, best_iou_metrics = best_threshold(results, "iou")
    best_f1_threshold, best_f1_metrics = best_threshold(results, "f1")
    print(
        f"best_iou_threshold={best_iou_threshold:.4f} "
        f"iou={best_iou_metrics['iou']:.6f}"
    )
    print(
        f"best_f1_threshold={best_f1_threshold:.4f} "
        f"f1={best_f1_metrics['f1']:.6f}"
    )
    if args.output_csv:
        write_results_csv(results, args.output_csv)
        print(f"Saved threshold sweep CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
