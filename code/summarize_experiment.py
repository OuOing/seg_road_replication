"""Summarize a Seg-Road experiment checkpoint as report-ready Markdown."""

import argparse
import csv
from pathlib import Path

import torch

from model import SegRoad
from model_stats import count_parameters, format_millions
from threshold_sweep import best_threshold


METRIC_FIELDS = (
    "iou",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "predicted_positive_ratio",
    "target_positive_ratio",
)


def format_float(value):
    return f"{float(value):.4f}"


def load_threshold_csv(path):
    results = {}
    with Path(path).open(newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            threshold = float(row["threshold"])
            results[threshold] = {
                key: float(value)
                for key, value in row.items()
                if key != "threshold"
            }
    if not results:
        raise ValueError(f"No threshold rows found in {path}.")
    return results


def checkpoint_row(checkpoint_path, model_size):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metrics = checkpoint["metrics"]
    model = SegRoad(model_size=model_size)
    counts = count_parameters(model)
    return {
        "name": Path(checkpoint_path).parent.name,
        "epoch": checkpoint["epoch"],
        "threshold": 0.5,
        "parameters": counts["total"],
        "parameters_m": format_millions(counts["total"]),
        **{field: metrics[field] for field in METRIC_FIELDS},
    }


def threshold_row(threshold_csv, model_size, checkpoint_path, metric_name="f1"):
    results = load_threshold_csv(threshold_csv)
    threshold, metrics = best_threshold(results, metric_name)
    row = checkpoint_row(checkpoint_path, model_size)
    row.update(
        {
            "name": f"{row['name']} threshold-selected",
            "threshold": threshold,
            **{field: metrics[field] for field in METRIC_FIELDS},
        }
    )
    return row


def markdown_table(rows):
    headers = (
        "Experiment",
        "Epoch",
        "Threshold",
        "IoU",
        "F1",
        "Precision",
        "Recall",
        "pred+",
        "target+",
        "Params",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    str(row["epoch"]),
                    format_float(row["threshold"]),
                    format_float(row["iou"]),
                    format_float(row["f1"]),
                    format_float(row["precision"]),
                    format_float(row["recall"]),
                    format_float(row["predicted_positive_ratio"]),
                    format_float(row["target_positive_ratio"]),
                    row["parameters_m"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-size", choices=("s", "m", "l"), default="s")
    parser.add_argument(
        "--threshold-csv",
        help="Optional threshold sweep CSV used to add a threshold-selected row.",
    )
    parser.add_argument(
        "--select-metric",
        choices=("iou", "f1"),
        default="f1",
        help="Metric used to choose the threshold-selected row.",
    )
    parser.add_argument("--output-md", help="Write the Markdown table to a file.")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = [checkpoint_row(args.checkpoint, args.model_size)]
    if args.threshold_csv:
        rows.append(
            threshold_row(
                args.threshold_csv,
                args.model_size,
                args.checkpoint,
                metric_name=args.select_metric,
            )
        )
    table = markdown_table(rows)
    print(table)
    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(table + "\n")
        print(f"Saved experiment summary to {args.output_md}")


if __name__ == "__main__":
    main()
