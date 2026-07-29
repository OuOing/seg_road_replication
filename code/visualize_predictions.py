"""Save validation image, mask, prediction, and error-overlay comparisons."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from dataset import collect_image_mask_pairs, select_pairs_from_list
from device import DEVICE_CHOICES, select_device
from model import SegRoad


TP_COLOR = np.array([46, 204, 113], dtype=np.float32)
FP_COLOR = np.array([231, 76, 60], dtype=np.float32)
FN_COLOR = np.array([52, 152, 219], dtype=np.float32)


def choose_pairs(pairs, sample_count, seed):
    """Choose a reproducible subset without changing the original pair order."""
    pairs = list(pairs)
    if sample_count <= 0:
        raise ValueError("--samples must be positive.")
    if sample_count >= len(pairs):
        return pairs
    indices = sorted(random.Random(seed).sample(range(len(pairs)), sample_count))
    return [pairs[index] for index in indices]


def error_overlay(image, target, prediction, alpha=0.65):
    """Overlay true positives in green, false positives in red, and misses in blue."""
    image = np.asarray(image, dtype=np.uint8)
    target = np.asarray(target, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    if image.shape[:2] != target.shape or target.shape != prediction.shape:
        raise ValueError("Image, target, and prediction spatial shapes must match.")

    overlay = image.astype(np.float32).copy()
    classes = (
        (target & prediction, TP_COLOR),
        (~target & prediction, FP_COLOR),
        (target & ~prediction, FN_COLOR),
    )
    for selection, color in classes:
        overlay[selection] = (1.0 - alpha) * overlay[selection] + alpha * color
    return np.clip(overlay, 0, 255).astype(np.uint8)


def labeled_panel(image, label):
    """Add a stable-height label bar above a PIL image."""
    label_height = 28
    panel = Image.new("RGB", (image.width, image.height + label_height), "white")
    panel.paste(image.convert("RGB"), (0, label_height))
    ImageDraw.Draw(panel).text((8, 7), label, fill="black")
    return panel


def comparison_row(image, target, prediction):
    target_image = Image.fromarray(target.astype(np.uint8) * 255)
    prediction_image = Image.fromarray(prediction.astype(np.uint8) * 255)
    overlay_image = Image.fromarray(error_overlay(image, target, prediction))
    panels = [
        labeled_panel(image, "Image"),
        labeled_panel(target_image, "Ground truth"),
        labeled_panel(prediction_image, "Prediction"),
        labeled_panel(overlay_image, "TP green | FP red | FN blue"),
    ]
    row = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height))
    x = 0
    for panel in panels:
        row.paste(panel, (x, 0))
        x += panel.width
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--split-list", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-size", choices=("s", "m", "l"), default="s")
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
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
    pairs = select_pairs_from_list(pairs, args.split_list)
    pairs = choose_pairs(pairs, args.samples, args.seed)
    image_size = (args.image_width, args.image_height)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    with torch.no_grad():
        for image_path, mask_path in pairs:
            image = Image.open(image_path).convert("RGB").resize(
                image_size, Image.BILINEAR
            )
            target = np.asarray(
                Image.open(mask_path).convert("L").resize(image_size, Image.NEAREST)
            ) > 0
            image_array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(
                np.transpose(image_array, (2, 0, 1)).copy()
            ).unsqueeze(0)
            seg_out, _ = model(tensor.to(device))
            probability = torch.sigmoid(seg_out)[0, 0].cpu().numpy()
            prediction = probability >= args.threshold
            row = comparison_row(image, target, prediction)
            row.save(output_dir / f"{image_path.stem}_comparison.png")
            rows.append(row)

    sheet = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet_path = output_dir / "comparison_sheet.png"
    sheet.save(sheet_path)
    print(f"Saved {len(rows)} comparisons to {output_dir}")
    print(f"Contact sheet: {sheet_path}")


if __name__ == "__main__":
    main()
