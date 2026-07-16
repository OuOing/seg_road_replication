"""Run Seg-Road inference on one image and save a binary road mask."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import SegRoad


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-size", choices=("s", "m", "l"), default="s")
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def load_image(path, image_size):
    image = Image.open(path).convert("RGB")
    original_size = image.size
    resized = image.resize((image_size[1], image_size[0]), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).unsqueeze(0), original_size


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = SegRoad(model_size=args.model_size).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image, original_size = load_image(
        args.image, (args.image_height, args.image_width)
    )
    with torch.no_grad():
        seg_out, _ = model(image.to(device))
        probability = torch.sigmoid(seg_out)[0, 0].cpu().numpy()

    prediction = (probability >= args.threshold).astype(np.uint8) * 255
    output = Image.fromarray(prediction).resize(original_size, Image.NEAREST)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    print(f"Saved road mask to {args.output}")


if __name__ == "__main__":
    main()
