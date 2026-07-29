"""Report parameter count and optional inference latency for Seg-Road."""

import argparse
import time

import torch

from device import DEVICE_CHOICES, select_device
from model import SegRoad


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}


def format_millions(value):
    return f"{value / 1_000_000:.3f}M"


def benchmark_forward(model, image_size, batch_size, device, warmup=3, repeats=10):
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive.")

    model.eval()
    sample = torch.randn(
        batch_size, 3, image_size[0], image_size[1], device=device
    )
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started_at = time.perf_counter()
        for _ in range(repeats):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - started_at
    return elapsed / repeats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=("s", "m", "l"), default="s")
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a small forward-pass latency benchmark.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device)
    model = SegRoad(model_size=args.model_size).to(device)
    counts = count_parameters(model)

    print(f"model_size: {args.model_size}")
    print(f"device: {device}")
    print(f"total_parameters: {counts['total']} ({format_millions(counts['total'])})")
    print(
        f"trainable_parameters: {counts['trainable']} "
        f"({format_millions(counts['trainable'])})"
    )
    if args.benchmark:
        seconds = benchmark_forward(
            model,
            image_size=(args.image_height, args.image_width),
            batch_size=args.batch_size,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        print(f"forward_latency_seconds: {seconds:.6f}")
        print(f"forward_latency_ms: {seconds * 1000:.3f}")


if __name__ == "__main__":
    main()
