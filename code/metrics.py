"""Binary segmentation metrics used by Seg-Road evaluation."""

import numpy as np


def confusion_counts(prediction, target):
    """Return TP, FP, FN, TN counts for binary arrays."""
    prediction = np.asarray(prediction).astype(bool)
    target = np.asarray(target).astype(bool)
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match target {target.shape}."
        )

    true_positive = np.logical_and(prediction, target).sum()
    false_positive = np.logical_and(prediction, ~target).sum()
    false_negative = np.logical_and(~prediction, target).sum()
    true_negative = np.logical_and(~prediction, ~target).sum()
    return true_positive, false_positive, false_negative, true_negative


def binary_metrics(prediction, target, epsilon=1e-7):
    """Compute IoU, F1, precision, recall, and accuracy."""
    true_positive, false_positive, false_negative, true_negative = confusion_counts(
        prediction, target
    )
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    iou = true_positive / (
        true_positive + false_positive + false_negative + epsilon
    )
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    accuracy = (true_positive + true_negative) / (
        true_positive + false_positive + false_negative + true_negative + epsilon
    )
    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
    }


def merge_counts(counts):
    """Add confusion counts collected from multiple batches."""
    return tuple(sum(batch[index] for batch in counts) for index in range(4))


def metrics_from_counts(counts, epsilon=1e-7):
    """Compute metrics after accumulating counts over a complete dataset."""
    true_positive, false_positive, false_negative, true_negative = counts
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    iou = true_positive / (
        true_positive + false_positive + false_negative + epsilon
    )
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    accuracy = (true_positive + true_negative) / (
        true_positive + false_positive + false_negative + true_negative + epsilon
    )
    return {
        "iou": float(iou),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "accuracy": float(accuracy),
    }
