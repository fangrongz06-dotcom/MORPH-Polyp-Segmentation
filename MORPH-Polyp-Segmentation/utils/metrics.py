from __future__ import annotations

import numpy as np
import cv2
from collections import defaultdict
import torch
from torch.nn import functional as F


def _boundary(mask, width=1):
    kernel = np.ones((2 * width + 1, 2 * width + 1), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask & ~eroded.astype(bool)


def _surface_distances(a, b):
    a_surface, b_surface = _boundary(a), _boundary(b)
    if not a.any() and not b.any():
        return np.array([0.0]), np.array([0.0])
    if not a_surface.any() or not b_surface.any():
        return np.array([np.inf]), np.array([np.inf])
    distance_to_b = cv2.distanceTransform((~b_surface).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    distance_to_a = cv2.distanceTransform((~a_surface).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return distance_to_b[a_surface], distance_to_a[b_surface]


def compute_metrics(prediction, target, eps=1e-7, boundary_width=1):
    prediction, target = np.asarray(prediction, bool), np.asarray(target, bool)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Metrics require equal-sized 2D binary masks")
    if boundary_width < 1:
        raise ValueError("boundary_width must be positive")
    tp = np.logical_and(prediction, target).sum()
    fp = np.logical_and(prediction, ~target).sum()
    fn = np.logical_and(~prediction, target).sum()
    tn = np.logical_and(~prediction, ~target).sum()
    forward_distances, backward_distances = _surface_distances(prediction, target)
    distances = np.concatenate([forward_distances, backward_distances])
    hd95 = float("inf") if np.isinf(distances).any() else float(np.percentile(distances, 95))
    assd = float((forward_distances.mean() + backward_distances.mean()) / 2)
    p_boundary, t_boundary = _boundary(prediction, boundary_width), _boundary(target, boundary_width)
    boundary_union = np.logical_or(p_boundary, t_boundary).sum()
    return {
        "dice": (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou": (tp + eps) / (tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "sensitivity": (tp + eps) / (tp + fn + eps),
        "specificity": (tn + eps) / (tn + fp + eps),
        "hd95": hd95,
        "assd": assd,
        "boundary_iou": (np.logical_and(p_boundary, t_boundary).sum() + eps) / (boundary_union + eps),
    }


"""Keep dataset-macro and image-weighted averages distinct."""


METRICS = ("dice", "iou", "precision", "sensitivity", "specificity", "hd95", "assd", "boundary_iou")


def aggregate_rows(rows, metrics=METRICS):
    if not rows: raise ValueError("Cannot aggregate an empty evaluation")
    grouped = defaultdict(list)
    for row in rows: grouped[row["dataset"]].append(row)
    datasets = []
    for name, samples in sorted(grouped.items()):
        datasets.append({"dataset": name, "count": len(samples), **{
            metric: float(np.mean([float(sample[metric]) for sample in samples])) for metric in metrics
        }})
    overall = {
        "image_weighted": {metric: float(np.mean([float(row[metric]) for row in rows])) for metric in metrics},
        "dataset_macro": {metric: float(np.mean([dataset[metric] for dataset in datasets])) for metric in metrics},
        "image_count": len(rows), "dataset_count": len(datasets),
    }
    return datasets, overall


"""Detached, per-image metrics for augmented training batches."""



class TrainingMetricAccumulator:
    def __init__(self, threshold=0.5, epsilon=1e-7, dice_smooth=1e-6):
        self.threshold, self.epsilon, self.dice_smooth = threshold, epsilon, dice_smooth
        self.total = None
        self.count = 0
        self.names = ("loss", "bce_loss", "dice_loss", "train_soft_dice", "train_dice", "train_iou",
                      "train_precision", "train_recall", "train_specificity", "train_accuracy")

    @torch.no_grad()
    def update(self, logits, target, loss):
        logits, target = logits.detach().float(), target.detach().float()
        probability = torch.sigmoid(logits)
        dimensions = tuple(range(1, logits.ndim))
        prediction, truth = probability.ge(self.threshold), target.ge(0.5)
        tp = (prediction & truth).sum(dimensions, dtype=torch.float64)
        fp = (prediction & ~truth).sum(dimensions, dtype=torch.float64)
        fn = (~prediction & truth).sum(dimensions, dtype=torch.float64)
        tn = (~prediction & ~truth).sum(dimensions, dtype=torch.float64)
        e = self.epsilon
        soft_dice = (2 * (probability * target).sum(dimensions) + self.dice_smooth) / (
            probability.sum(dimensions) + target.sum(dimensions) + self.dice_smooth)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dimensions)
        size = logits.shape[0]
        values = torch.stack((loss.detach().to(torch.float64) * size, bce.sum(), (1 - soft_dice).sum(),
            soft_dice.sum(), ((2 * tp + e) / (2 * tp + fp + fn + e)).sum(),
            ((tp + e) / (tp + fp + fn + e)).sum(), ((tp + e) / (tp + fp + e)).sum(),
            ((tp + e) / (tp + fn + e)).sum(), ((tn + e) / (tn + fp + e)).sum(),
            ((tp + tn) / (tp + fp + fn + tn)).sum()))
        self.total = values if self.total is None else self.total + values
        self.count += size

    def compute(self):
        if not self.count:
            raise ValueError("Cannot compute metrics for an empty training epoch")
        return dict(zip(self.names, (self.total / self.count).cpu().tolist()))


def compute_boundary_metrics(prediction, target):
    result = compute_metrics(prediction, target)
    return {key: result[key] for key in ("hd95", "assd", "boundary_iou")}
