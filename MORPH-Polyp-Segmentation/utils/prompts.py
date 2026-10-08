from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import cv2
import torch


@dataclass(frozen=True)
class Click:
    y: int
    x: int
    positive: bool


def _sample(mask: np.ndarray, positive: bool, rng: np.random.Generator) -> Click | None:
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return None
    y, x = coordinates[int(rng.integers(len(coordinates)))]
    return Click(int(y), int(x), positive)


def sample_first_click(gt: np.ndarray, rng: np.random.Generator) -> Click:
    click = _sample(gt.astype(bool), True, rng)
    if click is None:
        raise ValueError("Cannot sample a positive click from an empty ground-truth mask")
    return click


def _largest_component(mask: np.ndarray):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return None, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas)) + 1
    return labels == label, int(areas[label - 1])


def sample_second_click(gt: np.ndarray, prediction: np.ndarray, rng: np.random.Generator):
    gt, prediction = gt.astype(bool), prediction.astype(bool)
    largest_fn, fn_area = _largest_component(gt & ~prediction)
    largest_fp, fp_area = _largest_component(prediction & ~gt)
    if fn_area == fp_area == 0:
        return None
    # Equal area deliberately chooses FN.
    if fn_area >= fp_area:
        return _sample(largest_fn, True, rng)
    return _sample(largest_fp, False, rng)


def render_dense_click_map(
    clicks: list[list[Click]], image_size: int = 1024, grid_size: int = 128, radius: int = 5,
    device=None,
) -> torch.Tensor:
    output = torch.zeros(len(clicks), 2, grid_size, grid_size, device=device)
    yy, xx = torch.meshgrid(
        torch.arange(grid_size, device=device), torch.arange(grid_size, device=device), indexing="ij"
    )
    scale = grid_size / image_size
    for batch_index, sample_clicks in enumerate(clicks):
        for click in sample_clicks:
            cy = min(grid_size - 1, max(0, int(click.y * scale)))
            cx = min(grid_size - 1, max(0, int(click.x * scale)))
            disk = (yy - cy).square() + (xx - cx).square() <= radius * radius
            output[batch_index, 0 if click.positive else 1, disk] = 1
    return output
