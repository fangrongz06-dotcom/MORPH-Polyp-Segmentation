from __future__ import annotations

import torch
from torch import nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        dims = tuple(range(1, probability.ndim))
        intersection = (probability * target).sum(dims)
        denominator = probability.sum(dims) + target.sum(dims)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()


class MORPHSegmentationLoss(nn.Module):
    def __init__(self, bce_weight=0.3, dice_weight=1.2, smooth=1e-6) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(smooth)

    def forward(self, logits, target):
        bce = self.bce(logits, target)
        dice = self.dice(logits, target)
        return self.bce_weight * bce + self.dice_weight * dice
