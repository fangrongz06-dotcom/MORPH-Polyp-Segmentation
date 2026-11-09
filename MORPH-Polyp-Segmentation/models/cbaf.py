from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class BranchProcessing(nn.Sequential):
    def __init__(self, channels=512, groups=8):
        super().__init__(
            nn.Conv2d(channels, channels, 3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
        )


class CBAF(nn.Module):
    def __init__(self, channels=512, groups=8, reduction=16, mode="cbaf") -> None:
        super().__init__()
        self.mode = mode
        self.vit_align = nn.Conv2d(768, channels, 1)
        uses_branches = mode in {"cbaf", "cbaf_no_base_vit"}
        self.cnn_branch = BranchProcessing(channels, groups) if uses_branches else nn.Identity()
        self.vit_branch = BranchProcessing(channels, groups) if uses_branches else nn.Identity()
        hidden = channels // reduction
        self.recalibration = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        ) if mode in {"cbaf", "cbaf_no_base_vit"} else nn.Identity()
        self.concat = nn.Conv2d(channels * 2, channels, 1) if mode in {"concat", "cbaf_no_base_vit"} else None
        self.final = nn.Conv2d(channels * 3, channels, 1) if mode == "cbaf" else None

    def forward(self, cnn: torch.Tensor, vit: torch.Tensor) -> torch.Tensor:
        vit = F.interpolate(vit, (32, 32), mode="bilinear", align_corners=False)
        vit = self.vit_align(vit)
        if self.mode == "vit_only" or self.mode == "disabled":
            return vit
        if self.mode == "add":
            return cnn + vit
        if self.mode == "concat":
            return self.concat(torch.cat([cnn, vit], dim=1))
        if self.mode not in {"cbaf", "cbaf_no_base_vit"}:
            raise ValueError(f"Unknown CBAF mode: {self.mode}")
        fc = self.cnn_branch(cnn)
        fv = self.vit_branch(vit)
        mask = self.recalibration(F.relu(fc + fv, inplace=False))
        if self.mode == "cbaf_no_base_vit":
            return self.concat(torch.cat([fv * mask, fc], dim=1))
        return self.final(torch.cat([fv, fv * mask, fc], dim=1))
