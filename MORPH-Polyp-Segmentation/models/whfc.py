from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FixedHaarDWT(nn.Module):
    """Channel-wise orthonormal Haar analysis with signed coefficients."""

    def __init__(self) -> None:
        super().__init__()
        scale = 0.5
        filters = torch.tensor(
            [
                [[1, 1], [1, 1]],   # approximation
                [[-1, -1], [1, 1]], # horizontal detail
                [[-1, 1], [-1, 1]], # vertical detail
                [[1, -1], [-1, 1]], # diagonal detail
            ],
            dtype=torch.float32,
        ).mul_(scale)
        self.register_buffer("filters", filters[:, None], persistent=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        channels = x.shape[1]
        weight = self.filters.repeat(channels, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, groups=channels)
        y = y.view(x.shape[0], channels, 4, y.shape[-2], y.shape[-1])
        return tuple(y[:, :, index] for index in range(4))


class FixedWaveletDWT(nn.Module):
    """Fixed channel-wise analysis for an explicitly named PyWavelets family/order."""

    def __init__(self, wavelet: str) -> None:
        super().__init__()
        try:
            import pywt
            w = pywt.Wavelet(wavelet)
        except Exception as exc:
            raise ValueError(f"Unsupported or unavailable exact wavelet name: {wavelet}") from exc
        low = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)
        high = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
        filters = torch.stack([
            torch.outer(low, low), torch.outer(high, low),
            torch.outer(low, high), torch.outer(high, high),
        ])[:, None]
        self.register_buffer("filters", filters)
        self.padding = (len(w.dec_lo) - 2) // 2

    def forward(self, x):
        channels = x.shape[1]
        y = F.conv2d(x, self.filters.repeat(channels, 1, 1, 1), stride=2,
                     padding=self.padding, groups=channels)
        y = y.view(x.shape[0], channels, 4, y.shape[-2], y.shape[-1])
        return tuple(y[:, :, index] for index in range(4))


class WHFC(nn.Module):
    def __init__(self, stage_channels=(64, 128, 256, 512), levels: int = 3, wavelet="haar") -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive")
        self.levels = levels
        self.wavelet = wavelet
        self.dwt = FixedHaarDWT() if wavelet == "haar" else FixedWaveletDWT(wavelet)
        input_channels = 3 * 3 * levels
        self.projections = nn.ModuleList(
            nn.Conv2d(input_channels, channels, kernel_size=1) for channels in stage_channels
        )

    def forward(self, image_01: torch.Tensor, size=(64, 64)):
        approximation = image_01
        groups = []
        details = []
        for _ in range(self.levels):
            approximation, dh, dv, dd = self.dwt(approximation)
            group = torch.cat([dd, dh, dv], dim=1)
            details.append((dh, dv, dd))
            groups.append(F.interpolate(group, size=size, mode="bilinear", align_corners=False))
        basis = torch.cat(groups, dim=1)
        features = tuple(projection(basis) for projection in self.projections)
        return features, {"basis": basis, "details": details}
