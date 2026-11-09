from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SpatialEstimator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return torch.sigmoid(self.conv(pooled))


class ChannelEstimator(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.projector = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.GELU(), nn.Conv2d(hidden, channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.projector(F.adaptive_avg_pool2d(x, 1))
        maximum = self.projector(F.adaptive_max_pool2d(x, 1))
        return torch.sigmoid(avg + maximum)


class MEAA(nn.Module):
    """Three-source morphology-enhanced attention adapter."""

    def __init__(
        self,
        channels: int,
        aggregation: str = "branch_softmax",
        enabled_sources=("cnn", "whfc", "vit"),
        prompt_dim: int = 64,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.aggregation = aggregation
        self.enabled_sources = tuple(enabled_sources)
        if len(set(self.enabled_sources)) != len(self.enabled_sources) or any(
            source not in {"cnn", "whfc", "vit"} for source in self.enabled_sources
        ):
            raise ValueError("MEAA sources must be unique members of cnn/whfc/vit")
        self.vit_projection = nn.Conv2d(768, channels, 1) if "vit" in self.enabled_sources else nn.Identity()
        # One shared estimator pair is reused for every source inside this block.
        attention_weighted = aggregation in {"branch_softmax", "independent_sigmoid"}
        self.spatial_estimator = SpatialEstimator() if attention_weighted else nn.Identity()
        self.channel_estimator = ChannelEstimator(channels) if attention_weighted else nn.Identity()
        self.concat_projection = (
            nn.Conv2d(channels * len(self.enabled_sources), channels, 1)
            if self.enabled_sources and aggregation == "concat_proj" else nn.Identity()
        )
        self.MLP_tune = nn.Conv2d(channels, prompt_dim, 1)
        self.activation = nn.GELU()
        self.MLP_up = nn.Conv2d(prompt_dim, 768, 1)

    def forward(self, cnn: torch.Tensor, whfc: torch.Tensor, vit: torch.Tensor):
        sources = {}
        if "cnn" in self.enabled_sources:
            sources["cnn"] = F.interpolate(cnn, (64, 64), mode="bilinear", align_corners=False)
        if "whfc" in self.enabled_sources:
            if whfc is None: raise ValueError("WHFC source enabled but no WHFC feature was supplied")
            sources["whfc"] = whfc
        if "vit" in self.enabled_sources:
            sources["vit"] = self.vit_projection(vit)
        active = [sources[name] for name in self.enabled_sources]
        if not active:
            return torch.zeros_like(vit), None
        for source in active:
            if source.shape[1:] != (self.channels, 64, 64):
                raise ValueError(f"MEAA source has unexpected shape: {source.shape}")
        alpha = None
        if self.aggregation == "equal_sum":
            fused = torch.stack(active, dim=1).mean(dim=1)
        elif self.aggregation == "concat_proj":
            fused = self.concat_projection(torch.cat(active, dim=1))
        else:
            responses = torch.stack(
                [self.spatial_estimator(source) * self.channel_estimator(source) for source in active],
                dim=1,
            )
            if self.aggregation == "branch_softmax":
                alpha = torch.softmax(responses, dim=1)
            elif self.aggregation == "independent_sigmoid":
                alpha = torch.sigmoid(responses)
            else:
                raise ValueError(f"Unknown MEAA aggregation: {self.aggregation}")
            fused = (alpha * torch.stack(active, dim=1)).sum(dim=1)
        prompt = self.MLP_up(self.activation(self.MLP_tune(fused)))
        return prompt, alpha
