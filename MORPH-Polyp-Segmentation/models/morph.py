from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from pathlib import Path
from typing import Callable
import torch.nn.functional as F
from .cbaf import CBAF
from .meaa import MEAA
from .whfc import WHFC


class FrozenResNet18Encoder(nn.Module):
    """Frozen ImageNet ResNet-18 returning layer1..layer4 feature maps."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        try:
            net = resnet18(weights=weights)
        except Exception as exc:
            raise RuntimeError("Failed to load ResNet-18 IMAGENET1K_V1 weights") from exc
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        # Frozen BatchNorm running statistics must never change.
        super().train(False)
        return self

    def forward(self, image_01: torch.Tensor) -> tuple[torch.Tensor, ...]:
        with torch.no_grad():
            x = (image_01 - self.mean) / self.std
            x = self.stem(x)
            f1 = self.layer1(x)
            f2 = self.layer2(f1)
            f3 = self.layer3(f2)
            f4 = self.layer4(f3)
        return f1, f2, f3, f4


class FrozenSAMViTBEncoder(nn.Module):
    """Official SAM ViT-B with MEAA prompts injected after blocks 2/5/8/11.

    Frozen parameters remain in the autograd graph so gradients can flow from
    later frozen blocks back into earlier prompt generators.
    """

    interaction_blocks = (2, 5, 8, 11)  # one-based block identifiers

    def __init__(self, checkpoint: str | Path, activation_checkpointing: bool = False,
                 activation_checkpointing_scope: str = "all") -> None:
        super().__init__()
        self.activation_checkpointing = bool(activation_checkpointing)
        if activation_checkpointing_scope not in {"all", "global"}:
            raise ValueError("activation_checkpointing_scope must be all or global")
        self.activation_checkpointing_scope = activation_checkpointing_scope
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Required SAM ViT-B checkpoint not found: {checkpoint}")
        try:
            from segment_anything import sam_model_registry
        except ImportError as exc:
            raise RuntimeError(
                "Official segment-anything package is required; no fallback backbone is allowed"
            ) from exc
        try:
            sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
        except Exception as exc:
            raise RuntimeError(f"Failed to load official SAM checkpoint: {checkpoint}") from exc
        self.encoder = sam.image_encoder
        # MORPH consumes the 768-channel block-12 representation directly.
        # Remove the unused 256-channel SAM neck from registration/counting.
        self.encoder.neck = nn.Identity()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self

    @staticmethod
    def preprocess(image_01: torch.Tensor) -> torch.Tensor:
        # Official SAM pixel mean/std for RGB values in [0,255].
        mean = image_01.new_tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        std = image_01.new_tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
        return (image_01 * 255.0 - mean) / std

    def forward(
        self,
        image_01: torch.Tensor,
        prompt_fn: Callable[[int, torch.Tensor], torch.Tensor | None],
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        image = self.preprocess(image_01)
        x = self.encoder.patch_embed(image)  # official SAM layout: B,H,W,C
        if self.encoder.pos_embed is not None:
            x = x + self.encoder.pos_embed
        prompts: dict[int, torch.Tensor] = {}
        # Deliberately no torch.no_grad(): graph through frozen operations is required.
        for block_id, block in enumerate(self.encoder.blocks, start=1):
            checkpoint_block = getattr(self, "activation_checkpointing_scope", "all") == "all" or block.window_size == 0
            if self.activation_checkpointing and checkpoint_block and torch.is_grad_enabled() and x.requires_grad:
                from torch.utils.checkpoint import checkpoint as recompute
                x = recompute(block, x, use_reentrant=False)
            else:
                x = block(x)
            if block_id in self.interaction_blocks:
                x_nchw = x.permute(0, 3, 1, 2).contiguous()
                prompt = prompt_fn(block_id, x_nchw)
                if prompt is not None:
                    if prompt.shape != x_nchw.shape:
                        raise ValueError(
                            f"MEAA prompt at block {block_id} has {prompt.shape}, expected {x_nchw.shape}"
                        )
                    x = x + prompt.permute(0, 2, 3, 1)
                    prompts[block_id] = prompt
        return x.permute(0, 3, 1, 2).contiguous(), prompts


class UpBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return super().forward(x)


class ProgressiveDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        channels = (512, 256, 128, 64, 32, 16)
        self.stages = nn.ModuleList(UpBlock(a, b) for a, b in zip(channels, channels[1:]))
        self.classifier = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return self.classifier(x)


class DensePromptProjection(nn.Sequential):
    def __init__(self) -> None:
        super().__init__(
            nn.Conv2d(2, 4, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(4),
            nn.GELU(),
            nn.Conv2d(4, 16, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 512, kernel_size=1, bias=False),
        )

    def forward(self, dense_map):
        projected = super().forward(dense_map)
        # Preserve the exact 0-m identity even after BatchNorm affine training.
        occupied = dense_map.abs().flatten(1).sum(1).gt(0).to(projected.dtype)
        return projected * occupied[:, None, None, None]


class MORPH(nn.Module):
    interaction_blocks = (2, 5, 8, 11)

    def __init__(self, config: dict, cnn=None, vit=None) -> None:
        super().__init__()
        model_cfg = config.get("model", config)
        if tuple(model_cfg.get("interaction_blocks", self.interaction_blocks)) != self.interaction_blocks:
            raise ValueError("Final MORPH interaction blocks must be [2, 5, 8, 11]")
        channels = (64, 128, 256, 512)
        self.frequency_mode = model_cfg.get("frequency_mode", "morph_prompt")
        sources = tuple(model_cfg.get("meaa_sources", ["cnn", "whfc", "vit"]))
        if self.frequency_mode not in {"none", "direct_add", "concat_projection", "morph_prompt"}:
            raise ValueError(f"Unknown frequency mode: {self.frequency_mode}")
        if self.frequency_mode == "none":
            sources = tuple(source for source in sources if source != "whfc")
        uses_meaa = bool(sources) and self.frequency_mode in {"none", "morph_prompt"}
        fusion_mode = model_cfg.get("fusion", "cbaf")
        self.use_cnn = (uses_meaa and "cnn" in sources) or fusion_mode in {"add", "concat", "cbaf", "cbaf_no_base_vit"}
        self.cnn = (cnn or FrozenResNet18Encoder(pretrained=True)) if self.use_cnn else None
        self.vit = vit or FrozenSAMViTBEncoder(
            model_cfg["sam_checkpoint"], activation_checkpointing=model_cfg.get("activation_checkpointing", False),
            activation_checkpointing_scope=model_cfg.get("activation_checkpointing_scope", "all")
        )
        self.use_whfc = (uses_meaa and "whfc" in sources) or self.frequency_mode in {"direct_add", "concat_projection"}
        projected_channels = channels if uses_meaa and "whfc" in sources else ()
        self.whfc = WHFC(projected_channels, levels=int(model_cfg.get("wavelet_levels", 3)),
                         wavelet=model_cfg.get("wavelet", "haar")) if self.use_whfc else None
        aggregation = model_cfg.get("aggregation", "branch_softmax")
        self.meaas = nn.ModuleList(MEAA(c, aggregation, sources) for c in channels) if uses_meaa else nn.ModuleList()
        whfc_input_channels = 9 * int(model_cfg.get("wavelet_levels", 3))
        self.frequency_direct = nn.ModuleList(
            nn.Conv2d(whfc_input_channels, 768, 1) for _ in channels
        ) if self.frequency_mode == "direct_add" else nn.ModuleList()
        self.frequency_concat = nn.ModuleList(
            nn.Conv2d(whfc_input_channels + 768, 768, 1) for _ in channels
        ) if self.frequency_mode == "concat_projection" else nn.ModuleList()
        self.cbaf = CBAF(mode=fusion_mode)
        self.dense_prompt = DensePromptProjection()
        self.decoder = ProgressiveDecoder()
        if config.get('execution', {}).get('fp32_batchnorm', False):
            from utils.precision import use_fp32_batchnorm
            use_fp32_batchnorm(self)
        if config.get('execution', {}).get('robust_batchnorm_layers'):
            from utils.precision import use_robust_batchnorm
            use_robust_batchnorm(self, config['execution']['robust_batchnorm_layers'])

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cnn is not None: self.cnn.eval()
        self.vit.encoder.eval()
        return self

    def forward(self, image_01: torch.Tensor, dense_click_map: torch.Tensor | None = None):
        if image_01.shape[1:] != (3, 1024, 1024):
            raise ValueError(f"MORPH requires Bx3x1024x1024 RGB input, got {tuple(image_01.shape)}")
        if dense_click_map is not None and dense_click_map.shape[1:] != (2, 128, 128):
            raise ValueError(f"Dense prompt must be Bx2x128x128, got {tuple(dense_click_map.shape)}")
        if dense_click_map is not None and dense_click_map.shape[0] != image_01.shape[0]:
            raise ValueError("Image and dense prompt batch sizes differ")
        cnn_features = self.cnn(image_01) if self.cnn is not None else (None,) * 4
        if self.whfc is not None:
            whfc_features, whfc_aux = self.whfc(image_01)
            if not whfc_features: whfc_features = (None,) * 4
        else:
            whfc_features, whfc_aux = (None,) * 4, {"basis": None, "details": []}
        alphas = {}

        def make_prompt(block_id: int, vit_feature: torch.Tensor):
            index = self.interaction_blocks.index(block_id)
            if self.frequency_mode in {"none", "morph_prompt"} and not self.meaas:
                return None
            if self.frequency_mode == "direct_add":
                prompt, alpha = self.frequency_direct[index](whfc_aux["basis"]), None
            elif self.frequency_mode == "concat_projection":
                enhanced = self.frequency_concat[index](torch.cat([vit_feature, whfc_aux["basis"]], dim=1))
                # The SAM adapter performs x += prompt. Subtract x so ordinary
                # concat enhancement replaces it with phi(cat(x,E)).
                prompt = enhanced - vit_feature
                alpha = None
            elif self.frequency_mode in {"none", "morph_prompt"}:
                prompt, alpha = self.meaas[index](cnn_features[index], whfc_features[index], vit_feature)
            else:
                raise ValueError(f"Unknown frequency mode: {self.frequency_mode}")
            alphas[block_id] = alpha
            return prompt

        final_vit, prompts = self.vit(image_01, make_prompt)
        fused = self.cbaf(cnn_features[-1], final_vit)
        return {
            **self.decode_fused(fused, dense_click_map),
            "cnn_features": cnn_features,
            "vit_feature": final_vit,
            "whfc": whfc_aux,
            "whfc_stage_features": whfc_features,
            "prompts": prompts,
            "branch_weights": alphas,
        }

    def decode_fused(self, unconditioned_fused, dense_click_map=None):
        """Decode the same image representation with another dense prompt.

        All image-dependent branches precede the dense prompt. Reuse is valid
        during eval, with BN/dropout in eval mode; callers enforce this mode.
        """
        if dense_click_map is not None and (
            dense_click_map.shape[1:] != (2, 128, 128) or dense_click_map.shape[0] != unconditioned_fused.shape[0]
        ):
            raise ValueError("Dense prompt shape/batch mismatch")
        dense_prompt = torch.zeros_like(unconditioned_fused)
        fused = unconditioned_fused
        if dense_click_map is not None:
            dense_prompt = self.dense_prompt(dense_click_map)
            fused = fused + dense_prompt
        return {"logits": self.decoder(fused), "fused": fused,
                "unconditioned_fused": unconditioned_fused, "dense_prompt": dense_prompt}
