from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from pytorch_wavelets import DWTForward


class FixedWaveletDWT(nn.Module):
    """Single-level library adapter retained for coefficient diagnostics."""

    def __init__(self, wavelet: str = "haar") -> None:
        super().__init__()
        self.dwt = DWTForward(J=1, wave=wavelet, mode="symmetric")
        self.dwt.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        lowpass, highpass = self.dwt(x)
        # Library order on dimension 2: LH (horizontal), HL (vertical), HH.
        return lowpass, *highpass[0].unbind(dim=2)


class FixedHaarDWT(FixedWaveletDWT):
    """Single-level Haar diagnostic adapter; no custom analysis kernels."""

    def __init__(self) -> None:
        super().__init__("haar")


class WHFC(nn.Module):
    """Fixed RGB DWT and signed high-frequency basis with trainable alignment.

    Main model: DWTForward(J=3, wave="haar", mode="symmetric").
    Input is geometrically preprocessed RGB in [0,1], before backbone
    normalization. The library decomposes each RGB channel independently.
    Other depths/families remain explicit ablation options, not the main model.
    """

    implementation = "pytorch_wavelets_1.3.0_LH_HL_HH"

    def __init__(self, stage_channels=(64, 128, 256, 512), levels: int = 3,
                 wavelet: str = "haar", mode: str = "symmetric") -> None:
        super().__init__()
        if not isinstance(levels, int) or isinstance(levels, bool) or levels < 1:
            raise ValueError("levels must be a positive integer")
        if mode != "symmetric":
            raise ValueError("WHFC requires symmetric wavelet boundary mode")
        self.levels = levels
        self.wavelet = wavelet
        self.mode = mode
        self.dwt = DWTForward(J=levels, wave=wavelet, mode=mode)
        self.dwt.requires_grad_(False)
        self.projections = nn.ModuleList(
            nn.Conv2d(9 * levels, channels, kernel_size=1, bias=True)
            for channels in stage_channels
        )

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if prefix + "dwt.filters" in state_dict:
            error_msgs.append(
                "Legacy custom-DWT WHFC checkpoint detected. The new "
                "pytorch_wavelets/symmetric implementation requires fresh "
                "training and new evaluation; legacy exact resume is forbidden."
            )
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, image_01: torch.Tensor, size=(64, 64)):
        if image_01.ndim != 4 or image_01.shape[1] != 3:
            raise ValueError("WHFC input must be Bx3xHxW RGB")
        if not image_01.is_floating_point():
            raise ValueError("WHFC input must be floating-point RGB in [0,1]")
        if min(image_01.shape[-2:]) < 2 ** self.levels:
            raise ValueError("WHFC input is too small for the requested depth")
        with torch.no_grad():
            valid = (torch.isfinite(image_01).all()
                     & (image_01.amin() >= 0) & (image_01.amax() <= 1))
            if not bool(valid):
                raise ValueError("WHFC input must be finite RGB in [0,1]; normalize backbones separately")

        details, groups = [], []
        # Keep fixed coefficients and resizing in the filter dtype (normally
        # FP32), including inside BF16 training. Do not detach the input:
        # fixed filters can still propagate gradients to an upstream tensor.
        with torch.autocast(device_type=image_01.device.type, enabled=False):
            lowpass, highpass = self.dwt(image_01.to(dtype=self.dwt.h0_col.dtype))
            # highpass[0] is the finest scale, highpass[-1] the coarsest.
            for bands in highpass:
                dh, dv, dd = bands.unbind(dim=2)
                details.append((dh, dv, dd))
                # Direction-major RGB layout: LH_RGB, HL_RGB, HH_RGB.
                group = torch.cat([dh, dv, dd], dim=1)
                groups.append(F.interpolate(
                    group, size=size, mode="bilinear",
                    align_corners=False, antialias=False,
                ))
            basis = torch.cat(groups, dim=1)
        features = tuple(projection(basis) for projection in self.projections)
        return features, {"basis": basis, "details": details, "lowpass": lowpass}
