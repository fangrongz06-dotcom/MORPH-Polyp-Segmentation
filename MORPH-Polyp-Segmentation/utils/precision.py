"""Explicit run-level precision policy; stable reductions remain FP32."""
from contextlib import nullcontext

import torch
from torch import nn


class NumericalForwardError(RuntimeError):
    """A detected numerical failure eligible for the guarded FP32 replay."""


class RobustBatchNorm2d(nn.BatchNorm2d):
    """Explicit BN equations with FP64 statistics for selected small features.

    Training normalization uses biased variance; running variance uses its
    unbiased correction, matching BatchNorm. Affine parameters stay FP32.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.track_running_stats:
            self.running_mean = self.running_mean.double()
            self.running_var = self.running_var.double()

    def forward(self, input):
        self._check_input_dim(input)
        self._last_bn_input = input.detach()
        self._last_bn_variance = None
        with torch.autocast(device_type=input.device.type, enabled=False):
            if not bool(torch.isfinite(input).all()):
                raise NumericalForwardError('Non-finite input to robust BatchNorm')
            x = input.double()
            use_batch = self.training or self.running_mean is None
            if use_batch:
                count = x.numel() // x.shape[1]
                if count <= 1:
                    raise ValueError('Expected more than 1 value per channel when training')
                variance, mean = torch.var_mean(x, dim=(0, 2, 3), unbiased=False, keepdim=True)
                self._last_bn_variance = variance.detach()
            else:
                mean = self.running_mean.view(1, -1, 1, 1)
                variance = self.running_var.view(1, -1, 1, 1)
            if not bool((torch.isfinite(mean).all() & torch.isfinite(variance).all() & (variance >= 0).all())):
                raise NumericalForwardError('Invalid robust BatchNorm statistics')
            output = ((x - mean) * torch.rsqrt(variance + self.eps)).float()
            if self.affine:
                output = output * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
            if not bool(torch.isfinite(output).all()):
                raise NumericalForwardError('Non-finite output from robust BatchNorm affine transform')
            if self.training and self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked.add_(1)
                    momentum = self.momentum if self.momentum is not None else 1. / int(self.num_batches_tracked)
                    self.running_mean.lerp_(mean.detach().reshape(-1), momentum)
                    self.running_var.lerp_(variance.detach().reshape(-1) * (count / (count - 1)), momentum)
            return output


def use_robust_batchnorm(model, layer_names):
    if not layer_names or len(layer_names) != len(set(layer_names)):
        raise ValueError('Explicit unique robust BatchNorm layer names are required')
    for path in layer_names:
        parts = path.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child = getattr(parent, parts[-1])
        if not isinstance(child, nn.BatchNorm2d):
            raise ValueError('Requested robust layer is not BatchNorm2d: ' + path)
        replacement = RobustBatchNorm2d(child.num_features, child.eps, child.momentum,
                                      child.affine, child.track_running_stats)
        if child.affine:
            replacement.weight, replacement.bias = child.weight, child.bias
        replacement.running_mean = child.running_mean.double() if child.running_mean is not None else None
        replacement.running_var = child.running_var.double() if child.running_var is not None else None
        replacement.num_batches_tracked = child.num_batches_tracked
        replacement.train(child.training)
        setattr(parent, parts[-1], replacement)
    return model


class FP32BatchNorm2d(nn.BatchNorm2d):
    """Use native FP32 normalization/statistics inside a BF16 model forward.

    Keep parameters and state_dict keys unchanged. Disabling autocast alone
    would leave an already-BF16 input unchanged; the explicit float() matters.
    """
    def forward(self, input):
        with torch.autocast(device_type=input.device.type, enabled=False):
            with torch.backends.cudnn.flags(enabled=False):
                return super().forward(input.float())


def use_fp32_batchnorm(model):
    for name, child in list(model.named_children()):
        if isinstance(child, nn.BatchNorm2d) and not isinstance(child, FP32BatchNorm2d):
            replacement = FP32BatchNorm2d(
                child.num_features, eps=child.eps, momentum=child.momentum,
                affine=child.affine, track_running_stats=child.track_running_stats)
            # Reuse the existing objects, including frozen requires_grad flags.
            if child.affine:
                replacement.weight, replacement.bias = child.weight, child.bias
            replacement.running_mean = child.running_mean
            replacement.running_var = child.running_var
            replacement.num_batches_tracked = child.num_batches_tracked
            replacement.train(child.training)
            setattr(model, name, replacement)
        else:
            use_fp32_batchnorm(child)
    return model


def autocast_context(execution, device):
    precision = execution.get("precision", "float32")
    if precision == "float32":
        return nullcontext()
    if precision != "amp_bfloat16":
        raise ValueError("Unsupported precision: " + str(precision))
    device_type = torch.device(device).type
    return torch.autocast(device_type=device_type, dtype=torch.bfloat16)


def configure_precision(execution, device):
    precision = execution.get("precision", "float32")
    if precision not in {"float32", "amp_bfloat16"}:
        raise ValueError("Unsupported precision: " + str(precision))
    if torch.device(device).type == "cuda" and precision == "amp_bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unsupported on this GPU/runtime; activate the intended CUDA environment")
    tf32 = bool(execution.get("allow_tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
