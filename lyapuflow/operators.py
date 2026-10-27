"""Differentiable measurement operators for images [B,C,H,W] and audio [B,C,T]."""
import math
import torch
import torch.nn.functional as F
from torch import nn


class Measurement(nn.Module):
    def __init__(self, operator, observation):
        super().__init__()
        self.operator = operator
        self.register_buffer("observation", observation.detach().clone())

    def forward(self, x):
        predicted = self.operator(x)
        if predicted.shape != self.observation.shape:
            raise ValueError(f"measurement shape {predicted.shape} != observation {self.observation.shape}")
        return predicted - self.observation


class Identity(nn.Module):
    def forward(self, x):
        return x


class Mask(nn.Module):
    """1 denotes observed samples/pixels; 0 denotes missing data."""
    def __init__(self, mask):
        super().__init__()
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("mask must be binary")
        self.register_buffer("mask", mask)

    def forward(self, x):
        return x * self.mask


class ImageSuperResolution(nn.Module):
    def __init__(self, factor=4):
        super().__init__()
        if not isinstance(factor, int) or factor < 1:
            raise ValueError("factor must be a positive integer")
        self.factor = factor

    def forward(self, x):
        if min(x.shape[-2:]) < self.factor:
            raise ValueError("image smaller than super-resolution factor")
        return F.interpolate(x, size=(x.shape[-2] // self.factor, x.shape[-1] // self.factor),
                             mode="bicubic", align_corners=False, antialias=True)


class ImageBlur(nn.Module):
    def __init__(self, kind="gaussian", size=61, sigma=3.0, angle=0.0):
        super().__init__()
        if size < 1 or size % 2 != 1 or sigma <= 0:
            raise ValueError("size must be positive odd and sigma positive")
        a = torch.arange(size, dtype=torch.float32) - size // 2
        yy, xx = torch.meshgrid(a, a, indexing="ij")
        if kind == "gaussian":
            kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma**2))
        elif kind == "motion":
            theta = math.radians(angle)
            # Rasterized straight-line motion PSF, explicit extension where the
            # paper does not specify its precise motion-kernel construction.
            distance = -xx * math.sin(theta) + yy * math.cos(theta)
            kernel = (distance.abs() <= 0.5).float()
        else:
            raise ValueError("kind must be gaussian or motion")
        self.register_buffer("kernel", (kernel / kernel.sum())[None, None])
        self.padding = size // 2

    def forward(self, x):
        kernel = self.kernel.to(x).expand(x.shape[1], 1, -1, -1)
        # Replicate boundaries support even small demo images.
        return F.conv2d(F.pad(x, (self.padding,) * 4, mode="replicate"), kernel, groups=x.shape[1])


class AudioLowPass(nn.Module):
    """Windowed-sinc FIR followed by optional decimation; anti-aliasing included."""
    def __init__(self, sample_rate=16000, cutoff=3500.0, factor=1, taps=101):
        super().__init__()
        if factor < 1 or not isinstance(factor, int) or taps < 3 or taps % 2 != 1:
            raise ValueError("factor must be positive integer; taps must be odd >=3")
        if not 0 < cutoff <= sample_rate / (2 * factor):
            raise ValueError("cutoff must be positive and no greater than output Nyquist")
        n = torch.arange(taps, dtype=torch.float32) - taps // 2
        fc = cutoff / sample_rate
        kernel = 2 * fc * torch.sinc(2 * fc * n) * torch.hann_window(taps, periodic=False)
        self.register_buffer("kernel", (kernel / kernel.sum())[None, None])
        self.factor, self.padding = factor, taps // 2

    def forward(self, x):
        kernel = self.kernel.to(x).expand(x.shape[1], 1, -1)
        return F.conv1d(F.pad(x, (self.padding,) * 2, mode="replicate"), kernel,
                        stride=self.factor, groups=x.shape[1])


class AudioReverb(nn.Module):
    """Known causal room impulse response for dereverberation."""
    def __init__(self, impulse_response):
        super().__init__()
        impulse_response = torch.as_tensor(impulse_response, dtype=torch.float32)
        if impulse_response.ndim != 1 or impulse_response.numel() == 0:
            raise ValueError("impulse response must be a nonempty vector")
        self.register_buffer("ir", impulse_response)

    def forward(self, x):
        kernel = self.ir.to(x).flip(0)[None, None].expand(x.shape[1], 1, -1)
        return F.conv1d(F.pad(x, (len(self.ir)-1, 0)), kernel, groups=x.shape[1])
