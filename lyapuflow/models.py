"""Small time-conditioned convolutional flow priors for domain training.

These are practical baselines, not the pretrained SD3.5 model from the paper.
No batch-dependent layers: the controller's per-sample gradients stay valid.
"""
import math
import torch
from torch import nn


class TimeEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer("freq", torch.exp(torch.linspace(0, math.log(1000), width // 2)))
        self.net = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, t):
        angles = t[:, None] * self.freq[None] * (2 * math.pi)
        return self.net(torch.cat((angles.sin(), angles.cos()), dim=1))


class ResidualBlock(nn.Module):
    def __init__(self, dim, width, dilation):
        super().__init__()
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        self.norm1 = nn.GroupNorm(8, width)
        self.norm2 = nn.GroupNorm(8, width)
        self.conv1 = conv(width, width, 3, padding=dilation, dilation=dilation)
        self.conv2 = conv(width, width, 3, padding=1)
        self.time = nn.Linear(width, width)
        self.act = nn.SiLU()
        self.dim = dim

    def forward(self, x, t):
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time(t).reshape(-1, h.shape[1], *([1] * self.dim))
        return x + self.conv2(self.act(self.norm2(h)))


class ConvFlow(nn.Module):
    def __init__(self, domain="image", channels=3, width=64, depth=8):
        super().__init__()
        if domain not in ("image", "audio") or width % 8 or width < 8 or depth < 1 or channels < 1:
            raise ValueError("invalid domain/channels/depth; width must be a positive multiple of 8")
        self.spec = dict(domain=domain, channels=channels, width=width, depth=depth)
        dim = 1 if domain == "audio" else 2
        conv = nn.Conv1d if dim == 1 else nn.Conv2d
        self.time = TimeEmbedding(width)
        self.input = conv(channels, width, 3, padding=1)
        self.blocks = nn.ModuleList(ResidualBlock(dim, width, 2**(i % 4)) for i in range(depth))
        self.output = conv(width, channels, 3, padding=1)

    def forward(self, x, t):
        t = self.time(t)
        h = self.input(x)
        for block in self.blocks:
            h = block(h, t)
        return self.output(h)


def flow_matching_loss(model, data):
    """Straight-line conditional flow matching: x_t=(1-t)noise+t*data."""
    noise = torch.randn_like(data)
    t = torch.rand(data.shape[0], device=data.device, dtype=data.dtype)
    tb = t.reshape(-1, *([1] * (data.ndim - 1)))
    prediction = model((1-tb) * noise + tb * data, t)
    return (prediction - (data - noise)).square().mean()
