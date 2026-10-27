"""Algorithms 1/2 and Eqs. (5–23), arXiv:2610.04326.

Model contract: velocity(x, t) -> x-shaped tensor, t has shape [B].
Samples must be independent across the batch (no training-mode BatchNorm).
Residual and decoder must be differentiable and preserve the batch axis.
"""
from dataclasses import dataclass
from typing import Callable
import math
import torch
from torch import Tensor


def dot(a: Tensor, b: Tensor) -> Tensor:
    return (a * b).flatten(1).sum(1)


def expand(a: Tensor, x: Tensor) -> Tensor:
    return a.reshape(-1, *([1] * (x.ndim - 1)))


@dataclass(frozen=True)
class ControllerConfig:
    steps: int = 100
    alpha: float = 5.0
    trust_region: float | None = 1.0
    # 0 = full derivatives; steps = stop velocity gradients throughout.
    switch_step: int = 0
    gradient_epsilon: float = 1e-12

    def __post_init__(self):
        if self.steps < 1 or not 0 <= self.switch_step <= self.steps:
            raise ValueError("steps must be positive; switch_step must lie in [0, steps]")
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be finite and nonnegative")
        if self.trust_region is not None and (not math.isfinite(self.trust_region) or self.trust_region < 0):
            raise ValueError("trust_region must be None or finite and nonnegative")
        if not math.isfinite(self.gradient_epsilon) or self.gradient_epsilon <= 0:
            raise ValueError("gradient_epsilon must be finite and positive")


@dataclass
class SampleResult:
    sample: Tensor
    state: Tensor
    history: list[dict[str, Tensor]]


class LyapuFlow:
    def __init__(self, velocity: Callable, residual: Callable, config=ControllerConfig(), decoder=None):
        self.velocity = velocity
        self.residual = residual
        self.config = config
        self.decoder = decoder if decoder is not None else lambda x: x

    @torch.enable_grad()
    def feedback(self, state: Tensor, time: float | Tensor, *, stop_velocity=False):
        """Return nominal velocity, bounded feedback, and detached per-sample diagnostics.

        With full derivatives, q = partial_t V includes the explicit -v term.
        With stopgrad, q + g.v cancels exactly, yielding Eq. (22).
        Degenerate gradients receive zero control and report remaining slack.
        """
        x = state.detach().requires_grad_(True)
        t = torch.as_tensor(time, device=x.device, dtype=x.dtype)
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        if t.shape != (x.shape[0],):
            raise ValueError("time must be scalar or shape [batch]")
        t = t.detach().clone().requires_grad_(True)
        if stop_velocity:
            with torch.no_grad():
                v = self.velocity(x, t)
        else:
            v = self.velocity(x, t)
        if v.shape != x.shape:
            raise ValueError("velocity must have the same shape as state")
        endpoint = x + expand(1 - t, x) * v
        r = self.residual(self.decoder(endpoint))
        if r.ndim < 2 or r.shape[0] != x.shape[0]:
            raise ValueError("residual must have shape [batch, ...] with at least two axes")
        energy = 0.5 * r.flatten(1).square().sum(1)
        if energy.requires_grad:
            g, q = torch.autograd.grad(energy.sum(), (x, t), allow_unused=True)
            g = torch.zeros_like(x) if g is None else g
            q = torch.zeros_like(t) if q is None else q
        else:
            g, q = torch.zeros_like(x), torch.zeros_like(t)
        v, g, q, energy = (a.detach() for a in (v, g, q, energy))
        drift = torch.zeros_like(energy) if stop_velocity else q + dot(g, v)
        demand = (drift + self.config.alpha * energy).clamp_min(0)
        norm2 = dot(g, g)
        gain = torch.where(norm2 > self.config.gradient_epsilon,
                           demand / norm2.clamp_min(self.config.gradient_epsilon), 0)
        raw = -expand(gain, g) * g
        raw_norm = dot(raw, raw).sqrt()
        v_norm = dot(v, v).sqrt()
        if self.config.trust_region is None:
            scale = torch.ones_like(raw_norm)
        else:
            bound = self.config.trust_region * v_norm
            scale = torch.where(raw_norm > 0, (bound / raw_norm.clamp_min(torch.finfo(x.dtype).tiny)).clamp(max=1), 1)
        u = expand(scale, raw) * raw
        stats = dict(time=t.detach(), energy=energy, drift=drift,
                     gradient_norm=norm2.sqrt(), velocity_norm=v_norm,
                     raw_control_norm=raw_norm, control_norm=dot(u, u).sqrt(),
                     slack=(drift + dot(g, u) + self.config.alpha * energy).clamp_min(0),
                     degenerate=norm2 <= self.config.gradient_epsilon,
                     clipped=scale < 1)
        if not all(torch.isfinite(a).all() for a in (v, u, energy, g, q)):
            raise FloatingPointError("non-finite model, residual, or feedback; check model precision and inputs")
        return v, u, stats

    def sample(self, initial: Tensor, *, times: Tensor | None = None, record=True) -> SampleResult:
        """Euler integrate noise at t=0 to data at t=1; retain no sampling graph."""
        if initial.ndim < 2 or not initial.is_floating_point() or initial.shape[0] == 0:
            raise ValueError("initial must be a nonempty floating tensor [batch, ...]")
        cfg = self.config
        grid = (torch.linspace(0, 1, cfg.steps + 1, device=initial.device, dtype=initial.dtype)
                if times is None else torch.as_tensor(times, device=initial.device, dtype=initial.dtype))
        if (grid.ndim != 1 or len(grid) != cfg.steps + 1 or not torch.isfinite(grid).all()
                or grid[0] != 0 or grid[-1] != 1 or not (grid[1:] > grid[:-1]).all()):
            raise ValueError("times must contain steps+1 strictly increasing values from 0 to 1")
        x, history = initial.detach().clone(), []
        for k in range(cfg.steps):
            v, u, stats = self.feedback(x, grid[k], stop_velocity=k < cfg.switch_step)
            x = (x + (grid[k+1] - grid[k]) * (v + u)).detach()
            if not torch.isfinite(x).all():
                raise FloatingPointError("non-finite Euler state")
            if record:
                history.append({name: value.cpu() for name, value in stats.items()})
        with torch.no_grad():
            decoded = self.decoder(x).detach()
        return SampleResult(decoded, x, history)
