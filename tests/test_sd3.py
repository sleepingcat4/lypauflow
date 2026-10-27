from types import SimpleNamespace
import torch
from torch import nn
from lyapuflow.sd3 import SD3Prior
from lyapuflow import LyapuFlow, ControllerConfig


class Transformer(nn.Module):
    dtype = torch.float32
    config = SimpleNamespace(patch_size=2, in_channels=2)

    def forward(self, hidden_states, timestep, encoder_hidden_states, pooled_projections, return_dict):
        # distinct conditional and unconditional predictions, plus time dependence
        return (hidden_states * 0.1 + timestep[:, None, None, None]/1000
                + encoder_hidden_states[:, :1, :1, None],)


class VAE(nn.Module):
    dtype = torch.float32
    config = SimpleNamespace(scaling_factor=2., shift_factor=.5)

    def decode(self, z, return_dict):
        return (z*3,)


class Pipeline:
    def __init__(self):
        self.transformer, self.vae = Transformer(), VAE()
        self.scheduler = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        self.vae_scale_factor = 8

    def to(self, device):
        return self

    def encode_prompt(self, **kwargs):
        return torch.ones(1, 1, 1), torch.zeros(1, 1, 1), torch.ones(1, 1), torch.zeros(1, 1)


def test_sd3_sign_cfg_decoder_and_gradients():
    prior = SD3Prior("unused", "cat", "cpu", 2, pipeline=Pipeline())
    x = torch.ones(1, 2, 4, 4, requires_grad=True)
    t = torch.tensor([.25], requires_grad=True)
    v = prior.velocity(x, t)
    torch.testing.assert_close(v, torch.full_like(x, -2.85))
    dx, dt = torch.autograd.grad(v.sum(), (x, t))
    torch.testing.assert_close(dx, torch.full_like(x, -.1))
    torch.testing.assert_close(dt, torch.tensor([float(x.numel())]))
    decoded = prior.decode(x)
    torch.testing.assert_close(decoded, torch.full_like(x, 3.))
    assert prior.noise(32, 32).shape == (1, 2, 4, 4)
    c = LyapuFlow(prior.velocity, lambda x: x, ControllerConfig(steps=2, switch_step=1), prior.decode)
    result = c.sample(x.detach())
    assert torch.isfinite(result.sample).all()
