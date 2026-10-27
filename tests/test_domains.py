import pytest
import torch
from lyapuflow import ControllerConfig, LyapuFlow
from lyapuflow.models import ConvFlow, flow_matching_loss
from lyapuflow.operators import *


@pytest.mark.parametrize("domain,shape", [("image", (2, 3, 16, 16)), ("audio", (2, 1, 256))])
def test_training_and_full_feedback(domain, shape):
    torch.set_num_threads(2)
    model = ConvFlow(domain, shape[1], width=8, depth=2)
    optimizer = torch.optim.Adam(model.parameters())
    data = torch.rand(shape)*2-1
    loss = flow_matching_loss(model, data)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    optimizer.step()
    model.eval().requires_grad_(False)
    controller = LyapuFlow(model, Measurement(Identity(), data), ControllerConfig(steps=2))
    result = controller.sample(torch.randn_like(data))
    assert result.sample.shape == data.shape and torch.isfinite(result.sample).all()


@pytest.mark.parametrize("op,shape", [
    (ImageSuperResolution(4), (1, 3, 16, 16)),
    (ImageBlur("gaussian", 5), (1, 3, 16, 16)),
    (ImageBlur("motion", 5, angle=45), (1, 3, 16, 16)),
    (AudioLowPass(factor=4, cutoff=1500), (1, 1, 256)),
    (AudioReverb([1, .5, .2]), (1, 1, 256)),
    (Mask(torch.tensor([[[1., 0., 1.]]])), (1, 1, 3)),
])
def test_operator_gradients(op, shape):
    x = torch.randn(shape, requires_grad=True)
    out = op(x)
    out.square().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_causal_reverb():
    x = torch.zeros(1, 1, 8)
    x[..., 0] = 1
    y = AudioReverb([1, .5, .25])(x)
    torch.testing.assert_close(y[0, 0], torch.tensor([1., .5, .25, 0, 0, 0, 0, 0]))


def test_guidance_reduces_measurement_error_both_domains():
    for shape in [(1, 3, 8, 8), (1, 1, 256)]:
        torch.manual_seed(2)
        target = torch.rand(shape)*0.5
        initial = torch.randn(shape)
        flow = lambda x, t: -0.5*x / (1-0.5*t).reshape(-1, *([1]*(x.ndim-1)))
        residual = Measurement(Identity(), target)
        c = LyapuFlow(flow, residual, ControllerConfig(steps=100, alpha=10, trust_region=5))
        guided = c.sample(initial, record=False).sample
        baseline = LyapuFlow(flow, residual, ControllerConfig(steps=100, alpha=0, trust_region=0)).sample(initial, record=False).sample
        assert (guided-target).square().mean() < (baseline-target).square().mean() * 0.1
