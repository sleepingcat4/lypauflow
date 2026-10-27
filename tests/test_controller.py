import pytest
import torch
from lyapuflow import ControllerConfig, LyapuFlow
from lyapuflow.controller import dot


def test_full_derivatives_match_analytic_time_dependent_flow():
    # v=a*x+b*t; endpoint=e, d e/dx=1+(1-t)a,
    # d e/dt=-v+(1-t)b. This catches omission of explicit time derivative.
    x = torch.tensor([[0.7, -0.2], [-0.4, 0.8]], dtype=torch.float64)
    t = torch.tensor([0.2, 0.6], dtype=x.dtype)
    a, b, alpha = 0.4, 0.3, 2.0
    flow = lambda x, t: a*x + b*t[:, None]
    controller = LyapuFlow(flow, lambda x: x, ControllerConfig(alpha=alpha, trust_region=None))
    v, u, stats = controller.feedback(x, t)
    e = x + (1-t[:, None])*v
    g = e*(1+(1-t[:, None])*a)
    q = (e*(-v+(1-t[:, None])*b)).sum(1)
    energy = e.square().sum(1)/2
    demand = (q + dot(g, v) + alpha*energy).clamp_min(0)
    expected = -demand[:, None]/dot(g, g)[:, None]*g
    torch.testing.assert_close(u, expected)
    torch.testing.assert_close(stats["drift"], q+dot(g, v))
    assert (stats["slack"] < 1e-12).all()


def test_inactive_when_nominal_already_decreases_energy():
    controller = LyapuFlow(lambda x, t: -2*x, lambda x: x,
                           ControllerConfig(alpha=0.1, trust_region=None))
    _, u, stats = controller.feedback(torch.ones(2, 3), 0.1)
    assert torch.equal(u, torch.zeros_like(u))
    assert (stats["drift"] < 0).all()


def test_trust_region_zero_velocity_and_degenerate_gradient():
    x = torch.ones(2, 3)
    c = LyapuFlow(lambda x, t: 0.01*x, lambda x: x, ControllerConfig(alpha=100, trust_region=0.2))
    v, u, stats = c.feedback(x, 0.5)
    assert (u.norm(dim=1) <= 0.2*v.norm(dim=1)+1e-7).all()
    assert stats["clipped"].all() and (stats["slack"] > 0).all()
    c = LyapuFlow(lambda x, t: x*0, lambda x: x, ControllerConfig())
    _, u, _ = c.feedback(x, 0.5)
    assert u.count_nonzero() == 0
    c = LyapuFlow(lambda x, t: x, lambda x: torch.ones_like(x), ControllerConfig())
    _, u, stats = c.feedback(x, 0.5)
    assert u.count_nonzero() == 0 and stats["degenerate"].all()
    assert (stats["slack"] > 0).all()


def test_stop_gradient_cancellation_and_latent_decoder():
    x = torch.tensor([[1., 2.], [2., 3.]])
    c = LyapuFlow(lambda x, t: x*t[:, None], lambda x: x-1,
                  ControllerConfig(alpha=3, trust_region=None), decoder=lambda z: z*2)
    t = 0.4
    _, u, stats = c.feedback(x, t, stop_velocity=True)
    endpoint = x + (1-t)*x*t
    r = 2*endpoint-1
    g = 2*r
    expected = -3*stats["energy"][:, None] / g.square().sum(1)[:, None]*g
    torch.testing.assert_close(u, expected)
    torch.testing.assert_close(stats["drift"], torch.zeros(2))


def test_batch_independence_and_sampling_has_no_graph():
    x = torch.randn(3, 2, 4)
    flow = lambda x, t: x*0.2 + t[:, None, None]
    c = LyapuFlow(flow, lambda x: x, ControllerConfig(steps=4, switch_step=2))
    result = c.sample(x)
    separate = torch.cat([c.sample(row[None]).sample for row in x])
    torch.testing.assert_close(result.sample, separate)
    assert not result.sample.requires_grad and len(result.history) == 4
    assert all(not v.requires_grad for h in result.history for v in h.values())


def test_feedback_agrees_with_finite_difference_drift():
    x = torch.tensor([[0.3, 0.6]], dtype=torch.float64)
    flow = lambda x, t: x.sin() * (1+t[:, None])
    c = LyapuFlow(flow, lambda x: x.square()-1, ControllerConfig())
    v, _, stats = c.feedback(x, 0.3)
    def energy(x, t):
        e = x+(1-t)*flow(x, torch.tensor([t], dtype=x.dtype))
        return 0.5*(e.square()-1).square().sum()
    eps = 1e-6
    numerical = (energy(x+eps*v, 0.3+eps)-energy(x-eps*v, 0.3-eps))/(2*eps)
    torch.testing.assert_close(stats["drift"][0], numerical, atol=1e-8, rtol=1e-6)


def test_invalid_grid_and_nonfinite():
    c = LyapuFlow(lambda x, t: x, lambda x: x, ControllerConfig(steps=2))
    with pytest.raises(ValueError):
        c.sample(torch.ones(1, 2), times=torch.tensor([0., 0., 1.]))
    c = LyapuFlow(lambda x, t: x*float("nan"), lambda x: x)
    with pytest.raises(FloatingPointError):
        c.feedback(torch.ones(1, 2), 0.2)
