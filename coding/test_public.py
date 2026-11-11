"""Small numerical/gradient checks; these do not establish training quality."""
import pytest
import torch
import algorithms as a


def test_returns():
    r = torch.tensor([1., -1., 3.])
    assert torch.allclose(a.returns_to_go(r, .9), torch.tensor([2.53, 1.7, 3.]))
    assert torch.equal(r, torch.tensor([1., -1., 3.]))
    assert torch.allclose(a.returns_to_go(torch.tensor([4.]), .9), torch.tensor([4.]))


def test_actor_gradient():
    lp = torch.tensor([-.5, -1.], requires_grad=True)
    adv = torch.tensor([2., -1.], requires_grad=True)
    loss = a.policy_loss(lp, adv, torch.tensor([1., .9]), 1)
    assert loss.ndim == 0
    assert loss.item() == pytest.approx(.1)
    loss.backward()
    assert torch.allclose(lp.grad, torch.tensor([-2., .9]))
    assert adv.grad is None


def test_baseline():
    v = torch.tensor([1., 2.], requires_grad=True)
    g = torch.tensor([3., 1.], requires_grad=True)
    adv = a.baseline_advantages(g, v)
    assert torch.allclose(adv, torch.tensor([2., -1.]))
    assert not adv.requires_grad
    loss = a.value_loss(v, g)
    assert loss.item() == pytest.approx(2.5)
    loss.backward()
    assert torch.allclose(v.grad, torch.tensor([-2., 1.]))
    assert g.grad is None


def test_td():
    nxt = torch.tensor([5., 99.], requires_grad=True)
    y = a.td_targets(torch.tensor([1., -2.]), nxt, torch.tensor([False, True]), .9)
    assert torch.allclose(y, torch.tensor([5.5, -2.]))
    assert not y.requires_grad
    adv = a.td_advantages(y, torch.tensor([2., 3.], requires_grad=True))
    assert torch.allclose(adv, torch.tensor([3.5, -5.]))
    assert not adv.requires_grad


@pytest.mark.extension
def test_nstep():
    r = torch.tensor([1., -1., 3.])
    v = torch.tensor([2., 1., 4., 0.], requires_grad=True)
    assert torch.allclose(a.n_step_targets(r, v, .9, 1), torch.tensor([1.9, 2.6, 3.]))
    assert torch.allclose(a.n_step_targets(r, v, .9, 2), torch.tensor([3.34, 1.7, 3.]))
    result = a.n_step_targets(r, v, .9, 5)
    assert torch.allclose(result, torch.tensor([2.53, 1.7, 3.]))
    assert not result.requires_grad

