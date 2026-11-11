"""Small synthetic-batch checks for the C4 training connections."""
from types import SimpleNamespace
import pytest
import torch
import train
from models import Agent


@pytest.mark.parametrize("method", ["reinforce", "baseline", "actor_critic"])
def test_update_pipeline(method, monkeypatch):
    torch.manual_seed(7)
    agent = Agent()
    obs = torch.randn(3, 9)
    nxt = torch.randn(3, 9)
    rewards = torch.tensor([1., -1., 3.])
    ended = torch.tensor([False, False, True])
    episodes = [dict(obs=obs, next_obs=nxt, actions=torch.tensor([0, 1, 2]),
                     rewards=rewards, ended=ended)]
    with torch.no_grad():
        expected = (torch.tensor([2.53, 1.7, 3.]) if method != "actor_critic"
                    else rewards + .9 * torch.where(
                        ended, torch.zeros(3), agent.values(nxt)))
        expected_weights = (expected if method == "reinforce"
                            else expected - agent.values(obs))
    old_actor = [p.detach().clone() for p in agent.policy.parameters()]
    old_value = [p.detach().clone() for p in agent.value.parameters()]
    calls = {"actor": 0, "value": 0}
    original_policy, original_value = train.alg.policy_loss, train.alg.value_loss

    def check_policy(log_probs, weights, discounts, count):
        assert log_probs.shape == (3,) and log_probs.requires_grad
        assert torch.allclose(weights, expected_weights)
        assert torch.allclose(discounts, torch.tensor([1., .9, .81]))
        assert count == 1
        calls["actor"] += 1
        return original_policy(log_probs, weights, discounts, count)

    def check_value(values, targets):
        assert values.shape == (3,) and values.requires_grad
        assert torch.allclose(targets, expected)
        assert not targets.requires_grad
        calls["value"] += 1
        return original_value(values, targets)

    monkeypatch.setattr(train.alg, "policy_loss", check_policy)
    monkeypatch.setattr(train.alg, "value_loss", check_value)
    train.update(agent, torch.optim.Adam(agent.policy.parameters(), lr=.001),
                 torch.optim.Adam(agent.value.parameters(), lr=.001),
                 episodes, SimpleNamespace(method=method, gamma=.9, n_step=1))
    assert calls == {"actor": 1, "value": 0 if method == "reinforce" else 5}
    assert any(not torch.equal(a, b) for a, b in zip(old_actor, agent.policy.parameters()))
    value_changed = any(not torch.equal(a, b) for a, b in zip(old_value, agent.value.parameters()))
    assert value_changed == (method != "reinforce")
