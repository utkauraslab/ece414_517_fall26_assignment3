"""Provided networks; no student edits required."""
import torch
from torch import nn


def make_network(output_size):
    # Eight environment observations plus normalized remaining time.
    return nn.Sequential(nn.Linear(9, 64), nn.Tanh(), nn.Linear(64, 64),
                         nn.Tanh(), nn.Linear(64, output_size))


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy = make_network(4)
        self.value = make_network(1)

    def distribution(self, observations):
        """Float observations [N,9] -> categorical distribution over 4 actions.

        sample() returns int64 [N]; log_prob(actions [N]) returns float [N].
        A single observation [9] gives scalar action/log-probability outputs.
        """
        return torch.distributions.Categorical(logits=self.policy(observations))

    def values(self, observations):
        """Float [N,9] -> predicted values [N], removing the final size-1 axis.

        A single observation [9] gives a scalar tensor [].
        """
        return self.value(observations).squeeze(-1)
