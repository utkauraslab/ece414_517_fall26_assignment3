"""Student TODOs. See the handout for equations and tensor shapes.

All tensors are CPU float32 unless indicated. Never convert a differentiable
loss to NumPy or to a Python number. Edit only the two marked C4 blocks in train.py; keep other helpers unchanged.
"""
import torch


def returns_to_go(rewards, gamma):
    """Core C1: discounted returns for ONE complete episode.

    Args:
        rewards: floating tensor [T]; rewards[t] is R_(t+1), received after
            taking action A_t in state S_t. T is the number of transitions.
            Entries are in time order, including the final reward.
        gamma: Python float discount factor (default 0.99).
    Returns:
        Floating tensor [T]; entry t is G_t, the discounted return starting
        with rewards[t]. Keep chronological order and input dtype/device.
        No baseline subtraction or normalization. No gradient is needed.
    Do not modify rewards in place or combine different episodes here."""
    raise NotImplementedError("C1: returns_to_go")


def policy_loss(log_probs, advantages, discounts, episode_count):
    """Core C1: actor loss to MINIMIZE.

    Args:
        log_probs: floating tensor [N]; log probability of the action
            actually sampled at each transition, NOT all four action logits.
        advantages: floating tensor [N]; actor update weights: G for
            REINFORCE, G-V for baseline, or y-V for actor-critic.
        discounts: floating tensor [N]; gamma**t at each transition, with
            t restarting at zero for each episode. Supplied by train.py.
        episode_count: positive Python int B, number of complete episodes.
            N is the sum of their lengths, not the number of episodes.
    Returns:
        Floating scalar tensor, shape [] (ndim=0), NOT [1] or a Python float.
        Use the actor-loss equation in the handout; divide by B, not N.
        Keep gradients through log_probs only; detach the update weights.
    All three vectors must be aligned in the same transition order.
    Do not add a second gamma**t factor or normalize the update weights."""
    raise NotImplementedError("C1: policy_loss")


def baseline_advantages(returns, values):
    """Core C2: return-minus-baseline actor update weights.

    Algorithm table M2, middle column: G-V is the actor update weight.
    train.py passes this output to policy_loss, NOT as a value-loss target.
    The baseline itself is trained separately by value_loss(values, returns).

    Args:
        returns: floating tensor [N]; sampled returns G at each transition.
        values: floating tensor [N]; predicted V(S) at the SAME states,
            evaluated before the current network updates.
    Returns:
        Detached floating tensor [N] containing G-V in input order.
        No averaging or normalization; preserve input dtype/device.
    Both vectors refer to the same transitions. Do not mutate either input."""
    raise NotImplementedError("C2: baseline_advantages")


def value_loss(values, targets):
    """Core C2/C3: mean squared error for the value network.

    Algorithm-table value-target column and the L_v equation:
    M2 uses G as targets; M3 uses TD y as targets. M1 does not call this.
    This scalar trains the value network, not the policy. Do not pass
    G-V or y-V actor update weights as the regression targets.

    Args:
        values: floating tensor [N]; current predicted V(S) for each state.
        targets: floating tensor [N]; fixed G targets for the Monte Carlo
            baseline, or fixed TD targets y for actor-critic.
    Returns:
        Floating scalar tensor [] (ndim=0), NOT [1] or a Python number.
        Average squared errors over N transitions, without a factor 1/2.
        Gradients reach values, not targets (use targets.detach()).
    The vectors have matching entries and no trailing dimension [N,1]."""
    raise NotImplementedError("C2: value_loss")


def td_targets(rewards, next_values, ended, gamma):
    """Core C3: one-step TD value targets.

    Algorithm table M3, last column: construct the value target y.
    train.py sends y both to value_loss (critic training) and td_advantages
    (to prepare the actor weight). This output is not yet y-V(S).

    Args:
        rewards: floating tensor [N]; reward after each sampled action.
        next_values: floating tensor [N]; pre-update V(S_next) for each
            transition, NOT the values of its current state.
        ended: Boolean tensor [N]; True if that transition ends the task
            by physical termination OR the assignment's finite horizon.
        gamma: Python float discount factor.
    Returns:
        Detached floating tensor [N] with one target per transition.
        Use reward only where ended is True; otherwise include the
        discounted next-state value. Preserve reward dtype/device.
    Entries of all vectors describe the same transitions. Do not use the
    next episode's reset state as the successor of a terminal transition."""
    raise NotImplementedError("C3: td_targets")


def td_advantages(targets, values):
    """Core C3: TD target-minus-current-value actor update weights.

    Algorithm table M3, middle column: subtract CURRENT-state V(S)
    from the TD target y. train.py passes this output to policy_loss.
    td_targets uses NEXT-state values to construct y; this helper instead
    prepares an actor weight. The critic is trained separately by value_loss.

    Args:
        targets: floating tensor [N]; one-step or n-step TD targets y.
        values: floating tensor [N]; pre-update V(S) at the current states
            in the SAME transition order (not next-state predictions).
    Returns:
        Detached floating tensor [N] containing y-V(S), not a scalar mean.
        Preserve input dtype/device; no normalization or input mutation."""
    raise NotImplementedError("C3: td_advantages")


def n_step_targets(rewards, values, gamma, n):
    """Required ECE517 extension ONLY: targets for one complete episode.

    Args:
        rewards: floating tensor [T]; rewards[t] is R_(t+1).
        values: floating tensor [T+1]; values[t] is pre-update V(S_t).
            The supplied last entry values[T] is zero at the task ending.
        gamma: Python float discount factor.
        n: positive Python int; requested number of reward steps.
    Returns:
        Detached floating tensor [T]; entry t is the n-step target for S_t.
        Follow the handout equation, shortening the reward sum near the
        end of the episode. Preserve rewards' dtype/device.
    T is this episode's length, not the batch length N. Do not cross an
    episode boundary or modify inputs. No batch averaging is performed."""
    raise NotImplementedError("ECE517: n_step_targets")

