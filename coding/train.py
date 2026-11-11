"""Provided CPU training/evaluation program. Run from the package root.

Complete algorithms.py and the two C4 loss-call TODOs in update below.
Keep the remaining training code unchanged. Outputs are never overwritten.
The finite 1000-step horizon is part of the task; remaining time is observed.
"""
import argparse
import csv
import json
import platform
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import algorithms as alg
from models import Agent

HORIZON = 1000


def observation(raw, t):
    return np.append(raw, (HORIZON - t) / HORIZON).astype(np.float32)


def episode(env, agent, reset_seed):
    """Collect one episode of length T and return aligned tensors.

    obs/next_obs: float32 [T,9]; actions: int64 [T], entries 0..3;
    rewards: float32 [T]; ended: bool [T]. Row t describes S_t, A_t,
    R_(t+1), S_(t+1), and whether that transition ends the task.
    """
    raw, _ = env.reset(seed=reset_seed)
    obs = observation(raw, 0)
    data = {k: [] for k in ("obs", "next_obs", "actions", "rewards", "ended")}
    for t in range(HORIZON):
        with torch.no_grad():
            action = agent.distribution(torch.tensor(obs)).sample().item()
        raw, reward, terminated, truncated, _ = env.step(action)
        nxt = observation(raw, t + 1)
        ended = terminated or truncated or t + 1 == HORIZON
        for key, value in zip(data, (obs, nxt, action, reward, ended)):
            data[key].append(value)
        obs = nxt
        if ended:
            break
    return {k: torch.tensor(np.array(v), dtype=(torch.long if k == "actions"
            else torch.bool if k == "ended" else torch.float32))
            for k, v in data.items()}


def update(agent, actor_opt, critic_opt, episodes, args):
    """Update networks from B episode dictionaries; N=sum of their lengths.

    Concatenation yields obs [N,9], logits [N,4], actions [N], and aligned
    value/target/log-probability/discount/update-weight vectors [N].
    Updates model parameters in place; returns None.
    """
    obs = torch.cat([e["obs"] for e in episodes])
    actions = torch.cat([e["actions"] for e in episodes])
    discounts = torch.cat([args.gamma ** torch.arange(len(e["rewards"]))
                           for e in episodes])
    # STEP 1: Prepare fixed targets and actor weights before either update.
    with torch.no_grad():
        values = agent.values(obs)
        if args.method in ("reinforce", "baseline"):
            targets = torch.cat([alg.returns_to_go(e["rewards"], args.gamma)
                                 for e in episodes])
            advantages = (targets if args.method == "reinforce" else
                          alg.baseline_advantages(targets, values))
        elif args.n_step == 1:
            targets = alg.td_targets(torch.cat([e["rewards"] for e in episodes]),
                agent.values(torch.cat([e["next_obs"] for e in episodes])),
                torch.cat([e["ended"] for e in episodes]), args.gamma)
            advantages = alg.td_advantages(targets, values)
        else:
            targets = torch.cat([alg.n_step_targets(e["rewards"],
                torch.cat([agent.values(e["obs"]), torch.zeros(1)]),
                args.gamma, args.n_step) for e in episodes])
            advantages = alg.td_advantages(targets, values)
    # STEP 2: Actor update (all three methods).
    # obs [N,9] -> logits [N,4] -> sampled-action log_probs [N].
    # advantages and discounts are fixed aligned [N] vectors; B=len(episodes).
    logits = agent.policy(obs)
    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions)
    # TODO C4-A: assign scalar loss using alg.policy_loss and the variables above.
    raise NotImplementedError("C4-A: connect actor loss in train.update")
    actor_opt.zero_grad()       # Clear gradients from the preceding batch.
    loss.backward()            # Autograd differentiates the policy loss.
    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 5.0)
    actor_opt.step()            # Adam changes only the policy parameters.
    # STEP 3: Value-network update (M2/M3 only; M1 skips this block).
    if args.method != "reinforce":
        # targets [N] stay fixed: G for M2, TD y for M3.
        # Recompute predictions each step, since the value weights change.
        for _ in range(5):
            critic_opt.zero_grad()
            current_values = agent.values(obs)  # [N], WITH a gradient graph
            # TODO C4-B: assign scalar critic_loss using alg.value_loss.
            # Choose the target from STEP 1, not the actor update weights.
            raise NotImplementedError("C4-B: connect value loss in train.update")
            critic_loss.backward()  # Autograd differentiates squared error.
            torch.nn.utils.clip_grad_norm_(agent.value.parameters(), 5.0)
            critic_opt.step()       # Adam changes only the value parameters.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["reinforce", "baseline", "actor_critic"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--batch-episodes", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--n-step", type=int, default=1)
    args = parser.parse_args()
    if min(args.steps, args.eval_episodes, args.batch_episodes, args.n_step) < 1:
        parser.error("Counts must be positive")
    if not 0 < args.gamma <= 1 or min(args.actor_lr, args.critic_lr) <= 0:
        parser.error("Require 0 < gamma <= 1, positive learning rates")
    if args.n_step != 1 and args.method != "actor_critic":
        parser.error("--n-step applies only to actor_critic")
    args.output.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy()
    config["output"] = str(config["output"])
    config.update(python=platform.python_version(), torch=torch.__version__,
                  numpy=np.__version__, gymnasium=gym.__version__, horizon=HORIZON)
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    agent = Agent()
    actor_opt = torch.optim.Adam(agent.policy.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(agent.value.parameters(), lr=args.critic_lr)
    env = gym.make("LunarLander-v3", continuous=False, enable_wind=False,
                   max_episode_steps=HORIZON)
    steps = count = batch = 0
    try:
        with (args.output / "training.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["episode", "env_steps", "batch",
                "return", "length", "seed"])
            writer.writeheader()
            while steps < args.steps:
                episodes = []
                batch += 1
                for _ in range(args.batch_episodes):
                    e = episode(env, agent, args.seed * 1000000 + count)
                    episodes.append(e)
                    count += 1
                    length = len(e["rewards"])
                    steps += length
                    writer.writerow(dict(episode=count, env_steps=steps, batch=batch,
                        **{"return": e["rewards"].sum().item()}, length=length, seed=args.seed))
                update(agent, actor_opt, critic_opt, episodes, args)
                file.flush()
                if batch % 10 == 0:
                    print(f"batch={batch} steps={steps} episodes={count}", flush=True)
        torch.save(agent.state_dict(), args.output / "agent.pt")
        # Sample from the frozen stochastic policy, not argmax; no optimizer updates.
        torch.manual_seed(args.seed + 200000)
        with (args.output / "evaluation.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["episode", "return", "length",
                                                     "train_seed", "reset_seed"])
            writer.writeheader()
            for i in range(args.eval_episodes):
                reset_seed = args.seed * 1000000 + 500000 + i
                e = episode(env, agent, reset_seed)
                writer.writerow(dict(episode=i+1, **{"return": e["rewards"].sum().item()},
                    length=len(e["rewards"]), train_seed=args.seed, reset_seed=reset_seed))
        print(f"Completed: {args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
