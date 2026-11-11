"""Provided helper: make training/evaluation plots and formatted summary table.

Example: python coding/plot_results.py --input results/core --output figures/core
Only complete run directories (config.json, training.csv, evaluation.csv) count.
Requires >=2 seeds/configuration to draw sample-SD bands; one seed is allowed
for a quick check but does not satisfy the assignment.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def label(c):
    names = {"reinforce": "REINFORCE", "baseline": "REINFORCE + baseline",
             "actor_critic": f"Actor-critic (n={c['n_step']})"}
    return names[c["method"]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    groups = defaultdict(list)
    shared = None
    for path in sorted(args.input.rglob("config.json")):
        root = path.parent
        if not (root / "evaluation.csv").exists():
            continue
        c = json.loads(path.read_text())
        settings = tuple(c[k] for k in ("steps", "gamma", "actor_lr", "critic_lr", "batch_episodes", "horizon", "eval_episodes"))
        if shared is not None and settings != shared:
            raise ValueError("Compare runs with the same core hyperparameters/budget")
        shared = settings
        train, ev = read_csv(root / "training.csv"), read_csv(root / "evaluation.csv")
        if not train or not ev:
            raise ValueError(f"Empty results: {root}")
        groups[label(c)].append((c, train, ev))
    if not groups:
        raise ValueError("No completed runs found")
    for name, runs in groups.items():
        seeds = [r[0]["seed"] for r in runs]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate training seeds for {name}")
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, runs in groups.items():
        # Smooth per run, then interpolate on a common step grid. No extrapolation.
        curves = []
        for c, train, ev in runs:
            steps = np.array([float(r["env_steps"]) for r in train])
            ret = np.array([float(r["return"]) for r in train])
            smooth = np.array([ret[max(0, i-49):i+1].mean() for i in range(len(ret))])
            curves.append((steps, smooth))
            rows.append([name, c["seed"], float(ret[-50:].mean()),
                         np.mean([float(r["return"]) for r in ev]), int(steps[-1])])
        lo = max(x[0] for x, _ in curves)
        hi = min(min(x[-1], c["steps"]) for (x, _), (c, _, _) in zip(curves, runs))
        if lo > hi:
            raise ValueError("Budget too short for a common plot interval")
        grid = np.linspace(lo, hi, 200)
        vals = np.stack([np.interp(grid, x, y) for x, y in curves])
        line, = ax.plot(grid, vals.mean(axis=0), label=name)
        if len(runs) > 1:
            sd = vals.std(axis=0, ddof=1)
            ax.fill_between(grid, vals.mean(axis=0)-sd, vals.mean(axis=0)+sd,
                            color=line.get_color(), alpha=.18)
    ax.set(xlabel="Training environment steps", ylabel="Undiscounted episode return",
           title="Trailing 50-episode means; across-seed mean +/- sample SD")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "training.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    table = []
    for i, name in enumerate(groups):
        subset = [r for r in rows if r[0] == name]
        means = [r[3] for r in subset]
        offsets = np.linspace(-.12, .12, len(means))
        ax.scatter(i+offsets, means, color="tab:blue")
        for x, r in zip(i+offsets, subset):
            ax.annotate(str(r[1]), (x, r[3]), xytext=(4, 4), textcoords="offset points")
            table.append([name, str(r[1]), f"{r[2]:.1f}", f"{r[3]:.1f}", str(r[4])])
        ax.scatter([i], [np.mean(means)], marker="_", s=220, color="black")
        fmt = lambda v: f"{np.mean(v):.1f} +/- {np.std(v, ddof=1):.1f}" if len(v)>1 else f"{v[0]:.1f} (one seed)"
        table.append([name, "mean +/- SD", fmt([r[2] for r in subset]), fmt(means), "--"])
    ax.set_xticks(range(len(groups)), list(groups), rotation=12, ha="right")
    ax.set(ylabel="Mean frozen-policy evaluation return",
           title="Each labeled point = one training seed; black bar = group mean")
    fig.tight_layout()
    fig.savefig(args.output / "evaluation.png", dpi=180)
    plt.close(fig)
    headers = ["Method", "Seed", "Last 50 train", "Evaluation", "Actual steps"]
    with (args.output / "summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(table)
    fig, ax = plt.subplots(figsize=(13, max(3, .4*(len(table)+1))))
    ax.axis("off")
    t = ax.table(cellText=table, colLabels=headers, loc="center", cellLoc="center",
                 colWidths=[.34, .16, .18, .18, .14])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.6)
    fig.tight_layout()
    fig.savefig(args.output / "summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Created training.png, evaluation.png, summary.png and summary.csv in {args.output}")


if __name__ == "__main__":
    main()
