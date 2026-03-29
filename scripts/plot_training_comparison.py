"""
Overlay PPO, SAC, and random-baseline performance against training env steps.

Expected PPO/SAC log lines:
    Step       256 | Ep      0 | Return    40.28 | Steps   16 | Avg100 R=  40.28 L= 16.0

Expected random-baseline sources:
    1. logs/random_baseline.npy from scripts/random_baseline.py, or
    2. a text log containing:
           Mean   : <value>
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


TRAIN_LINE_RE = re.compile(
    r"Step\s+(\d+)\s+\|\s+Ep\s+\d+\s+\|\s+Return\s+[-+]?\d*\.?\d+"
    r"\s+\|\s+Steps\s+\d+\s+\|\s+Avg100 R=\s*([-+]?\d*\.?\d+)"
)
RANDOM_MEAN_RE = re.compile(r"Mean\s*:\s*([-+]?\d*\.?\d+)")


def parse_training_log(path: str) -> tuple[np.ndarray, np.ndarray]:
    steps: list[int] = []
    avg_returns: list[float] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = TRAIN_LINE_RE.search(line)
            if match is None:
                continue
            steps.append(int(match.group(1)))
            avg_returns.append(float(match.group(2)))

    if not steps:
        raise ValueError(f"No 'Avg100 R=' training lines found in {path}.")

    return np.asarray(steps, dtype=np.int64), np.asarray(avg_returns, dtype=np.float64)


def parse_random_mean_from_log(path: str) -> float:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    match = RANDOM_MEAN_RE.search(text)
    if match is None:
        raise ValueError(f"Could not find random baseline mean in {path}.")
    return float(match.group(1))


def load_random_mean(random_npy: str | None, random_log: str | None) -> float:
    if random_npy:
        values = np.load(random_npy)
        return float(np.mean(values))
    if random_log:
        return parse_random_mean_from_log(random_log)
    raise ValueError("Provide either --random-npy or --random-log.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PPO, SAC, and random mean return vs training env steps."
    )
    parser.add_argument("--ppo-log", required=True, help="Path to PPO training stdout log.")
    parser.add_argument("--sac-log", required=True, help="Path to SAC training stdout log.")
    parser.add_argument(
        "--random-npy",
        default=None,
        help="Path to logs/random_baseline.npy saved by scripts/random_baseline.py.",
    )
    parser.add_argument(
        "--random-log",
        default=None,
        help="Path to a random baseline text log containing a printed Mean value.",
    )
    parser.add_argument(
        "--output",
        default="results/plots/onpolicy_offpolicy_random_return_vs_steps.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--title",
        default="Mean Return vs Training Environment Steps",
        help="Plot title.",
    )
    args = parser.parse_args()

    if not args.random_npy and not args.random_log:
        parser.error("Provide either --random-npy or --random-log.")

    ppo_steps, ppo_returns = parse_training_log(args.ppo_log)
    sac_steps, sac_returns = parse_training_log(args.sac_log)
    sac_max_step = sac_steps.max()
    ppo_mask = ppo_steps <= sac_max_step
    ppo_steps = ppo_steps[ppo_mask]
    ppo_returns = ppo_returns[ppo_mask]

    random_mean = load_random_mean(args.random_npy, args.random_log)

    max_step = max(float(ppo_steps.max()), float(sac_steps.max()))

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(ppo_steps, ppo_returns, color="#2E7D32", linewidth=2, label="PPO (on-policy)")
    plt.plot(sac_steps, sac_returns, color="#1565C0", linewidth=2, label="SAC (off-policy)")
    plt.hlines(
        random_mean,
        xmin=0,
        xmax=max_step,
        color="#616161",
        linestyle="--",
        linewidth=2,
        label=f"Random baseline (mean={random_mean:.2f})",
    )

    ax = plt.gca()
    ax.set_xlim(0, sac_max_step)
    ax.set_xticks(np.linspace(0, sac_max_step, 5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1e6:.1f}"))


    plt.xlabel("Training environment steps (millions)")
    plt.ylabel("Mean return (Avg100 R)")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)

    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
