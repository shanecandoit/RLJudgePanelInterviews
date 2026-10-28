"""Plot PPO vs panel learning curves from saved run logs.

    python -m panel.analysis.plot runs/h1            # writes runs/h1/h1_curves.png
"""
from __future__ import annotations

import glob
import os
import sys
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from panel.analysis.compare import asymptote
from panel.analysis.runlog import RunLog

AXES = (("env_steps", "environment samples"), ("iteration", "iteration (collect/query + update)"),
        ("wall_s", "training wall time (s)"))
COLORS = {"ppo": "#1f77b4", "panel": "#d95f02"}


def _band(logs: List[RunLog], x: str):
    """Mean +/- std of eval_return over seeds on a shared x grid (step interpolation)."""
    hi = min(l.column(x)[-1] for l in logs)
    grid = np.linspace(0, hi, 200)
    ys = np.stack([np.interp(grid, l.column(x), l.column("eval_return")) for l in logs])
    return grid, ys.mean(0), ys.std(0)


def plot_dir(run_dir: str, frac: float = 0.8) -> str:
    logs = {a: [RunLog.load(p) for p in sorted(glob.glob(os.path.join(run_dir, f"{a}_s*.json")))]
            for a in ("ppo", "panel")}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    thr = frac * np.mean([asymptote(l) for l in logs["ppo"]]) if logs["ppo"] else None
    for ax, (x, label) in zip(axes, AXES):
        for algo, ls in logs.items():
            if not ls:
                continue
            g, m, s = _band(ls, x)
            ax.plot(g, m, color=COLORS[algo], label=f"{algo} (n={len(ls)})")
            ax.fill_between(g, m - s, m + s, color=COLORS[algo], alpha=0.2)
        if thr is not None:
            ax.axhline(thr, color="gray", ls="--", lw=1, label=f"{frac:.0%} of PPO asymptote")
        ax.set_xlabel(label); ax.set_ylabel("eval return"); ax.grid(alpha=0.3)
    axes[0].legend()
    out = os.path.join(run_dir, "h1_curves.png")
    fig.savefig(out, dpi=150)
    return out


if __name__ == "__main__":
    print(plot_dir(sys.argv[1]))
