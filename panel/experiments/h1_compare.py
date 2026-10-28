"""H1 on MiniGrid: PPO vs the static judge panel.

    python -m panel.experiments.h1_compare --env MiniGrid-DoorKey-5x5-v0 --seeds 3 \
        --ppo-steps 200000 --panel-steps 50000 --out runs/h1
"""
from __future__ import annotations

import argparse
import os

import torch

from panel.agent.ppo import train_ppo
from panel.analysis.compare import compare, format_table
from panel.analysis.plot import plot_dir
from panel.panel.trainer import train_panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MiniGrid-DoorKey-5x5-v0")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--ppo-steps", type=int, default=200_000)
    ap.add_argument("--panel-steps", type=int, default=50_000)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|mps; tiny nets can be faster on cpu")
    ap.add_argument("--out", default="runs/h1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ppo_logs, panel_logs = [], []
    for s in range(args.seeds):
        model, log = train_ppo(args.env, args.ppo_steps, seed=s, eval_episodes=args.eval_episodes, device=args.device)
        log.save(os.path.join(args.out, f"ppo_s{s}.json")); ppo_logs.append(log)
        torch.save(model.state_dict(), os.path.join(args.out, f"ppo_s{s}.pt"))
        model, judges, log = train_panel(args.env, args.panel_steps, seed=s, eval_episodes=args.eval_episodes,
                                      device=args.device)
        log.save(os.path.join(args.out, f"panel_s{s}.json")); panel_logs.append(log)
        torch.save({"policy": model.state_dict(), "judges": [m.state_dict() for m in judges.models],
                    "weights": judges.weights}, os.path.join(args.out, f"panel_s{s}.pt"))
        print(f"seed {s}: ppo final {ppo_logs[-1].rows[-1]['eval_return']:.3f} "
              f"({ppo_logs[-1].rows[-1]['wall_s']:.0f}s), panel final {log.rows[-1]['eval_return']:.3f} "
              f"({log.rows[-1]['wall_s']:.0f}s)", flush=True)
    res = compare(ppo_logs, panel_logs)
    table = format_table(res)
    print(table)
    with open(os.path.join(args.out, "h1_summary.txt"), "w") as f:
        f.write(table + "\n")
    print("plot:", plot_dir(args.out))


if __name__ == "__main__":
    main()
