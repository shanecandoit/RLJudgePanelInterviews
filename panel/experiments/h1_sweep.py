"""Random-search sweep of the panel agent for H1 (sample efficiency), with judge-diagnostic logging.

    python -m panel.experiments.h1_sweep --envs MiniGrid-Empty-5x5-v0 --runs 20 --out runs/sweep_pilot
    python -m panel.experiments.h1_sweep --runs 40 --workers 5 --out runs/sweep \
        --envs MiniGrid-Empty-5x5-v0 MiniGrid-Empty-8x8-v0 MiniGrid-DoorKey-5x5-v0

Each run draws one value per factor. Values are balanced (every value used about equally often per env) and shuffled,
so a few dozen runs give usable marginal tables. One JSON line per finished run goes to <out>/results.jsonl; rerunning
with the same arguments skips finished runs. Tables and plots: panel.analysis.sweep_report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# allow `python panel/experiments/h1_sweep.py` (and spawned workers) as well as `python -m ...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

import numpy as np

SPACE: Dict[str, list] = {
    "ent_coef": [0.003, 0.01, 0.03, 0.1],
    "imag_horizon": [1, 2, 4, 8],
    "n_judges": [2, 4, 8, 16],
    "imag_frac": [1.0, 0.9, 0.8, 0.5],  # share of imagined samples; < 1 adds 8x4 real rollouts per round
}
DEFAULT_STEPS = {"MiniGrid-Empty-5x5-v0": 4_000, "MiniGrid-Empty-8x8-v0": 8_000, "MiniGrid-DoorKey-5x5-v0": 20_000}


def design(env: str, n: int, design_seed: int) -> List[dict]:
    """n configs, each factor's values tiled to n and shuffled independently (balanced random search)."""
    cols = {}
    for f, vals in SPACE.items():
        rng = np.random.default_rng([design_seed, sum(map(ord, env)), sum(map(ord, f))])
        col = (list(vals) * (n // len(vals) + 1))[:n]
        cols[f] = [col[i] for i in rng.permutation(n)]
    short = env.replace("MiniGrid-", "").replace("-v0", "")
    return [{"id": f"{short}_{i:03d}", "env": env, "seed": i, "config": {f: cols[f][i] for f in SPACE}} for i in range(n)]


def make_cfg(config: dict):
    from panel.agent.ppo import PPOConfig
    from panel.panel.trainer import PanelConfig
    real = config["imag_frac"] < 1.0
    return PanelConfig(n_judges=config["n_judges"], imag_horizon=config["imag_horizon"], imag_frac=config["imag_frac"],
                       real_starts=8 if real else 0, real_horizon=4,
                       ppo=PPOConfig(epochs=2, ent_coef=config["ent_coef"]))


def run_one(job: dict) -> dict:
    import torch
    from panel.panel.trainer import train_panel
    torch.set_num_threads(1)
    try:
        _, _, log = train_panel(job["env"], job["steps"], seed=job["seed"], cfg=make_cfg(job["config"]),
                                eval_episodes=job["eval_episodes"], device="cpu")
        return {"id": job["id"], "env": job["env"], "seed": job["seed"], "config": job["config"], "rows": log.rows}
    except Exception as e:  # keep the sweep going; report counts failures
        return {"id": job["id"], "env": job["env"], "seed": job["seed"], "config": job["config"], "error": repr(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["MiniGrid-Empty-5x5-v0"])
    ap.add_argument("--runs", type=int, default=20, help="runs per env")
    ap.add_argument("--steps", nargs="*", default=[], help="ENV=samples overrides")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--design-seed", type=int, default=0)
    ap.add_argument("--ref", action="append", default=[], help="ENV=ppo_asymptote for the report, repeatable")
    ap.add_argument("--out", default="runs/sweep")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.jsonl")
    steps = {**DEFAULT_STEPS, **{k: int(v) for k, v in (s.split("=") for s in a.steps)}}
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    jobs = [{**j, "steps": steps[j["env"]], "eval_episodes": a.eval_episodes}
            for env in a.envs for j in design(env, a.runs, a.design_seed) if j["id"] not in done]
    print(f"{len(jobs)} runs to do ({len(done)} already finished), {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex, open(path, "a") as out:
        for k, fut in enumerate(as_completed([ex.submit(run_one, j) for j in jobs]), 1):
            rec = fut.result()
            out.write(json.dumps(rec) + "\n"); out.flush()
            last = rec["rows"][-1]["eval_return"] if "rows" in rec else rec["error"]
            print(f"[{k}/{len(jobs)}] {rec['id']} {rec['config']} -> {last}", flush=True)
    from panel.analysis.sweep_report import report
    print(report(a.out, {k: float(v) for k, v in (s.split("=") for s in a.ref)}))


if __name__ == "__main__":
    main()
