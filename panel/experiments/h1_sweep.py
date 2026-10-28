"""Overnight sweep for H1 (sample efficiency): PPO baselines, random search, and seed replicates.

    # pilot
    python panel/experiments/h1_sweep.py --envs MiniGrid-Empty-5x5-v0 --runs 20 --out runs/sweep_pilot
    # nightly: PPO reference per env + 30 random configs per env + 5-seed replicates of 3 named configs, stop starting
    # new runs after 8 hours (runs in flight finish). Safe to re-run: finished runs are skipped.
    python panel/experiments/h1_sweep.py --runs 30 --replicate base h8_j2 h2_j16 --seeds 1 2 3 4 5 \
        --workers 5 --hours 8 --out runs/nightly

Random runs draw one value per factor, balanced and shuffled, so a few dozen runs give usable marginal tables.
Replicates run named configs (PRESETS) over several seeds to measure run-to-run noise. PPO jobs give each env's
asymptote, which the report uses for the H1 threshold. One JSON line per finished run goes to <out>/results.jsonl.
Report: python -m panel.analysis.sweep_report <out>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# allow `python panel/experiments/h1_sweep.py` (and spawned workers) as well as `python -m ...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from itertools import zip_longest
from typing import Dict, List

import numpy as np

SPACE: Dict[str, list] = {
    "ent_coef": [0.003, 0.01, 0.03, 0.1],
    "imag_horizon": [1, 2, 4, 8],
    "n_judges": [2, 4, 8, 16],
    "imag_frac": [1.0, 0.9, 0.8, 0.5],  # share of imagined samples; < 1 adds 8x4 real rollouts per round
}
PRESETS: Dict[str, dict] = {
    "base":   dict(ent_coef=0.01, imag_horizon=2, n_judges=4, imag_frac=1.0),
    "h8_j2":  dict(ent_coef=0.01, imag_horizon=8, n_judges=2, imag_frac=1.0),    # best run of the Empty-5x5 pilot
    "h2_j16": dict(ent_coef=0.003, imag_horizon=2, n_judges=16, imag_frac=1.0),  # 2nd best, ~6x the cost
    "mix80":  dict(ent_coef=0.01, imag_horizon=2, n_judges=4, imag_frac=0.8),
}
# env -> (panel samples, PPO samples). Panel runs cost ~40 ms/sample single-threaded (more with 16 judges).
ENVS: Dict[str, tuple] = {
    "MiniGrid-Empty-5x5-v0": (4_000, 40_000),
    "MiniGrid-Empty-Random-5x5-v0": (6_000, 60_000),   # random start cell and heading: needs generalisation
    "MiniGrid-Empty-8x8-v0": (8_000, 100_000),
    "MiniGrid-LavaGapS5-v0": (10_000, 150_000),        # sparse reward, one hazard to route around
    "MiniGrid-DoorKey-5x5-v0": (20_000, 300_000),      # sparse reward, key -> door -> goal
    "MiniGrid-DoorKey-6x6-v0": (40_000, 500_000),      # opt-in via --envs
    "MiniGrid-SimpleCrossingS9N1-v0": (20_000, 300_000),
}
DEFAULT_ENVS = ["MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-Random-5x5-v0", "MiniGrid-Empty-8x8-v0",
                "MiniGrid-LavaGapS5-v0", "MiniGrid-DoorKey-5x5-v0"]


def short_name(env: str) -> str:
    return env.replace("MiniGrid-", "").replace("-v0", "")


def design(env: str, n: int, design_seed: int) -> List[dict]:
    """n configs, each factor's values tiled to n and shuffled independently (balanced random search)."""
    cols = {}
    for f, vals in SPACE.items():
        rng = np.random.default_rng([design_seed, sum(map(ord, env)), sum(map(ord, f))])
        col = (list(vals) * (n // len(vals) + 1))[:n]
        cols[f] = [col[i] for i in rng.permutation(n)]
    return [{"id": f"{short_name(env)}_{i:03d}", "kind": "panel", "group": "random", "env": env, "seed": i,
             "config": {f: cols[f][i] for f in SPACE}} for i in range(n)]


def replicates(env: str, names: List[str], seeds: List[int]) -> List[dict]:
    return [{"id": f"{short_name(env)}_{n}_s{s}", "kind": "panel", "group": n, "env": env, "seed": s,
             "config": dict(PRESETS[n])} for s in seeds for n in names]


def ppo_jobs(env: str, n_seeds: int) -> List[dict]:
    return [{"id": f"ppo_{short_name(env)}_s{s}", "kind": "ppo", "group": "ppo", "env": env, "seed": s, "config": {}}
            for s in range(n_seeds)]


def build_jobs(envs, runs, design_seed, names, seeds, ppo_seeds, steps, eval_episodes) -> List[dict]:
    """PPO baselines first (they set the thresholds), then everything else round-robin across envs so that a
    time cut-off leaves every env equally covered."""
    first = [j for e in envs for j in ppo_jobs(e, ppo_seeds)]
    per_env = [(replicates(e, names, seeds) if names else []) + design(e, runs, design_seed) for e in envs]
    rest = [j for tier in zip_longest(*per_env) for j in tier if j]
    for j in first + rest:
        j["steps"] = steps[j["env"]][1 if j["kind"] == "ppo" else 0]
        j["eval_episodes"] = eval_episodes
    return first + rest


def make_cfg(config: dict):
    from panel.agent.ppo import PPOConfig
    from panel.panel.trainer import PanelConfig
    real = config["imag_frac"] < 1.0
    return PanelConfig(n_judges=config["n_judges"], imag_horizon=config["imag_horizon"], imag_frac=config["imag_frac"],
                       real_starts=8 if real else 0, real_horizon=4,
                       ppo=PPOConfig(epochs=2, ent_coef=config["ent_coef"]))


def run_one(job: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    meta = {k: job[k] for k in ("id", "kind", "group", "env", "seed", "config", "steps")}
    try:
        if job["kind"] == "ppo":
            from panel.agent.ppo import train_ppo
            _, log = train_ppo(job["env"], job["steps"], seed=job["seed"], eval_episodes=job["eval_episodes"], device="cpu")
        else:
            from panel.panel.trainer import train_panel
            _, _, log = train_panel(job["env"], job["steps"], seed=job["seed"], cfg=make_cfg(job["config"]),
                                    eval_episodes=job["eval_episodes"], device="cpu")
        return {**meta, "rows": log.rows}
    except Exception as e:  # keep the sweep going; the report counts failures
        return {**meta, "error": repr(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=DEFAULT_ENVS, help=f"any of: {' '.join(ENVS)}")
    ap.add_argument("--runs", type=int, default=20, help="random-search runs per env (0 to skip)")
    ap.add_argument("--replicate", nargs="*", default=[], choices=list(PRESETS), help="named configs to replicate")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3, 4, 5], help="seeds for --replicate")
    ap.add_argument("--ppo-seeds", type=int, default=2, help="PPO baseline runs per env (0 to skip)")
    ap.add_argument("--steps", nargs="*", default=[], help="ENV=panel_samples overrides")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hours", type=float, default=0, help="stop starting new runs after this long (0 = no limit)")
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--design-seed", type=int, default=0)
    ap.add_argument("--ref", action="append", default=[], help="ENV=ppo_asymptote override for the report")
    ap.add_argument("--out", default="runs/sweep")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "results.jsonl")
    steps = dict(ENVS)
    for k, v in (s.split("=") for s in a.steps):
        steps[k] = (int(v), steps[k][1])
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {json.loads(l)["id"] for l in f if l.strip()}
    jobs = [j for j in build_jobs(a.envs, a.runs, a.design_seed, a.replicate, a.seeds, a.ppo_seeds, steps,
                                  a.eval_episodes) if j["id"] not in done]
    deadline = time.time() + a.hours * 3600 if a.hours else None
    print(f"{len(jobs)} runs queued ({len(done)} already finished), {a.workers} workers"
          + (f", no new runs after {a.hours} h" if deadline else ""), flush=True)
    queue, pending, finished = iter(jobs), set(), 0
    with ProcessPoolExecutor(a.workers) as ex, open(path, "a") as out:
        while True:
            while len(pending) < a.workers * 2 and (deadline is None or time.time() < deadline):
                j = next(queue, None)
                if j is None:
                    break
                pending.add(ex.submit(run_one, j))
            if not pending:
                break
            ready, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in ready:
                rec = fut.result()
                finished += 1
                out.write(json.dumps(rec) + "\n"); out.flush()
                last = f"{rec['rows'][-1]['eval_return']:.3f}" if "rows" in rec else rec["error"]
                print(f"[{finished}/{len(jobs)}] {rec['id']} {rec['config']} -> {last}", flush=True)
    if deadline and time.time() >= deadline:
        print("time limit reached; unstarted runs remain (re-run the same command to continue)", flush=True)
    from panel.analysis.sweep_report import report
    print(report(a.out, {k: float(v) for k, v in (s.split("=") for s in a.ref)}))


if __name__ == "__main__":
    main()
