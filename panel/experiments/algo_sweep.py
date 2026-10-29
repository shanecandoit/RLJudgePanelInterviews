"""Resumable algorithm x environment x judge-count experiment.

ppo = panel PPO; dqn = panel acquisition + real-only Double DQN updates;
dyna = identical panel DQN plus one-step imagined updates.
Real-only PPO and DQN baselines are included once per environment/seed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from panel.agent.dqn import DQNConfig
from panel.panel.trainer import PanelConfig
from panel.agent.ppo import PPOConfig

ENVS = {'empty5x5': ('MiniGrid-Empty-5x5-v0', 40_000),
        'lavagap': ('MiniGrid-LavaGapS5-v0', 150_000),
        'doorkey': ('MiniGrid-DoorKey-5x5-v0', 300_000)}
VERSION = 1


def build_jobs(algos, envs, judges, seeds, steps=None, eval_episodes=10, baselines=True):
    jobs = []
    # Interleave seeds/environments so a time limit does not exhaust one task first.
    variants = [(a, j, 'panel') for j in judges for a in algos]
    if baselines:
        variants = [('ppo', 0, 'baseline'), ('dqn', 0, 'baseline')] + variants
    for seed in seeds:
        for algo, count, variant in variants:
            for alias in envs:
                env, budget = ENVS[alias]
                panel_cfg = PanelConfig(n_judges=max(1, count), imag_horizon=1, imag_frac=.5,
                                        real_starts=8, real_horizon=4, judge_steps=25, eval_interval=1024)
                cfg = asdict(panel_cfg) if algo == 'ppo' and count else (
                    asdict(PPOConfig()) if algo == 'ppo' else asdict(DQNConfig()))
                spec = dict(version=VERSION, algo=algo, variant=variant, n_judges=count,
                            env=env, seed=seed, steps=steps or budget,
                            eval_episodes=eval_episodes, config=cfg)
                digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:10]
                jobs.append(dict(spec, id=f'{alias}_{algo}_{variant}_j{count}_s{seed}_{digest}'))
    return jobs


def run_one(job):
    import torch
    torch.set_num_threads(1)
    started = time.time()
    try:
        if job['algo'] == 'ppo':
            if job['variant'] == 'baseline':
                from panel.agent.ppo import train_ppo
                _, log = train_ppo(job['env'], job['steps'], seed=job['seed'],
                                   cfg=PPOConfig(**job['config']), eval_episodes=job['eval_episodes'])
            else:
                from panel.panel.trainer import train_panel
                cfg = dict(job['config'])
                cfg['ppo'] = PPOConfig(**cfg['ppo'])
                _, _, log = train_panel(job['env'], job['steps'], seed=job['seed'],
                                        cfg=PanelConfig(**cfg), eval_episodes=job['eval_episodes'])
        else:
            from panel.agent.dqn import train_dqn
            _, _, log = train_dqn(job['env'], job['steps'], seed=job['seed'],
                                  n_judges=job['n_judges'], dyna=job['algo'] == 'dyna',
                                  cfg=DQNConfig(**job['config']), eval_episodes=job['eval_episodes'])
        return dict(job, rows=log.rows, elapsed_s=time.time() - started)
    except Exception:
        return dict(job, error=traceback.format_exc(), elapsed_s=time.time() - started)


def load_records(path):
    if not Path(path).exists():
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def successful_ids(records):
    return {r['id'] for r in records if r.get('rows') and 'error' not in r}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--algos', nargs='+', choices=['ppo', 'dqn', 'dyna'], default=['ppo', 'dqn', 'dyna'])
    ap.add_argument('--envs', nargs='+', choices=list(ENVS), default=list(ENVS))
    ap.add_argument('--judges', nargs='+', type=int, default=[1, 2, 4])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--workers', type=int, default=5)
    ap.add_argument('--hours', type=float, default=8, help='stop starting jobs after this many hours; 0 = unlimited')
    ap.add_argument('--steps', type=int, help='override all per-environment budgets (useful for smoke tests)')
    ap.add_argument('--eval-episodes', type=int, default=10)
    ap.add_argument('--out', default='runs/algo_sweep')
    ap.add_argument('--no-baselines', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if min(args.judges) < 1 or args.workers < 1 or args.eval_episodes < 1 or args.hours < 0:
        ap.error('judges, workers and evaluation episodes must be positive; hours must be nonnegative')
    if args.steps is not None and args.steps < 1:
        ap.error('--steps must be positive')
    for values in (args.algos, args.envs, args.judges, args.seeds):
        if len(set(values)) != len(values):
            ap.error('duplicate axis values are not allowed')
    jobs = build_jobs(args.algos, args.envs, args.judges, args.seeds, args.steps,
                      args.eval_episodes, not args.no_baselines)
    out = Path(args.out)
    records = load_records(out / 'results.jsonl')
    done = successful_ids(records)
    remaining = [j for j in jobs if j['id'] not in done]
    print(f'{len(jobs)} planned; {len(jobs) - len(remaining)} completed; {len(remaining)} remaining', flush=True)
    print('Budgets: ' + ', '.join(f'{e}={args.steps or ENVS[e][1]:,}' for e in args.envs), flush=True)
    if args.dry_run:
        return
    out.mkdir(parents=True, exist_ok=True)
    lock = out / 'sweep.lock'
    try:
        lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f'{lock} exists. Check its PID before removing a stale lock.')
    with os.fdopen(lock_fd, 'w') as f:
        f.write(str(os.getpid()))
    try:
        manifest = out / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != jobs:
            raise SystemExit('Experiment configuration differs from the saved manifest; choose a different --out directory.')
        manifest.write_text(json.dumps(jobs, indent=2) + '\n')
        deadline = time.monotonic() + args.hours * 3600 if args.hours else float('inf')
        queue = iter(remaining)
        finished = 0
        # Only submit work that can start immediately; in-flight jobs finish after the deadline.
        with ProcessPoolExecutor(args.workers) as executor, (out / 'results.jsonl').open('a', encoding='utf-8') as f:
            pending = set()
            while True:
                while len(pending) < args.workers and time.monotonic() < deadline:
                    job = next(queue, None)
                    if job is None:
                        break
                    pending.add(executor.submit(run_one, job))
                if not pending:
                    break
                ready, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    rec = future.result()
                    f.write(json.dumps(rec) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                    finished += 1
                    value = f"final={rec['rows'][-1]['eval_return']:.3f}" if 'rows' in rec else rec['error']
                    print(f'[{finished}/{len(remaining)}] {rec["id"]}: {value}', flush=True)
        from panel.analysis.algo_report import report
        print(report(out), flush=True)
        left = {j['id'] for j in jobs} - successful_ids(load_records(out / 'results.jsonl'))
        print(f'{len(left)} runs remain; rerun the same command to resume.', flush=True)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
