"""Double DQN, optionally using panel-selected real queries and Dyna updates.

Panel DQN and Dyna share data collection and real replay updates. Only Dyna
adds one-step model transitions; these never enter the real replay buffer.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from panel.analysis.runlog import RunLog
from panel.envs.counting import Oracle, SampleCounter, make_minigrid
from panel.judges.ensemble import JudgeEnsemble, mean_pairwise_error_correlation
from panel.panel.query import select_queries
from panel.panel.replay import Replay


@dataclass
class DQNConfig:
    hidden: int = 64
    lr: float = 3e-4
    gamma: float = .99
    batch_size: int = 128
    warmup: int = 256
    round_steps: int = 96
    real_updates: int = 24
    model_updates: int = 24
    judge_steps: int = 25
    target_interval: int = 200
    epsilon_final: float = .05
    epsilon_decay: float = .5
    replay_capacity: int = 100_000
    eval_interval: int = 1024


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, obs):
        return self.net(obs)


def td_target(online, target, next_obs, reward, done, gamma):
    with torch.no_grad():
        actions = online(next_obs).argmax(-1, keepdim=True)
        return reward + gamma * (1 - done) * target(next_obs).gather(1, actions).squeeze(1)


def q_update(model, target, opt, batch, gamma):
    obs, act, reward, next_obs, done = batch
    y = td_target(model, target, next_obs, reward, done, gamma)
    loss = F.smooth_l1_loss(model(obs).gather(1, act[:, None]).squeeze(1), y)
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 10.)
    opt.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate_q(model, env_id, episodes, seed=10_000):
    env = make_minigrid(env_id)
    total = 0.
    try:
        for i in range(episodes):
            obs, _ = env.reset(seed=seed + i)
            done = False
            while not done:
                action = int(model(torch.as_tensor(obs).unsqueeze(0)).argmax(-1))
                obs, reward, term, trunc, _ = env.step(action)
                total += reward
                done = term or trunc
    finally:
        env.close()
    return total / episodes


def train_dqn(env_id, total_steps, seed=0, n_judges=0, dyna=False,
              cfg=None, eval_episodes=10, counter=None):
    cfg = cfg or DQNConfig()
    if dyna and n_judges < 1:
        raise ValueError('Dyna requires at least one judge')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    counter = counter or SampleCounter()
    oracle = Oracle(env_id, counter)
    replay = Replay(oracle.obs_dim, max(1, n_judges), 1000, rng)
    model = QNetwork(oracle.obs_dim, oracle.n_actions, cfg.hidden)
    target = copy.deepcopy(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    judges = JudgeEnsemble(oracle.obs_dim, oracle.n_actions, n_judges, seed) if n_judges else None
    log = RunLog('dyna' if dyna else 'dqn', env_id, seed)
    obs, snap = oracle.reset(seed=seed)
    replay.add_state(obs, snap)
    elapsed = cpu = 0.
    grad = iteration = real_grad = model_grad = 0
    next_eval = cfg.eval_interval

    def actions(x, epsilon):
        with torch.no_grad():
            a = model(x).argmax(-1).numpy()
        mask = rng.random(len(a)) < epsilon
        a[mask] = rng.integers(oracle.n_actions, size=int(mask.sum()))
        return a

    while counter.n < total_steps:
        start, cstart = time.perf_counter(), time.process_time()
        epsilon = max(cfg.epsilon_final, 1 - (1 - cfg.epsilon_final) *
                      counter.n / max(1, total_steps * cfg.epsilon_decay))
        err = None
        # Panel methods spend 64 samples on queries and 32 on a fresh trajectory.
        # The baseline spends all 96 samples on its current trajectory.
        if judges and counter.n >= cfg.warmup:
            count = min(64, total_steps - counter.n)
            idx = rng.choice(len(replay.pool_obs), max(256, count))
            candidates = torch.as_tensor(np.stack([replay.pool_obs[i] for i in idx]))
            acts = actions(candidates, epsilon)
            scores = judges.disagreement(candidates, torch.as_tensor(acts)).numpy()
            # One judge has no disagreement signal: use uniform queries explicitly.
            if n_judges == 1:
                scores = rng.random(len(idx))
            top, anchors = select_queries(scores, count, .15, rng)
            pending, anchor_data = [], []
            for k in np.concatenate([top, anchors]):
                old = replay.pool_obs[idx[k]]
                nxt, reward, term, trunc, nsnap = oracle.query(replay.pool_snaps[idx[k]], int(acts[k]))
                replay.add_transition(old, int(acts[k]), reward, nxt, term or trunc)
                if k in anchors:
                    anchor_data.append((old, acts[k], reward, nxt))
                if not (term or trunc):
                    pending.append((nxt, nsnap))
            for item in pending:
                replay.add_state(*item)
            ao, aa, ar, an = zip(*anchor_data)
            err = judges.update_competence(torch.as_tensor(np.stack(ao)), torch.as_tensor(np.array(aa)),
                                           torch.as_tensor(np.array(ar), dtype=torch.float32),
                                           torch.as_tensor(np.stack(an)))
            collect = 32
        else:
            collect = cfg.round_steps
        for _ in range(min(collect, total_steps - counter.n)):
            action = int(actions(torch.as_tensor(obs).unsqueeze(0), epsilon)[0])
            nxt, reward, term, trunc, nsnap = oracle.query(snap, action)
            replay.add_transition(obs, action, reward, nxt, term or trunc)
            if term or trunc:
                obs, snap = oracle.reset()
            else:
                obs, snap = nxt, nsnap
            replay.add_state(obs, snap)
        if len(replay) > cfg.replay_capacity:
            for name in ('obs', 'act', 'rew', 'next_obs', 'done', 'boot'):
                setattr(replay, name, getattr(replay, name)[-cfg.replay_capacity:])
        data = replay.tensors()
        if counter.n >= cfg.warmup:
            if judges:
                judges.fit(*data, cfg.judge_steps, cfg.batch_size, gen)
            for imagined in ([False, True] if dyna else [False]):
                for _ in range(cfg.model_updates if imagined else cfg.real_updates):
                    idx = torch.randint(len(replay), (min(cfg.batch_size, len(replay)),), generator=gen)
                    batch = tuple(x[idx] for x in data[:5])
                    if imagined:
                        o = batch[0]
                        a = torch.as_tensor(actions(o, epsilon))
                        with torch.no_grad():
                            p = judges.predict(o, a)
                            # Sample an individual judge, avoiding an average of predicted grids.
                            j = torch.randint(n_judges, (len(o),), generator=gen)
                            b = torch.arange(len(o))
                            batch = (o, a, p['reward'][j, b].clamp(-1, 1),
                                     p['next_obs'][j, b], (p['done_p'][j, b] > .5).float())
                    q_update(model, target, opt, batch, cfg.gamma)
                    grad += 1
                    if imagined:
                        model_grad += 1
                    else:
                        real_grad += 1
                    if grad % cfg.target_interval == 0:
                        target.load_state_dict(model.state_dict())
        iteration += 1
        elapsed += time.perf_counter() - start
        cpu += time.process_time() - cstart
        if counter.n >= next_eval or counter.n == total_steps:
            extra = {'err_corr': mean_pairwise_error_correlation(err)} if err is not None else {}
            log.add(iteration=iteration, env_steps=counter.n, wall_s=elapsed, cpu_s=cpu,
                    grad_steps=grad + (judges.grad_steps if judges else 0),
                    eval_return=evaluate_q(model, env_id, eval_episodes), epsilon=epsilon,
                    real_grad_steps=real_grad, model_grad_steps=model_grad, **extra)
            next_eval = counter.n + cfg.eval_interval
    oracle.env.close()
    return model, judges, log
