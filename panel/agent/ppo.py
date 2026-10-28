"""Compact PPO baseline for MiniGrid (fully observed, flat obs), plus shared helpers.

`compute_gae`, `ppo_update` and `evaluate` are reused by the panel agent, so both
methods share the exact same policy class and optimizer settings.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from panel.analysis.runlog import RunLog
from panel.envs.counting import SampleCounter, make_minigrid


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class PPOConfig:
    n_envs: int = 8
    n_steps: int = 128
    epochs: int = 4
    minibatch: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor):
        h = self.body(obs)
        return self.pi(h), self.v(h).squeeze(-1)


def compute_gae(rewards, values, dones, last_value, gamma: float, lam: float):
    """GAE over (T, N) tensors. dones[t] = 1 means the episode ended after step t.

    Truncation is treated as termination (no bootstrap); a small bias that is
    identical for every method here.
    """
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        next_v = last_value if t == T - 1 else values[t + 1]
        nonterm = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + values


def ppo_update(model: ActorCritic, opt: torch.optim.Optimizer, batch: Dict[str, torch.Tensor],
               cfg: PPOConfig, gen: Optional[torch.Generator] = None) -> Dict[str, float]:
    obs, act, old_logp, ret = batch["obs"], batch["act"], batch["logp"], batch["ret"]
    adv = batch["adv"]
    n = obs.shape[0]
    if n > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    grad_steps, loss_sum = 0, 0.0
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=gen).to(obs.device)
        for s in range(0, n, cfg.minibatch):
            mb = perm[s:s + cfg.minibatch]
            logits, v = model(obs[mb])
            dist = Categorical(logits=logits)
            ratio = torch.exp(dist.log_prob(act[mb]) - old_logp[mb])
            pg = -torch.min(ratio * adv[mb], torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[mb]).mean()
            vl = 0.5 * ((v - ret[mb]) ** 2).mean()
            loss = pg + cfg.vf_coef * vl - cfg.ent_coef * dist.entropy().mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            grad_steps += 1
            loss_sum += float(loss.detach())
    return {"loss": loss_sum / max(grad_steps, 1), "grad_steps": grad_steps}


@torch.no_grad()
def evaluate(model: ActorCritic, env_id: str, n_episodes: int = 10, seed: int = 10_000,
             greedy: bool = False) -> float:
    """Mean return on an uncounted env, sampling from the policy (own RNG, so training is unaffected).

    Greedy argmax is available but scores 0 on undertrained policies that loop on "forward" at a wall.
    Episodes are bounded by the env's TimeLimit.
    """
    gen = torch.Generator().manual_seed(seed)
    env = make_minigrid(env_id)
    dev = next(model.parameters()).device
    total = 0.0
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            logits, _ = model(torch.as_tensor(obs, device=dev).unsqueeze(0))
            a = logits.argmax(-1) if greedy else torch.multinomial(torch.softmax(logits, -1).cpu(), 1, generator=gen)
            obs, r, term, trunc, _ = env.step(int(a))
            total += r
            done = term or trunc
    return total / n_episodes


def train_ppo(env_id: str, total_steps: int, seed: int = 0, cfg: Optional[PPOConfig] = None,
              eval_episodes: int = 10, counter: Optional[SampleCounter] = None,
              device: str = "cpu") -> "tuple[ActorCritic, RunLog]":
    """Train until at least `total_steps` env samples are charged (overshoots by < one iteration)."""
    cfg = cfg or PPOConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)
    counter = counter or SampleCounter()
    envs = [make_minigrid(env_id, counter) for _ in range(cfg.n_envs)]
    obs = np.stack([e.reset(seed=seed * 1000 + i)[0] for i, e in enumerate(envs)])
    obs_dim, n_actions = obs.shape[1], int(envs[0].action_space.n)
    dev = pick_device(device)
    model = ActorCritic(obs_dim, n_actions, cfg.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)

    log = RunLog("ppo", env_id, seed)
    T, N = cfg.n_steps, cfg.n_envs
    ep_ret, finished = np.zeros(N), []
    train_s, grad_steps, it = 0.0, 0, 0
    while counter.n < total_steps:
        t0 = time.perf_counter()
        b_obs = torch.zeros(T, N, obs_dim, device=dev)
        b_act = torch.zeros(T, N, dtype=torch.long, device=dev)
        b_logp, b_val = (torch.zeros(T, N, device=dev) for _ in range(2))
        b_rew, b_done = np.zeros((T, N), np.float32), np.zeros((T, N), np.float32)
        for t in range(T):
            o = torch.as_tensor(obs, device=dev)
            with torch.no_grad():
                logits, v = model(o)
                dist = Categorical(logits=logits)
                a = dist.sample()
                logp = dist.log_prob(a)
            b_obs[t], b_act[t], b_logp[t], b_val[t] = o, a, logp, v
            a_np = a.cpu().numpy()
            for i, e in enumerate(envs):
                o2, r, term, trunc, _ = e.step(int(a_np[i]))
                ep_ret[i] += r
                b_rew[t, i] = r
                if term or trunc:
                    b_done[t, i] = 1.0
                    finished.append(ep_ret[i])
                    ep_ret[i] = 0.0
                    o2, _ = e.reset()
                obs[i] = o2
        with torch.no_grad():
            _, last_v = model(torch.as_tensor(obs, device=dev))
        b_rew, b_done = torch.as_tensor(b_rew, device=dev), torch.as_tensor(b_done, device=dev)
        adv, ret = compute_gae(b_rew, b_val, b_done, last_v, cfg.gamma, cfg.gae_lambda)
        batch = {"obs": b_obs.reshape(T * N, -1), "act": b_act.reshape(-1), "logp": b_logp.reshape(-1),
                 "adv": adv.reshape(-1), "ret": ret.reshape(-1)}
        stats = ppo_update(model, opt, batch, cfg)
        grad_steps += stats["grad_steps"]
        train_s += time.perf_counter() - t0
        it += 1
        log.add(iteration=it, env_steps=counter.n, wall_s=train_s, grad_steps=grad_steps,
                eval_return=evaluate(model, env_id, eval_episodes),
                train_return=float(np.mean(finished[-20:])) if finished else 0.0, loss=stats["loss"])
    return model, log
