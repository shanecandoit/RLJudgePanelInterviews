"""Real-transition buffer and the pool of resumable env states the panel queries from."""
from __future__ import annotations

from typing import Any, List

import numpy as np
import torch


class Replay:
    def __init__(self, obs_dim: int, n_judges: int, pool_size: int, rng: np.random.Generator) -> None:
        self.rng, self.n_judges, self.pool_size = rng, n_judges, pool_size
        self.obs, self.act, self.rew, self.next_obs, self.done, self.boot = [], [], [], [], [], []
        self.pool_obs: List[np.ndarray] = []
        self.pool_snaps: List[Any] = []

    def __len__(self) -> int:
        return len(self.obs)

    def add_transition(self, obs, act, rew, next_obs, done) -> None:
        self.obs.append(obs); self.act.append(act); self.rew.append(rew)
        self.next_obs.append(next_obs); self.done.append(float(done))
        self.boot.append(self.rng.poisson(1.0, self.n_judges).astype(np.float32))  # bootstrap weights

    def add_state(self, obs, snap) -> None:
        if len(self.pool_snaps) < self.pool_size:
            self.pool_obs.append(obs); self.pool_snaps.append(snap)
        else:
            i = int(self.rng.integers(self.pool_size))
            self.pool_obs[i], self.pool_snaps[i] = obs, snap

    def tensors(self, device="cpu"):
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=device)
        return (t(self.obs), t(self.act, torch.long), t(self.rew), t(self.next_obs), t(self.done), t(self.boot))
