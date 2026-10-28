"""Static judge ensemble: dynamics (head 1) and reward/outcome (head 2) per judge.

v0 diversity: seeds, bootstrap weights and hidden width. CNN/ViT backbones,
the action-prediction and self-calibration heads, and a proper scoring rule
(Gaussian NLL) are deferred; see plan.md Phase 1.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(nn.Linear(obs_dim + n_actions, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, obs_dim + 2))

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        x = torch.cat([obs, F.one_hot(act, self.n_actions).float()], -1)
        out = self.net(x)
        return out[..., :-2], out[..., -2], out[..., -1]  # delta_obs, reward, done_logit


class JudgeEnsemble:
    def __init__(self, obs_dim: int, n_actions: int, n_judges: int = 8, seed: int = 0, lr: float = 1e-3,
                 hidden_sizes: Sequence[int] = (64, 128), device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.J, self.obs_dim, self.n_actions = n_judges, obs_dim, n_actions
        self.models = []
        for j in range(n_judges):
            torch.manual_seed(seed * 100 + j)
            self.models.append(WorldModel(obs_dim, n_actions, hidden_sizes[j % len(hidden_sizes)]).to(self.device))
        params = [p for m in self.models for p in m.parameters()]
        self.opt = torch.optim.Adam(params, lr=lr)
        self.weights = torch.full((n_judges,), 1.0 / n_judges, device=self.device)  # competence; anchors only
        self.grad_steps = 0

    @torch.no_grad()
    def predict(self, obs: torch.Tensor, act: torch.Tensor) -> Dict[str, torch.Tensor]:
        outs = [m(obs, act) for m in self.models]
        return {"next_obs": torch.stack([(obs + o[0]).clamp(0, 1) for o in outs]),   # (J, B, D)
                "reward": torch.stack([o[1] for o in outs]),                          # (J, B)
                "done_p": torch.stack([torch.sigmoid(o[2]) for o in outs])}          # (J, B)

    def disagreement(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        p = self.predict(obs, act)
        return p["next_obs"].var(0, unbiased=False).mean(-1) + p["reward"].var(0, unbiased=False)

    def mix(self, p: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Competence-weighted consensus. Only these means leave the panel (information firewall)."""
        w = self.weights.view(-1, 1)
        return {"next_obs": (p["next_obs"] * w.unsqueeze(-1)).sum(0),
                "reward": (p["reward"] * w).sum(0), "done_p": (p["done_p"] * w).sum(0)}

    def fit(self, obs, act, rew, next_obs, done, boot, steps: int, batch_size: int,
            gen: torch.Generator) -> float:
        """boot: (N, J) bootstrap weights. Judges see different resamples of the same buffer."""
        n, total = obs.shape[0], 0.0
        for _ in range(steps):
            idx = torch.randint(0, n, (min(batch_size, n),), generator=gen).to(obs.device)
            loss = 0.0
            for j, m in enumerate(self.models):
                d, r, dl = m(obs[idx], act[idx])
                per = ((d - (next_obs[idx] - obs[idx])) ** 2).mean(-1) + (r - rew[idx]) ** 2 \
                    + F.binary_cross_entropy_with_logits(dl, done[idx], reduction="none")
                loss = loss + (boot[idx, j] * per).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            self.grad_steps += 1
            total += float(loss.detach()) / self.J
        return total / max(steps, 1)

    def update_competence(self, obs, act, rew, next_obs, tau: float = 1.0, ema: float = 0.5) -> np.ndarray:
        """Score judges on anchor transitions *before* they enter training. Returns (J, n) errors.

        Softmax over negative mean error, scaled by the panel-mean error. A cheap
        stand-in for the 2-parameter IRT fit planned in the architecture section.
        """
        p = self.predict(obs, act)
        err = ((p["next_obs"] - next_obs) ** 2).mean(-1) + (p["reward"] - rew) ** 2  # (J, n)
        m = err.mean(1)
        w = torch.softmax(-m / (tau * m.mean() + 1e-8), 0)
        self.weights = ema * self.weights + (1 - ema) * w
        self.weights = self.weights / self.weights.sum()
        return err.cpu().numpy()


def mean_pairwise_error_correlation(err: np.ndarray) -> float:
    """Mean off-diagonal correlation of per-sample judge errors (H4 gate). NaN if too few samples."""
    if err.shape[1] < 3:
        return float("nan")
    c = np.nan_to_num(np.corrcoef(err), nan=0.0)
    J = c.shape[0]
    return float((c.sum() - np.trace(c)) / (J * (J - 1)))
