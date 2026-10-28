"""Static judge-panel agent (Phase 1, no turnover).

One round:
  1. sample candidate stored states, let the policy pick an action for each
  2. score candidates by judge disagreement; resolve top-k + ~15% random anchors in the env
  3. score judges on the anchors (before training on them) -> competence weights
  4. refit judges on all real transitions
  5. improve the policy with PPO on rollouts imagined by the competence-weighted panel

The policy never sees judge parameters or gradients, only consensus predictions.
Real env reward reaches the policy only through the judges' reward head.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.distributions import Categorical

from panel.agent.ppo import ActorCritic, PPOConfig, compute_gae, evaluate, pick_device, ppo_update
from panel.analysis.runlog import RunLog
from panel.envs.counting import Oracle, SampleCounter
from panel.judges.ensemble import JudgeEnsemble, mean_pairwise_error_correlation
from panel.panel.query import select_queries
from panel.panel.replay import Replay


@dataclass
class PanelConfig:
    n_judges: int = 8
    seed_steps: int = 256          # random-policy real steps before round 1 (charged)
    n_candidates: int = 256
    queries_per_round: int = 64    # real steps per round (charged)
    anchor_frac: float = 0.15
    judge_steps: int = 100
    judge_batch: int = 128
    judge_lr: float = 1e-3
    imag_batch: int = 256
    imag_horizon: int = 8
    pool_size: int = 1000
    ppo: PPOConfig = field(default_factory=lambda: PPOConfig(epochs=2))


@torch.no_grad()
def imagine(model: ActorCritic, judges: JudgeEnsemble, start_obs: torch.Tensor, horizon: int,
            gamma: float, lam: float):
    """Roll the policy through the weighted panel consensus. Returns a PPO batch.

    Steps after a predicted termination are masked out of the batch.
    """
    B, D = start_obs.shape
    obs = start_obs
    dev = start_obs.device
    O = torch.zeros(horizon, B, D, device=dev); A = torch.zeros(horizon, B, dtype=torch.long, device=dev)
    LP, V, R, DN, ALIVE = (torch.zeros(horizon, B, device=dev) for _ in range(5))
    alive = torch.ones(B, device=dev)
    for t in range(horizon):
        logits, v = model(obs)
        dist = Categorical(logits=logits)
        a = dist.sample()
        mixed = judges.mix(judges.predict(obs, a))
        done = (mixed["done_p"] > 0.5).float()
        O[t], A[t], LP[t], V[t] = obs, a, dist.log_prob(a), v
        R[t], DN[t], ALIVE[t] = mixed["reward"].clamp(0, 1), done, alive
        alive = alive * (1.0 - done)
        obs = mixed["next_obs"]
    _, last_v = model(obs)
    adv, ret = compute_gae(R, V, DN, last_v, gamma, lam)
    keep = ALIVE.reshape(-1) > 0.5
    flat = lambda x: x.reshape(horizon * B, *x.shape[2:])[keep]
    return {"obs": flat(O), "act": flat(A), "logp": flat(LP), "adv": flat(adv), "ret": flat(ret)}


def train_panel(env_id: str, total_steps: int, seed: int = 0, cfg: Optional[PanelConfig] = None,
                eval_episodes: int = 10, counter: Optional[SampleCounter] = None, device: str = "cpu"):
    """Train until `total_steps` env samples are charged (seed rollouts and every query count)."""
    cfg = cfg or PanelConfig()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    counter = counter or SampleCounter()
    dev = pick_device(device)
    oracle = Oracle(env_id, counter)
    replay = Replay(oracle.obs_dim, cfg.n_judges, cfg.pool_size, rng)
    judges = JudgeEnsemble(oracle.obs_dim, oracle.n_actions, cfg.n_judges, seed, cfg.judge_lr, device=str(dev))
    model = ActorCritic(oracle.obs_dim, oracle.n_actions, cfg.ppo.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ppo.lr, eps=1e-5)
    log = RunLog("panel", env_id, seed)
    t0 = time.perf_counter()

    # seed data: random-policy real steps
    obs, snap = oracle.reset(seed=seed)
    replay.add_state(obs, snap)
    for _ in range(min(cfg.seed_steps, total_steps)):
        a = int(rng.integers(oracle.n_actions))
        o2, r, term, trunc, nxt = oracle.query(snap, a)
        replay.add_transition(obs, a, r, o2, term)
        if term or trunc:
            obs, snap = oracle.reset()
        else:
            obs, snap = o2, nxt
        replay.add_state(obs, snap)

    train_s, policy_steps, it = time.perf_counter() - t0, 0, 0
    while counter.n < total_steps:
        t0 = time.perf_counter()
        n_query = min(cfg.queries_per_round, total_steps - counter.n)
        n_cand = max(cfg.n_candidates, n_query)
        idx = rng.choice(len(replay.pool_snaps), n_cand, replace=len(replay.pool_snaps) < n_cand)
        c_obs = torch.as_tensor(np.stack([replay.pool_obs[i] for i in idx]), device=dev)
        with torch.no_grad():
            c_act = Categorical(logits=model(c_obs)[0]).sample()
        scores = judges.disagreement(c_obs, c_act).cpu().numpy()
        c_act_np = c_act.cpu().numpy()
        top, anchors = select_queries(scores, n_query, cfg.anchor_frac, rng)

        # resolve in the env; pool additions are deferred so candidate snapshots stay valid
        new_states, anchor_rows = [], []
        for k in np.concatenate([top, anchors]):
            o, a = replay.pool_obs[idx[k]], int(c_act_np[k])
            o2, r, term, trunc, nxt = oracle.query(replay.pool_snaps[idx[k]], a)
            if k in anchors:
                anchor_rows.append((o, a, r, o2))
            replay.add_transition(o, a, r, o2, term)
            if not (term or trunc):
                new_states.append((o2, nxt))
        for o2, nxt in new_states:
            replay.add_state(o2, nxt)

        a_obs, a_act, a_rew, a_next = (np.array(x) for x in zip(*anchor_rows))
        err = judges.update_competence(torch.as_tensor(a_obs, dtype=torch.float32, device=dev),
                                       torch.as_tensor(a_act, dtype=torch.long, device=dev),
                                       torch.as_tensor(a_rew, dtype=torch.float32, device=dev),
                                       torch.as_tensor(a_next, dtype=torch.float32, device=dev))
        judges.fit(*replay.tensors(dev), cfg.judge_steps, cfg.judge_batch, gen)

        starts = torch.as_tensor(np.stack([replay.pool_obs[i] for i in
                                           rng.choice(len(replay.pool_obs), cfg.imag_batch)]), device=dev)
        batch = imagine(model, judges, starts, cfg.imag_horizon, cfg.ppo.gamma, cfg.ppo.gae_lambda)
        if batch["obs"].shape[0] > 0:
            policy_steps += ppo_update(model, opt, batch, cfg.ppo, gen)["grad_steps"]
        train_s += time.perf_counter() - t0
        it += 1
        log.add(iteration=it, env_steps=counter.n, wall_s=train_s,
                grad_steps=policy_steps + judges.grad_steps,
                eval_return=evaluate(model, env_id, eval_episodes),
                mean_disagreement=float(scores.mean()), anchor_err=float(err.mean()),
                err_corr=mean_pairwise_error_correlation(err), n_real=len(replay))
    return model, judges, log
