import numpy as np
import pytest
import torch

from panel.agent.ppo import ActorCritic, PPOConfig
from panel.analysis.runlog import REQUIRED
from panel.envs.counting import SampleCounter
from panel.judges.ensemble import JudgeEnsemble, mean_pairwise_error_correlation
from panel.panel.query import select_queries
from panel.panel.trainer import PanelConfig, imagine, train_panel

ENV = "MiniGrid-DoorKey-5x5-v0"


def tiny_cfg(**kw):
    base = dict(n_judges=3, seed_steps=32, n_candidates=32, queries_per_round=16, judge_steps=5,
                imag_batch=32, imag_horizon=4, ppo=PPOConfig(epochs=1, minibatch=64))
    base.update(kw)
    return PanelConfig(**base)


# ---- query selection -------------------------------------------------------

def test_select_queries_top_k_plus_anchor_fraction():
    rng = np.random.default_rng(0)
    scores = np.arange(100, dtype=float)
    top, anchors = select_queries(scores, n_query=20, anchor_frac=0.15, rng=rng)
    assert len(anchors) == 3 and len(top) == 17  # ceil(0.15 * 20) anchors
    assert not set(top) & set(anchors)
    rest = np.setdiff1d(np.arange(100), anchors)
    assert set(top) == set(rest[np.argsort(-scores[rest])[:17]])  # highest remaining scores


def test_select_queries_rejects_too_few_candidates():
    with pytest.raises(ValueError):
        select_queries(np.zeros(5), 10, 0.15, np.random.default_rng(0))


def test_anchors_ignore_scores():
    """Anchors must be uniform, not disagreement-biased, or competence estimates are biased."""
    hits = np.zeros(50)
    scores = np.arange(50, dtype=float)
    for s in range(300):
        _, a = select_queries(scores, 10, 0.2, np.random.default_rng(s))
        hits[a] += 1
    assert hits[:25].sum() > 0.35 * hits.sum() and hits[25:].sum() > 0.35 * hits.sum()


# ---- judges ----------------------------------------------------------------

def _toy_data(n=256, d=6, a=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.rand(n, d, generator=g) * 0.5          # training region: [0, 0.5]
    act = torch.randint(0, a, (n,), generator=g)
    nxt = (obs + 0.1 * act.unsqueeze(-1).float()).clamp(0, 1)
    rew = obs[:, 0]
    return obs, act, rew, nxt, torch.zeros(n), torch.ones(n, 4)


def test_judges_fit_reduces_loss():
    j = JudgeEnsemble(6, 3, n_judges=4, seed=0, lr=3e-3)
    gen = torch.Generator().manual_seed(0)
    data = _toy_data()
    first = j.fit(*data, steps=5, batch_size=64, gen=gen)
    last = j.fit(*data, steps=300, batch_size=64, gen=gen)
    assert last < 0.2 * first


def test_disagreement_is_higher_off_distribution():
    j = JudgeEnsemble(6, 3, n_judges=6, seed=1, lr=3e-3)
    data = _toy_data()
    boot = torch.poisson(torch.ones(256, 6))
    j.fit(*data[:5], boot, steps=300, batch_size=64, gen=torch.Generator().manual_seed(0))
    obs, act = data[0], data[1]
    far = torch.rand(256, 6) * 0.5 + 0.5               # never trained on [0.5, 1]
    assert j.disagreement(far, act).mean() > 2 * j.disagreement(obs, act).mean()


def test_competence_weights_favour_the_accurate_judge():
    j = JudgeEnsemble(6, 3, n_judges=3, seed=0, lr=3e-3)
    obs, act, rew, nxt, done, _ = _toy_data()
    # train only judge 0 by giving the others zero bootstrap weight
    boot = torch.zeros(256, 3)
    boot[:, 0] = 1.0
    j.fit(obs, act, rew, nxt, done, boot, steps=400, batch_size=64, gen=torch.Generator().manual_seed(0))
    err = j.update_competence(obs, act, rew, nxt, ema=0.0)
    assert err.shape == (3, 256)
    assert j.weights.sum().item() == pytest.approx(1.0) and j.weights.argmax().item() == 0


def test_error_correlation_extremes():
    base = np.random.default_rng(0).random((1, 200))
    assert mean_pairwise_error_correlation(np.repeat(base, 4, 0)) == pytest.approx(1.0)
    assert abs(mean_pairwise_error_correlation(np.random.default_rng(1).random((6, 500)))) < 0.1
    assert np.isnan(mean_pairwise_error_correlation(np.zeros((3, 2))))


# ---- imagination -----------------------------------------------------------

def test_imagination_batch_shapes_and_masking():
    torch.manual_seed(0)
    model = ActorCritic(6, 3)
    j = JudgeEnsemble(6, 3, n_judges=2, seed=0)
    batch = imagine(model, j, torch.rand(10, 6), horizon=5, gamma=0.99, lam=0.95)
    n = batch["obs"].shape[0]
    assert 0 < n <= 50 and all(v.shape[0] == n for v in batch.values())
    assert torch.isfinite(batch["adv"]).all()


def test_imagination_masks_steps_after_termination():
    class AlwaysDone(JudgeEnsemble):
        def predict(self, obs, act):
            p = super().predict(obs, act)
            p["done_p"] = torch.ones_like(p["done_p"])
            return p
    batch = imagine(ActorCritic(6, 3), AlwaysDone(6, 3, n_judges=2), torch.rand(10, 6), 5, 0.99, 0.95)
    assert batch["obs"].shape[0] == 10  # only the first step of each rollout survives


def test_imagination_uses_no_grad():
    """Firewall: nothing the policy consumes from the panel carries judge gradients."""
    j = JudgeEnsemble(6, 3, n_judges=2, seed=0)
    batch = imagine(ActorCritic(6, 3), j, torch.rand(8, 6), 3, 0.99, 0.95)
    assert not any(v.requires_grad for v in batch.values())


# ---- end to end ------------------------------------------------------------

def test_train_panel_charges_every_env_step_once():
    c = SampleCounter()
    _, _, log = train_panel(ENV, total_steps=100, seed=0, cfg=tiny_cfg(), eval_episodes=1, counter=c)
    assert c.n == 100                                    # seed + queries, no overshoot
    assert log.rows[-1]["env_steps"] == 100
    assert log.rows[-1]["n_real"] == 100                 # one stored transition per charged step


def test_train_panel_log_schema_and_monotone_axes():
    _, judges, log = train_panel(ENV, total_steps=80, seed=0, cfg=tiny_cfg(), eval_episodes=1)
    assert all(k in log.rows[0] for k in REQUIRED + ("err_corr", "mean_disagreement", "anchor_err"))
    assert (np.diff(log.column("env_steps")) > 0).all()
    assert (np.diff(log.column("wall_s")) >= 0).all()
    assert log.rows[-1]["grad_steps"] >= judges.grad_steps > 0
    assert judges.weights.sum().item() == pytest.approx(1.0)


def test_train_panel_budget_smaller_than_seed_phase():
    c = SampleCounter()
    train_panel(ENV, total_steps=10, seed=0, cfg=tiny_cfg(), eval_episodes=1, counter=c)
    assert c.n == 10
