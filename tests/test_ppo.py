import pytest
import torch

from panel.agent.ppo import ActorCritic, PPOConfig, compute_gae, evaluate, ppo_update, train_ppo
from panel.analysis.runlog import REQUIRED
from panel.envs.counting import SampleCounter

ENV = "MiniGrid-DoorKey-5x5-v0"


def test_gae_matches_hand_computation():
    r = torch.tensor([[1.0], [0.0], [2.0]])
    v = torch.tensor([[0.5], [0.4], [0.3]])
    d = torch.tensor([[0.0], [0.0], [1.0]])
    adv, ret = compute_gae(r, v, d, torch.tensor([9.0]), gamma=0.9, lam=0.5)
    # last step is terminal: delta2 = 2 - 0.3
    a2 = 1.7
    a1 = (0.0 + 0.9 * 0.3 - 0.4) + 0.9 * 0.5 * a2
    a0 = (1.0 + 0.9 * 0.4 - 0.5) + 0.9 * 0.5 * a1
    assert adv.squeeze().tolist() == pytest.approx([a0, a1, a2])
    assert ret.squeeze().tolist() == pytest.approx([a0 + 0.5, a1 + 0.4, a2 + 0.3])


def test_gae_does_not_bootstrap_through_done():
    r = torch.zeros(2, 1)
    v = torch.zeros(2, 1)
    d = torch.tensor([[1.0], [0.0]])
    adv, _ = compute_gae(r, v, d, torch.tensor([5.0]), gamma=1.0, lam=1.0)
    assert adv[0].item() == 0.0  # the value after the episode end must not leak back
    assert adv[1].item() == 5.0


def test_ppo_update_moves_params_and_is_finite():
    torch.manual_seed(0)
    m = ActorCritic(6, 3)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    obs = torch.randn(64, 6)
    with torch.no_grad():
        logits, v = m(obs)
    dist = torch.distributions.Categorical(logits=logits)
    act = dist.sample()
    batch = {"obs": obs, "act": act, "logp": dist.log_prob(act), "adv": torch.randn(64), "ret": torch.randn(64)}
    before = [p.clone() for p in m.parameters()]
    stats = ppo_update(m, opt, batch, PPOConfig(epochs=2, minibatch=32))
    assert stats["grad_steps"] == 4 and torch.isfinite(torch.tensor(stats["loss"]))
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))


def test_ppo_update_prefers_high_advantage_action():
    torch.manual_seed(0)
    m = ActorCritic(4, 2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    obs = torch.zeros(128, 4)
    act = torch.cat([torch.zeros(64), torch.ones(64)]).long()
    adv = torch.cat([-torch.ones(64), torch.ones(64)])
    with torch.no_grad():
        logp = torch.distributions.Categorical(logits=m(obs)[0]).log_prob(act)
    for _ in range(20):
        ppo_update(m, opt, {"obs": obs, "act": act, "logp": logp, "adv": adv, "ret": torch.zeros(128)},
                   PPOConfig(epochs=1, minibatch=128, ent_coef=0.0))
    assert torch.softmax(m(obs[:1])[0], -1)[0, 1] > 0.5


def test_train_ppo_smoke_accounting_and_log_schema():
    c = SampleCounter()
    cfg = PPOConfig(n_envs=2, n_steps=16, minibatch=16)
    _, log = train_ppo(ENV, total_steps=100, seed=0, cfg=cfg, eval_episodes=1, counter=c)
    steps = log.column("env_steps")
    assert c.n == steps[-1] and 100 <= c.n < 100 + 2 * 16  # overshoot < one iteration
    assert (steps[1:] > steps[:-1]).all()
    assert (log.column("wall_s")[1:] >= log.column("wall_s")[:-1]).all()
    assert all(k in log.rows[0] for k in REQUIRED)


def test_train_ppo_is_seed_deterministic():
    cfg = PPOConfig(n_envs=2, n_steps=16, minibatch=16)
    _, a = train_ppo(ENV, 64, seed=3, cfg=cfg, eval_episodes=1)
    _, b = train_ppo(ENV, 64, seed=3, cfg=cfg, eval_episodes=1)
    assert [r["loss"] for r in a.rows] == [r["loss"] for r in b.rows]


@pytest.mark.slow
def test_ppo_learns_empty_grid():
    model, log = train_ppo("MiniGrid-Empty-5x5-v0", total_steps=60_000, seed=0)
    assert evaluate(model, "MiniGrid-Empty-5x5-v0", 10) > 0.5


def test_evaluate_samples_by_default_and_does_not_touch_global_rng():
    torch.manual_seed(0)
    m = ActorCritic(75, 7)
    state = torch.get_rng_state()
    a = evaluate(m, "MiniGrid-Empty-5x5-v0", 2)
    assert torch.equal(state, torch.get_rng_state())          # eval must not perturb training RNG
    assert a == evaluate(m, "MiniGrid-Empty-5x5-v0", 2)       # but is reproducible


def test_evaluate_sampling_escapes_a_greedy_loop():
    """A policy that prefers 'forward' (2) but keeps some mass elsewhere: argmax loops at the wall, sampling finishes."""
    m = ActorCritic(75, 7)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.pi.bias[2] = 3.0
    assert evaluate(m, "MiniGrid-Empty-5x5-v0", 10, greedy=True) == 0.0
    assert evaluate(m, "MiniGrid-Empty-5x5-v0", 10) > 0.0
