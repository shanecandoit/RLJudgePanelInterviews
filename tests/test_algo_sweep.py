import numpy as np
import pytest
import torch

from panel.agent.dqn import DQNConfig, td_target, train_dqn
from panel.analysis.algo_report import summarize
from panel.envs.counting import SampleCounter
from panel.experiments.algo_sweep import build_jobs, successful_ids
from panel.judges.ensemble import mean_pairwise_error_correlation

ENV = 'MiniGrid-Empty-5x5-v0'


def test_grid_and_resume_fingerprints():
    args = (['ppo', 'dqn', 'dyna'], ['empty5x5', 'lavagap', 'doorkey'], [1, 2, 4], [0, 1, 2])
    jobs = build_jobs(*args)
    assert len(jobs) == len({j['id'] for j in jobs}) == 99
    assert sum(j['variant'] == 'baseline' for j in jobs) == 18
    assert {j['steps'] for j in jobs if j['env'] == ENV} == {40_000}
    assert {j['id'] for j in build_jobs(*args, steps=100)}.isdisjoint(j['id'] for j in jobs)
    assert successful_ids([dict(id='failed', error='oops'), dict(id='ok', rows=[{}])]) == {'ok'}


def test_double_dqn_target_uses_online_action_and_masks_terminal():
    online = torch.nn.Linear(1, 2)
    target = torch.nn.Linear(1, 2)
    with torch.no_grad():
        online.weight.zero_(); target.weight.zero_()
        online.bias.copy_(torch.tensor([1., 2.]))
        target.bias.copy_(torch.tensor([10., 3.]))
    y = td_target(online, target, torch.zeros(2, 1), torch.ones(2), torch.tensor([0., 1.]), .9)
    assert y.tolist() == pytest.approx([3.7, 1.])
    assert not y.requires_grad


@pytest.mark.parametrize('judges,dyna', [(0, False), (1, False), (2, True)])
def test_dqn_accounting_and_real_vs_model_updates(judges, dyna):
    counter = SampleCounter()
    cfg = DQNConfig(warmup=16, real_updates=2, model_updates=2, judge_steps=1,
                    batch_size=16, eval_interval=96, target_interval=2)
    _, panel, log = train_dqn(ENV, 200, seed=0, n_judges=judges, dyna=dyna,
                              cfg=cfg, eval_episodes=1, counter=counter)
    assert counter.n == log.rows[-1]['env_steps'] == 200
    assert log.rows[-1]['real_grad_steps'] > 0
    assert (log.rows[-1]['model_grad_steps'] > 0) == dyna
    assert (panel is not None) == (judges > 0)
    assert np.isfinite(log.column('eval_return')).all()
    assert (np.diff(log.column('env_steps')) > 0).all()


def test_single_judge_correlation_is_undefined():
    assert np.isnan(mean_pairwise_error_correlation(np.ones((1, 10))))


def test_panel_sparse_evaluation_keeps_final_measurement():
    from panel.panel.trainer import train_panel
    from tests.test_panel import tiny_cfg
    _, _, log = train_panel(ENV, 80, seed=0, eval_episodes=1,
                             cfg=tiny_cfg(n_judges=1, eval_interval=1024))
    assert len(log.rows) == 1 and log.rows[0]['env_steps'] == 80


def test_report_distinguishes_peak_from_sustained_and_deduplicates():
    base = dict(id='base', env=ENV, algo='ppo', variant='baseline', n_judges=0, seed=0, steps=100)
    row = lambda x, y: dict(env_steps=x, eval_return=y, wall_s=x / 10)
    baseline = dict(base, rows=[row(20, .8), row(100, .8)])
    panel = dict(base, id='panel', variant='panel', n_judges=4,
                 rows=[row(20, .9), row(80, 0.), row(100, 0.)])
    frame, good = summarize([baseline, panel, panel])
    assert len(good) == len(frame) == 2
    result = frame[frame.id == 'panel'].iloc[0]
    assert result.reached and result.collapsed and not result.late_above_threshold
