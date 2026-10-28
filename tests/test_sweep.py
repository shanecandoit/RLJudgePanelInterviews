import json

import numpy as np
import pytest

from panel.analysis.sweep_report import report
from panel.experiments.h1_sweep import SPACE, design, make_cfg
from panel.judges.ensemble import spearman
from panel.panel.trainer import train_panel
from tests.test_panel import ENV, tiny_cfg


def test_design_is_balanced_and_deterministic():
    d = design("MiniGrid-Empty-5x5-v0", 20, 0)
    assert d == design("MiniGrid-Empty-5x5-v0", 20, 0)
    for f, vals in SPACE.items():
        counts = [sum(r["config"][f] == v for r in d) for v in vals]
        assert max(counts) - min(counts) <= 1


def test_make_cfg_adds_real_rollouts_only_when_mixing():
    base = {"ent_coef": 0.01, "imag_horizon": 2, "n_judges": 4}
    assert make_cfg({**base, "imag_frac": 1.0}).real_starts == 0
    assert make_cfg({**base, "imag_frac": 0.5}).real_starts == 8


def test_spearman():
    assert spearman(np.arange(10.0), np.arange(10.0) ** 3) == pytest.approx(1.0)
    assert np.isnan(spearman(np.arange(2.0), np.arange(2.0)))
    assert np.isnan(spearman(np.ones(5), np.arange(5.0)))


def test_panel_logs_calibration_columns():
    _, _, log = train_panel(ENV, total_steps=80, seed=0, cfg=tiny_cfg(), eval_episodes=1)
    for k in ("calib", "rank_corr", "n_eff", "anchor_disagree", "consensus_err"):
        assert k in log.rows[-1]
    assert 1.0 <= log.rows[-1]["n_eff"] <= 3.0 + 1e-6


def test_report_on_synthetic_results(tmp_path):
    rng = np.random.default_rng(0)
    with open(tmp_path / "results.jsonl", "w") as f:
        for r in design("MiniGrid-Empty-5x5-v0", 12, 0):
            rows = [{"iteration": i, "env_steps": 100 * (i + 1), "wall_s": float(i), "grad_steps": i,
                     "eval_return": float(min(0.8, 0.1 * i * rng.random())), "calib": 0.2, "rank_corr": 0.3, "n_eff": 2.0}
                    for i in range(15)]
            f.write(json.dumps({**r, "rows": rows}) + "\n")
    text = report(str(tmp_path))
    assert "marginals" in text and "top 10 runs" in text
    assert list(tmp_path.glob("sweep_*.png"))
