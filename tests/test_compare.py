import pytest

from panel.analysis.compare import asymptote, compare, first_reaching, format_table
from panel.analysis.runlog import RunLog


def make_log(algo, returns, step_per_iter, s_per_iter=1.0, seed=0):
    log = RunLog(algo, "env", seed)
    for i, r in enumerate(returns, 1):
        log.add(iteration=i, env_steps=i * step_per_iter, wall_s=i * s_per_iter, grad_steps=i, eval_return=r)
    return log


def test_runlog_requires_shared_columns():
    with pytest.raises(KeyError):
        RunLog("x", "e", 0).add(iteration=1, env_steps=1)


def test_runlog_roundtrip(tmp_path):
    log = make_log("ppo", [0.1, 0.2], 10)
    log.save(str(tmp_path / "l.json"))
    again = RunLog.load(str(tmp_path / "l.json"))
    assert again.rows == log.rows and again.algo == "ppo"


def test_asymptote_and_first_reaching():
    log = make_log("ppo", [0.0, 0.2, 0.5, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 100)
    assert asymptote(log) == pytest.approx(1.0)
    assert first_reaching(log, 0.8, "env_steps") == 400
    assert first_reaching(log, 2.0, "env_steps") is None


def test_compare_reports_ratios_on_samples_iterations_and_time():
    ppo = [make_log("ppo", [0, 0, 0, 0, 0.8, 1, 1, 1, 1, 1], 1000, 1.0)]       # reaches 0.8 at iter 5
    panel = [make_log("panel", [0, 0.9, 1, 1, 1, 1, 1, 1, 1, 1], 200, 4.0)]    # reaches 0.8 at iter 2
    r = compare(ppo, panel, frac=0.8)
    assert r["asymptote"] == pytest.approx(1.0) and r["threshold"] == pytest.approx(0.8)
    assert r["env_steps"]["ratio"] == pytest.approx(5000 / 400)
    assert r["iteration"]["ratio"] == pytest.approx(5 / 2)
    assert r["wall_s"]["ratio"] == pytest.approx(5.0 / 8.0)  # fewer samples but slower in wall time
    assert "PASS" in format_table(r)


def test_compare_handles_method_that_never_reaches_threshold():
    ppo = [make_log("ppo", [0, 1, 1, 1, 1], 100)]
    panel = [make_log("panel", [0, 0, 0, 0, 0], 10)]
    r = compare(ppo, panel)
    assert r["env_steps"]["panel"] is None and r["env_steps"]["ratio"] is None
    assert r["env_steps"]["panel_reached"] == 0.0
    assert "not measurable" in format_table(r)


def test_compare_median_over_seeds_counts_failures_as_infinite():
    ppo = [make_log("ppo", [0, 1, 1, 1, 1], 100, seed=s) for s in range(3)]
    good, bad = make_log("panel", [1, 1, 1, 1, 1], 10), make_log("panel", [0] * 5, 10)
    assert compare(ppo, [good, good, bad])["env_steps"]["panel"] == 10     # 2 of 3 reached -> median finite
    assert compare(ppo, [good, bad, bad])["env_steps"]["panel"] is None    # 1 of 3 -> median infinite
