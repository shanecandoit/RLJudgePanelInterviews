import os

from panel.analysis.plot import plot_dir
from panel.analysis.runlog import RunLog


def test_plot_dir_writes_png(tmp_path):
    for algo in ("ppo", "panel"):
        for s in range(2):
            log = RunLog(algo, "env", s)
            for i in range(1, 6):
                log.add(iteration=i, env_steps=i * 100, wall_s=float(i), grad_steps=i, eval_return=i / 5)
            log.save(str(tmp_path / f"{algo}_s{s}.json"))
    out = plot_dir(str(tmp_path))
    assert os.path.getsize(out) > 0
