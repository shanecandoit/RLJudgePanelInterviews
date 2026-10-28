"""H1 comparison: cost to reach a fraction of the PPO asymptote, by samples / iterations / time."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from panel.analysis.runlog import RunLog

METRICS = ("env_steps", "iteration", "wall_s")
H1_SAMPLE_RATIO = 5.0
MIN_ASYMPTOTE = 0.1  # below this PPO has not learned the task; a threshold of 80% of ~0 is meaningless


def asymptote(log: RunLog, last_frac: float = 0.2) -> float:
    """Mean eval return over the last `last_frac` of a run."""
    v = log.column("eval_return")
    return float(v[-max(1, int(round(len(v) * last_frac))):].mean())


def first_reaching(log: RunLog, threshold: float, x: str) -> Optional[float]:
    for row in log.rows:
        if row["eval_return"] >= threshold:
            return row[x]
    return None


def _median(vals: Sequence[Optional[float]]) -> Optional[float]:
    """Median over seeds; seeds that never reach the threshold count as infinity."""
    m = float(np.median([np.inf if v is None else v for v in vals]))
    return None if np.isinf(m) else m


def compare(ppo_logs: List[RunLog], panel_logs: List[RunLog], frac: float = 0.8) -> Dict:
    asym = float(np.mean([asymptote(l) for l in ppo_logs]))
    thr = frac * asym
    out: Dict = {"asymptote": asym, "threshold": thr, "frac": frac, "valid": asym >= MIN_ASYMPTOTE}
    for m in METRICS:
        p = [first_reaching(l, thr, m) for l in ppo_logs]
        q = [first_reaching(l, thr, m) for l in panel_logs]
        pm, qm = _median(p), _median(q)
        out[m] = {"ppo": pm, "panel": qm,
                  "ratio": pm / qm if pm is not None and qm else None,  # >1 means the panel is cheaper
                  "ppo_reached": sum(v is not None for v in p) / len(p),
                  "panel_reached": sum(v is not None for v in q) / len(q)}
    return out


def format_table(res: Dict) -> str:
    f = lambda v: "never" if v is None else f"{v:,.1f}" if isinstance(v, float) and v < 1000 else f"{v:,.0f}"
    lines = [f"PPO asymptote {res['asymptote']:.3f}; threshold ({res['frac']:.0%}) {res['threshold']:.3f}",
             f"{'metric':<10}{'ppo':>12}{'panel':>12}{'ppo/panel':>12}{'reached ppo/panel':>20}"]
    for m in METRICS:
        r = res[m]
        ratio = "-" if r["ratio"] is None else f"{r['ratio']:.2f}x"
        lines.append(f"{m:<10}{f(r['ppo']):>12}{f(r['panel']):>12}{ratio:>12}"
                     f"{r['ppo_reached']:>12.0%}/{r['panel_reached']:.0%}")
    ratio = res["env_steps"]["ratio"]
    if not res["valid"]:
        lines.append(f"H1 not measurable: PPO asymptote < {MIN_ASYMPTOTE}, i.e. the baseline has not learned this task")
        return "\n".join(lines)
    lines.append(f"H1 (>= {H1_SAMPLE_RATIO:.0f}x fewer samples than PPO): "
                 + ("not measurable (a method never reached the threshold)" if ratio is None
                    else "PASS" if ratio >= H1_SAMPLE_RATIO else "FAIL"))
    return "\n".join(lines)
