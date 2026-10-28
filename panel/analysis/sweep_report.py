"""Tables and scatter plots for a random-search sweep (results.jsonl from panel.experiments.h1_sweep).

    python -m panel.analysis.sweep_report runs/sweep [--ref MiniGrid-DoorKey-5x5-v0=0.7]

--ref gives a PPO asymptote per env; the H1 threshold is 80% of it. Without one, the threshold is 80% of the
90th-percentile peak across the sweep for that env (printed as "relative"), which only ranks configs.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FACTORS = ("ent_coef", "imag_horizon", "n_judges", "imag_frac")
DIAG = ("calib", "rank_corr", "n_eff")
HORIZON_COLORS = {1: "#2a78d6", 2: "#1baf7a", 4: "#eda100", 8: "#eb6834"}
pd.set_option("display.width", 200, "display.max_columns", 30, "display.float_format", lambda v: f"{v:.3g}")


def load(out_dir: str) -> List[dict]:
    with open(os.path.join(out_dir, "results.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


def run_metrics(rec: dict, thr: float, budget: float) -> dict:
    rows = rec["rows"]
    x, y = np.array([r["env_steps"] for r in rows], float), np.array([r["eval_return"] for r in rows], float)
    grid = np.linspace(0, budget, 60)
    peak, final = float(y.max()), float(y[-3:].mean())
    hit = np.nonzero(y >= thr)[0]
    after = lambda k: float(np.nanmedian([r[k] for r in rows[3:]] or [np.nan])) if len(rows) > 3 else np.nan
    return {"id": rec["id"], "env": rec["env"], **rec["config"],
            "auc": float(np.interp(grid, x, y, left=0.0).mean()), "peak": peak, "final": final,
            "collapsed": bool(peak > 0.1 and final < 0.3 * peak),
            "to_thr": float(x[hit[0]]) if len(hit) else np.nan, "wall_s": float(rows[-1]["wall_s"]),
            **{k: after(k) for k in DIAG}}


def build_frame(recs: List[dict], refs: Dict[str, float]):
    df, thresholds = [], {}
    for env in sorted({r["env"] for r in recs}):
        er = [r for r in recs if r["env"] == env and "rows" in r]
        budget = max(r["rows"][-1]["env_steps"] for r in er)
        if env in refs:
            thr, how = 0.8 * refs[env], f"80% of PPO asymptote {refs[env]:.2f}"
        else:
            thr = 0.8 * float(np.percentile([max(x["eval_return"] for x in r["rows"]) for r in er], 90))
            how = "relative: 80% of the sweep's 90th-percentile peak (no --ref)"
        thresholds[env] = (thr, how, budget)
        df += [run_metrics(r, thr, budget) for r in er]
    return pd.DataFrame(df), thresholds


def marginals(d: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for f in FACTORS:
        g = d.groupby(f).agg(n=("auc", "size"), auc=("auc", "mean"), peak=("peak", "mean"),
                             reached=("to_thr", lambda s: s.notna().mean()), to_thr_med=("to_thr", "median"),
                             collapsed=("collapsed", "mean"), wall_s=("wall_s", "median"),
                             calib=("calib", "median"), rank_corr=("rank_corr", "median"), n_eff=("n_eff", "median"))
        g.insert(0, "factor", f)
        g.index.name = "value"
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def scatter(d: pd.DataFrame, env: str, budget: float, thr: float, path: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    a = ax[0]
    for h, g in d.groupby("imag_horizon"):
        xs = g["to_thr"].fillna(budget * 1.12)
        a.scatter(xs, g["wall_s"], s=18 + 9 * g["n_judges"], c=HORIZON_COLORS.get(int(h), "gray"), alpha=.8,
                  edgecolors=np.where(g["collapsed"], "#d03b3b", "none"), linewidths=1.8, label=f"horizon {int(h)}")
    a.axvline(budget * 1.12, color="#c3c2b7", ls=":")
    a.set_xlabel(f"env samples to reach return {thr:.2f}  (right edge = never)")
    a.set_ylabel("training wall time (s)")
    a.set_title("Cost to threshold. size = judges, red ring = collapsed")
    a.legend(frameon=False, fontsize=8)
    b = ax[1]
    sc = b.scatter(d["calib"], d["auc"], s=22 + 6 * d["n_judges"], c=np.log2(d["n_judges"]), cmap="viridis", alpha=.85)
    b.axvline(1.0, color="#c3c2b7", ls=":")
    b.set_xscale("log")
    b.set_xlabel("panel calibration = disagreement / true error  (1 = calibrated, << 1 = judges too alike)")
    b.set_ylabel("AUC of eval return")
    b.set_title("Do calibrated panels learn faster?")
    cb = fig.colorbar(sc, ax=b)
    cb.set_label("log2(judges)")
    fig.suptitle(env)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def report(out_dir: str, refs: Optional[Dict[str, float]] = None, top: int = 10) -> str:
    recs = load(out_dir)
    errs = [r for r in recs if "error" in r]
    df, thr = build_frame(recs, refs or {})
    lines = [f"{len(df)} runs ({len(errs)} failed)"]
    cols = ["id", *FACTORS, "auc", "peak", "final", "to_thr", "wall_s", "collapsed", "calib", "rank_corr", "n_eff"]
    for env, (t, how, budget) in thr.items():
        d = df[df["env"] == env]
        lines += ["", f"=== {env}  budget {budget:,.0f} samples; threshold {t:.2f} ({how}) ===",
                  f"reached threshold: {d['to_thr'].notna().mean():.0%}   collapsed: {d['collapsed'].mean():.0%}",
                  "", f"-- top {top} runs by AUC", d.sort_values("auc", ascending=False).head(top)[cols].to_string(index=False),
                  "", "-- marginals (mean over all other factors; calib/rank_corr/n_eff are medians)",
                  marginals(d).to_string(index=False)]
        ok = d.dropna(subset=["rank_corr"])
        if len(ok) > 4:
            lines += ["", "-- Spearman across runs: " + ", ".join(
                f"{k} vs auc {ok[k].rank().corr(ok['auc'].rank()):+.2f}" for k in DIAG)]
        scatter(d, env, budget, t, os.path.join(out_dir, f"sweep_{env.split('-', 1)[-1]}.png"))
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "sweep_tables.txt"), "w") as f:
        f.write(text + "\n")
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--ref", action="append", default=[], help="ENV=ppo_asymptote, repeatable")
    a = ap.parse_args()
    print(report(a.out_dir, {k: float(v) for k, v in (s.split("=") for s in a.ref)}))
