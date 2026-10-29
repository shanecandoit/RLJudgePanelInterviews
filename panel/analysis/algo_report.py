"""Per-seed results and aggregate learning curves for the algorithm sweep."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panel.experiments.algo_sweep import load_records, successful_ids


def summarize(records):
    # Retries replace earlier failures; do not count successful runs twice.
    good = {r['id']: r for r in records if r.get('rows') and 'error' not in r}
    refs = {}
    for env in {r['env'] for r in good.values()}:
        values = []
        for r in good.values():
            if r['env'] == env and r['algo'] == 'ppo' and r['variant'] == 'baseline':
                rows = r['rows']
                cutoff = .8 * rows[-1]['env_steps']
                values.append(np.mean([x['eval_return'] for x in rows if x['env_steps'] >= cutoff]))
        refs[env] = float(np.mean(values)) if values else float('nan')
    result = []
    for r in good.values():
        rows = r['rows']
        x = np.array([a['env_steps'] for a in rows])
        y = np.array([a['eval_return'] for a in rows])
        late = y[x >= .8 * x[-1]]
        ref = refs[r['env']]
        threshold = .8 * ref if ref >= .1 else float('nan')
        hits = x[y >= threshold]
        result.append(dict(id=r['id'], env=r['env'], algo=r['algo'], variant=r['variant'],
                           judges=r['n_judges'], seed=r['seed'], samples=int(x[-1]),
                           auc=float(np.interp(np.linspace(0, r['steps'], 200), x, y, left=0).mean()),
                           peak=float(y.max()), late_return=float(late.mean()),
                           collapsed=bool(y.max() > .1 and late.mean() < .3 * y.max()),
                           threshold=threshold, reached=bool(len(hits)),
                           to_threshold=float(hits[0]) if len(hits) else float('nan'),
                           late_above_threshold=bool(late.mean() >= threshold),
                           train_s=rows[-1]['wall_s'], elapsed_s=r.get('elapsed_s', float('nan'))))
    return pd.DataFrame(result), good


def report(out):
    out = Path(out)
    records = load_records(out / 'results.jsonl')
    frame, good = summarize(records)
    manifest = out / 'manifest.json'
    import json
    planned = len(json.loads(manifest.read_text())) if manifest.exists() else None
    lines = [f'{len(good)} successful / {planned if planned is not None else "?"} planned; '
             f'{len({r["id"] for r in records if "error" in r} - successful_ids(records))} unresolved failures',
             'Threshold: 80% of the real-only PPO late return; unavailable if PPO return < 0.1.',
             'Late return: mean over the final 20% of recorded environment steps.',
             'DQN evaluation is greedy; PPO evaluation samples its policy. Evaluation samples are uncharged.',
             'First crossings are exploratory; sustained performance and seed variability matter.']
    if len(frame):
        frame.to_csv(out / 'per_seed.csv', index=False)
        table = frame.groupby(['env', 'variant', 'algo', 'judges']).agg(
            seeds=('id', 'size'), auc=('auc', 'mean'), late_return=('late_return', 'mean'),
            late_sd=('late_return', 'std'), peak=('peak', 'mean'), reached=('reached', 'mean'),
            threshold=('threshold', 'first'),
            sustained=('late_above_threshold', 'mean'), collapsed=('collapsed', 'mean'),
            train_s=('train_s', 'median'), elapsed_s=('elapsed_s', 'median'))
        table.to_csv(out / 'summary.csv')
        lines.append(table.to_string(float_format=lambda x: f'{x:.3f}'))
        for env, d in frame.groupby('env'):
            fig, ax = plt.subplots(figsize=(11, 6))
            for (variant, algo, judges), group in d.groupby(['variant', 'algo', 'judges']):
                budget = min(good[i]['steps'] for i in group.id)
                grid = np.linspace(0, budget, 150)
                ys = []
                for run_id in group.id:
                    rows = good[run_id]['rows']
                    ys.append(np.interp(grid, [r['env_steps'] for r in rows],
                                        [r['eval_return'] for r in rows], left=0))
                ys = np.array(ys)
                label = f'{algo} {variant}' + (f' J={judges}' if judges else '')
                line, = ax.plot(grid, ys.mean(0), label=label, ls='--' if variant == 'baseline' else '-')
                if len(ys) > 1:
                    ax.fill_between(grid, ys.mean(0)-ys.std(0), ys.mean(0)+ys.std(0),
                                    color=line.get_color(), alpha=.08)
            ax.set(title=env + ' (mean +/- seed SD)', xlabel='charged environment samples', ylabel='evaluation return')
            ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(out / f'{env}_curves.png', dpi=140)
            plt.close(fig)
    text = '\n\n'.join(lines)
    (out / 'summary.txt').write_text(text + '\n')
    return text


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    print(report(ap.parse_args().out))
