"""Rebuild the living H1 report from saved results, without touching active runs.

    python -m panel.analysis.h1_learnings

Narrative and proposed experiments live here; measurements, figures, provenance,
and the in-progress coverage matrix refresh from the local result files.
"""
from __future__ import annotations

import base64
from datetime import datetime
import hashlib
from html import escape
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panel.analysis.sweep_report import build_frame
from panel.analysis.algo_report import summarize

ROOT = Path(__file__).resolve().parents[2]
NAMES = {'MiniGrid-Empty-5x5-v0': 'Empty 5×5',
         'MiniGrid-Empty-Random-5x5-v0': 'Random-start Empty 5×5',
         'MiniGrid-Empty-8x8-v0': 'Empty 8×8',
         'MiniGrid-LavaGapS5-v0': 'LavaGap S5',
         'MiniGrid-DoorKey-5x5-v0': 'DoorKey 5×5'}
ORDER = list(NAMES)
COLORS = {'ppo': '#182e48', 'base': '#157b77', 'h8_j2': '#d47926', 'h2_j16': '#7956a3'}
LABELS = {'base': 'Base: 4 judges, H=2', 'h8_j2': '2 judges, H=8', 'h2_j16': '16 judges, H=2'}


def read_jsonl(path):
    # A live writer can leave an unfinished final line during this read.
    # Include only complete, newline-terminated records in the report snapshot.
    data = path.read_bytes() if path.exists() else b''
    lines = data.splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith(b'\n') and line.strip()], data


def late_return(rec):
    end = rec['rows'][-1]['env_steps']
    return float(np.mean([x['eval_return'] for x in rec['rows'] if x['env_steps'] >= .8 * end]))


def crossing(rec, threshold, key='env_steps'):
    return next((float(x[key]) for x in rec['rows'] if x['eval_return'] >= threshold), np.inf)


def number(value, digits=3):
    return '—' if not np.isfinite(value) else f'{value:,.{digits}f}'


def table(headers, rows, caption=None):
    body = ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows)
    cap = f'<caption>{escape(caption)}</caption>' if caption else ''
    return '<div class="table-wrap"><table>' + cap + '<thead><tr>' + ''.join(
        f'<th scope="col">{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + body + '</tbody></table></div>'


def figure(fig, name, assets, title, caption):
    fig.tight_layout()
    path = assets / f'{name}.png'
    fig.savefig(path, dpi=160, facecolor='#ffffff', bbox_inches='tight')
    plt.close(fig)
    src = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{src}" alt="{escape(title)}"><figcaption><b>{escape(title)}.</b> {caption} <a href="H1-learnings-assets/{name}.png">Download PNG</a></figcaption></figure>'


def build():
    reports = ROOT / 'reports'
    assets = reports / 'H1-learnings-assets'
    assets.mkdir(parents=True, exist_ok=True)
    nightly_path = ROOT / 'runs/nightly/results.jsonl'
    nightly, nightly_bytes = read_jsonl(nightly_path)
    old, thresholds = build_frame(nightly, {})
    good = {r['id']: r for r in nightly if r.get('rows') and 'error' not in r}
    old['late20'] = [late_return(good[i]) for i in old.id]
    old['late_above'] = [row.late20 >= thresholds[row.env][0] for row in old.itertuples()]
    old.to_csv(assets / 'nightly_per_run.csv', index=False)
    current_path = ROOT / 'runs/algo_sweep/results.jsonl'
    current, current_bytes = read_jsonl(current_path)
    current_frame, current_good = summarize(current)
    if len(current_frame):
        current_frame.to_csv(assets / 'algorithm_per_run.csv', index=False)
    manifest_path = ROOT / 'runs/algo_sweep/manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    now = datetime.now(ZoneInfo('America/Chicago'))
    stamp = now.strftime('%B %d, %Y · %I:%M %p Central')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold', 'axes.labelcolor': '#425469',
                         'text.color': '#182e48', 'axes.edgecolor': '#bdc9d3', 'grid.alpha': .25})

    env_rows, summary = [], []
    for env in ORDER:
        d = old[old.env == env]
        threshold, _, budget = thresholds[env]
        pp = [r for r in good.values() if r['env'] == env and r['kind'] == 'ppo']
        summary.append(dict(env=env, count=len(d), hit=int(d.to_thr.notna().sum()),
                            late=int(d.late_above.sum()), collapsed=int(d.collapsed.sum()),
                            threshold=threshold, budget=budget))
        verdict = {'MiniGrid-Empty-5x5-v0': 'Early sample savings; poor retention.',
                   'MiniGrid-Empty-Random-5x5-v0': 'Best current lead; retention still limited.',
                   'MiniGrid-Empty-8x8-v0': 'Inconsistent success; no late-average target attainment.',
                   'MiniGrid-LavaGapS5-v0': 'Transient success; frequent collapse.',
                   'MiniGrid-DoorKey-5x5-v0': 'No target attainment in this sweep.'}[env]
        env_rows.append([NAMES[env], f'{budget:,.0f}', f'{pp[0]["steps"]:,}', number(threshold),
                         f'{summary[-1]["hit"]}/{len(d)}', f'{summary[-1]["late"]}/{len(d)}',
                         f'{summary[-1]["collapsed"]}/{len(d)}', verdict])
    env_table = table(['Environment', 'Panel budget', 'PPO budget', 'Target return', 'Ever hit',
                       'Late mean ≥ target', 'Historical collapse', 'Interpretation'], env_rows)

    fig, ax = plt.subplots(figsize=(11.8, 4.5))
    pos = np.arange(len(summary))
    first = [s['hit']/s['count'] for s in summary]
    late = [s['late']/s['count'] for s in summary]
    ax.bar(pos-.18, first, .35, color='#157b77', label='Ever crossed target')
    ax.bar(pos+.18, late, .35, color='#d47926', label='Final-20% mean above target')
    for i, s in enumerate(summary):
        ax.text(i-.18, first[i]+.025, f"{s['hit']}/{s['count']}", ha='center', fontsize=9)
        ax.text(i+.18, late[i]+.025, f"{s['late']}/{s['count']}", ha='center', fontsize=9)
    ax.set_xticks(pos, [NAMES[s['env']].replace('Random-start ', 'Random-start\n') for s in summary])
    ax.set_ylim(0, 1.16)
    ax.set_ylabel('Fraction of panel runs')
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.legend(frameon=False, loc='upper left')
    retention_plot = figure(fig, '01_target_retention', assets, 'First crossing is not retained performance',
                            'All 182 completed panel runs, including random search and preset repetitions. Different configurations and budgets are pooled within each task. Late average is a diagnostic, not proof of sustained success.')

    presets, preset_rows = [], []
    for env in ORDER:
        threshold = thresholds[env][0]
        pp = [r for r in good.values() if r['env'] == env and r['kind'] == 'ppo']
        p_steps = np.median([crossing(r, threshold) for r in pp])
        p_time = np.median([crossing(r, threshold, 'wall_s') for r in pp])
        for group in LABELS:
            rs = [r for r in good.values() if r['env'] == env and r['group'] == group]
            steps = np.median([crossing(r, threshold) for r in rs])
            wall = np.median([crossing(r, threshold, 'wall_s') for r in rs])
            ratio = p_steps / steps if np.isfinite(steps) else np.nan
            vals = [late_return(r) for r in rs]
            auc = old[(old.env == env) & (old.group == group)].auc
            presets.append(dict(env=env, group=group, ratio=ratio, auc=float(auc.mean()),
                                auc_sd=float(auc.std()), late=float(np.mean(vals)), late_sd=float(np.std(vals, ddof=1))))
            preset_rows.append([NAMES[env], LABELS[group], str(len(rs)),
                                f'{sum(np.isfinite(crossing(r, threshold)) for r in rs)}/{len(rs)}',
                                number(steps, 0), number(ratio, 2) + ('×' if np.isfinite(ratio) else ''),
                                f'{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}',
                                number(wall, 1), number(p_time, 1)])
    preset_table = table(['Environment', 'Preset', 'Seeds', 'Ever hit', 'Median panel steps',
                          'PPO/panel samples', 'Late mean ± seed SD', 'Panel seconds to target', 'PPO seconds to target'], preset_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for j, group in enumerate(LABELS):
        ps = [p for p in presets if p['group'] == group]
        xx = np.arange(5) + (j-1)*.24
        axes[0].bar(xx, [p['auc'] for p in ps], .23, yerr=[p['auc_sd'] for p in ps],
                    color=COLORS[group], label=LABELS[group], capsize=2)
        axes[1].bar(xx, [p['late'] for p in ps], .23, yerr=[p['late_sd'] for p in ps],
                    color=COLORS[group], capsize=2)
    for a in axes:
        a.set_xticks(np.arange(5), ['Empty 5', 'Random 5', 'Empty 8', 'LavaGap', 'DoorKey'])
        a.grid(axis='y')
    axes[0].set(title='Learning over the panel sample budget', ylabel='Mean evaluation return over budget (AUC)')
    axes[1].set(title='Performance at the end', ylabel='Final-20% mean return')
    axes[0].legend(frameon=False, fontsize=8)
    preset_plot = figure(fig, '02_preset_comparison', assets, 'No preset wins across environments',
                         'Five seeds per preset and task; whiskers show seed standard deviation, not confidence intervals. Presets change several factors together, so this does not isolate judge count. AUC follows the existing report: 60-point interpolated mean return, with zero before the first evaluation.')

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4))
    for ax, env in zip(axes.flat, ORDER):
        for group in ['ppo', *LABELS]:
            rs = [r for r in good.values() if r['env'] == env and r['group'] == group]
            left = max(r['rows'][0]['env_steps'] for r in rs)
            right = min(r['rows'][-1]['env_steps'] for r in rs)
            x = np.geomspace(left, right, 160)
            ys = np.array([np.interp(x, [a['env_steps'] for a in r['rows']],
                                     [a['eval_return'] for a in r['rows']]) for r in rs])
            ax.plot(x, ys.mean(0), color=COLORS[group], label='PPO baseline (2 seeds)' if group == 'ppo' else LABELS[group])
            ax.fill_between(x, ys.mean(0)-ys.std(0), ys.mean(0)+ys.std(0), color=COLORS[group], alpha=.09)
        ax.axhline(thresholds[env][0], color='#9b4634', ls=':', lw=1)
        ax.set_xscale('log')
        ax.set(title=NAMES[env], xlabel='Real samples (log scale)', ylabel='Evaluation return', ylim=(-.12, 1.08))
        ax.grid(axis='y')
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc='center', frameon=False, fontsize=10)
    curves_plot = figure(fig, '03_historical_curves', assets, 'Learning curves expose the gap between an early peak and a usable policy',
                         'Mean ± one population seed SD; 5 panel seeds and 2 PPO seeds. Dotted lines are 80% of the PPO late-run reference. Curves end at each method’s actual budget; no panel performance is extrapolated to the PPO budget. Logarithmic sample axes make early gains visible.')

    long_rows, long_sources = [], []
    for path in sorted((ROOT / 'runs/h1_doorkey5').glob('*.json')):
        rec = json.loads(path.read_text())
        long_sources.append(path)
        ys = [r['eval_return'] for r in rec['rows']]
        long_rows.append([escape(path.stem), f'{rec["rows"][-1]["env_steps"]:,}', number(max(ys)),
                          number(ys[-1]), number(late_return(rec)), number(rec['rows'][-1]['wall_s'], 0)])
    long_table = table(['Saved run', 'Real samples', 'Peak', 'Last eval', 'Late mean', 'Training seconds'], long_rows)

    # Random-search marginals are descriptive, not controlled estimates of effects.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    random = old[old.group == 'random']
    for ax, env in zip(axes, [ORDER[0], ORDER[3], ORDER[4]]):
        d = random[random.env == env]
        for horizon, sub in d.groupby('imag_horizon'):
            ax.scatter(sub['n_judges'], sub['auc'], label=f'H={int(horizon)}', s=40,
                       alpha=.8, marker={1:'o', 2:'s', 4:'^', 8:'D'}[int(horizon)])
        ax.set(title=NAMES[env], xlabel='Number of judges', ylabel='AUC over panel budget', xticks=[2,4,8,16])
        ax.grid(axis='y')
    axes[0].legend(frameon=False, fontsize=8)
    scatter_plot = figure(fig, '04_judges_random_search', assets, 'More judges alone did not solve the hard tasks',
                          'Each point is one random-search run. Entropy, horizon, real-data mixing, and seed also vary. DoorKey remained below target for every configuration. This cannot establish an optimal judge count or a causal judge-count effect.')

    coverage_rows = []
    variants = [('baseline', 'ppo', 0), ('baseline', 'dqn', 0)] + [
        ('panel', algo, j) for j in [1,2,4] for algo in ['ppo','dqn','dyna']]
    for variant, algo, count in variants:
        row = [f'{algo.upper()} {variant}', str(count) if count else '—']
        for env in [ORDER[0], ORDER[3], ORDER[4]]:
            planned = [r for r in manifest if r['env'] == env and r['algo'] == algo
                       and r['variant'] == variant and r['n_judges'] == count]
            completed = sum(r['id'] in current_good for r in planned)
            row.append(f'{completed}/{len(planned)}')
        coverage_rows.append(row)
    coverage_table = table(['Method', 'Judges', 'Empty 5×5 seeds', 'LavaGap seeds', 'DoorKey seeds'], coverage_rows)
    live_rows = []
    for rec in current_good.values():
        live_rows.append([NAMES.get(rec['env'], rec['env']), rec['algo'].upper() + ' ' + rec['variant'],
                          str(rec['n_judges']) if rec['n_judges'] else '—', str(rec['seed']),
                          f'{rec["rows"][-1]["env_steps"]:,}', number(rec['rows'][-1]['eval_return']),
                          number(late_return(rec))])
    live_table = table(['Environment', 'Method', 'Judges', 'Seed', 'Actual samples', 'Last eval', 'Late mean'], live_rows)
    observations = []
    selected = [
        (ORDER[0], 'panel', 'ppo', 2, 0, 'A repeatable lead to check against the other seeds; not a fivefold first-crossing gain.'),
        (ORDER[0], 'panel', 'ppo', 2, 1, 'A second seed with a strong late mean; panel size and query behavior remain confounded.'),
        (ORDER[4], 'panel', 'dyna', 2, 0, 'High final score, but weaker late average. Reaching the target at about the PPO sample count is not an H1 efficiency win.'),
        (ORDER[0], 'baseline', 'dqn', 0, 0, 'Real-only replay can also alternate between success and failure; imagination is not required for instability.'),
        (ORDER[0], 'baseline', 'dqn', 0, 1, 'High final score does not establish a stable policy, even for the real-only baseline.'),
        (ORDER[3], 'panel', 'dqn', 4, 0, 'The last checkpoint overstates performance across the late training window.'),
    ]
    for env, variant, algo, count, seed, interpretation in selected:
        match = next((r for r in current_good.values() if r['env'] == env and r['variant'] == variant
                      and r['algo'] == algo and r['n_judges'] == count and r['seed'] == seed), None)
        if match:
            observations.append([NAMES[env], f'{algo.upper()} {variant}, J={count}, seed {seed}',
                                 number(match['rows'][-1]['eval_return']), number(late_return(match)), interpretation])
    observations_table = table(['Task', 'Completed run', 'Last eval', 'Late mean', 'What this changes'], observations)

    current_plot = ''
    seed_zero = [r for r in current_good.values() if r['seed'] == 0]
    if seed_zero:
        fig, axes = plt.subplots(1, 3, figsize=(13, 6.8), sharey=True)
        labels = ['PPO baseline', 'DQN baseline'] + [f'{a.upper()} J={j}' for j in [1,2,4] for a in ['ppo','dqn','dyna']]
        for ax, env in zip(axes, [ORDER[0], ORDER[3], ORDER[4]]):
            y = np.arange(len(variants))
            for i, (variant, algo, count) in enumerate(variants):
                match = next((r for r in seed_zero if r['env'] == env and r['variant'] == variant
                              and r['algo'] == algo and r['n_judges'] == count), None)
                if match:
                    ax.barh(i-.17, match['rows'][-1]['eval_return'], .32, color='#d47926', label='Last evaluation' if i == 0 else '')
                    ax.barh(i+.17, late_return(match), .32, color='#157b77', label='Final-20% mean' if i == 0 else '')
            ax.set(title=NAMES[env], xlabel='Evaluation return', xlim=(0, 1.02))
            ax.set_yticks(y, labels)
            ax.grid(axis='x')
        axes[0].invert_yaxis()
        axes[0].legend(frameon=False, fontsize=8, loc='lower right')
        current_plot = figure(fig, '05_current_seed0_retention', assets,
                              'New sweep: the last evaluation can hide instability',
                              'Seed 0 only, at matched real-sample budgets within each task. Orange is the last evaluation; teal is the mean over the final 20% of samples. Missing bars can represent zero or an unfinished job; consult the coverage table. These are exploratory outcomes, not replicated rankings.')

    evidence = table(['Question', 'Current answer', 'What would settle it?'], [
        ['Does H1 pass against PPO?', '<span class="badge amber">Partial evidence only</span> Early ≥5× crossings on small tasks; retention is weak.', 'Frozen configuration; independent seeds/layouts; sustained crossing and late-return checks; sample ratio with non-reaching seeds retained.'],
        ['Does H1 pass against Plan2Explore?', '<span class="badge gray">Untested</span>', 'Implement a matched Plan2Explore comparator. Dyna is not a substitute for this requirement.'],
        ['Does disagreement beat random queries?', '<span class="badge gray">Not isolated</span>', 'Same judge count, optimizer, update counts, starts, and budget; change only query selection.'],
        ['Does a panel beat one model?', '<span class="badge amber">Mixed / confounded</span>', 'Current 1/2/4 sweep is informative; add uniform-query multi-judge controls and match compute to separate ensemble size from acquisition.'],
        ['Do model updates help beyond replay?', '<span class="badge gray">New sweep pending</span>', 'Compare panel DQN vs Dyna, then equalize total policy-update count. Compare real-only DQN vs panel DQN for collection changes.'],
        ['Do error specialists add value?', '<span class="badge gray">Not implemented</span>', 'Four judges + specialist vs four judges trained longer and five ordinary judges; validate combined predictions and retained policy return.'],
        ['What is the cap on judge count?', '<span class="badge gray">Unknown</span>', 'Repeated marginal-gain tests at fixed data and compute. Failure of one candidate is not a capacity limit.'],
        ['Is the method cheaper overall?', '<span class="badge amber">Not established</span> Sample savings need not mean time savings.', 'Training time, full elapsed time, policy/model optimizer steps, and eventually FLOPs; include all model/query costs.'],
    ])
    methods = table(['Method in current sweep', 'Real data', 'Imagined policy updates', 'What its comparison tells us'], [
        ['Real-only PPO', 'Fresh on-policy trajectories', 'None', 'Original reference for the 80% target.'],
        ['Real-only Double DQN', 'Fresh trajectories + replay', 'None', 'Whether reuse of real experience is already a strong baseline.'],
        ['Panel PPO, J=1/2/4', 'Panel queries + 8 × 4 real rollouts per round', 'Horizon 1; target 50% imagined batch share', 'Combined PPO + model + acquisition behavior. Several settings differ from the historical sweep.'],
        ['Panel Double DQN, J=1/2/4', 'Panel queries + fresh trajectory; real replay', 'None', 'Real-only learning with panel-guided collection. It still pays judge-training costs.'],
        ['Dyna, J=1/2/4', 'Same collection schedule and real-update count as panel DQN', '24 extra one-step model updates per round', 'Incremental benefit of model updates plus their extra compute. Individual judges generate transitions; predictions remain continuous.'],
    ])
    ablations = table(['Priority', 'Comparison / intervention', 'Hold fixed', 'Question and interpretation'], [
        ['1 · finish current design', 'Complete 99 jobs; inspect all three seeds per cell', 'Saved manifest; task budgets; evaluation protocol', 'Which complete methods retain performance? Avoid choosing a winner from the first completed seed.'],
        ['2 · identify the gain', 'J=4 disagreement vs J=4 uniform queries', 'Algorithm, model fitting, real-data collection quota, sample budget', 'If disagreement does not help, narrow the claim to model/replay efficiency rather than panel-guided acquisition.'],
        ['2 · account for updates', 'Dyna: 24 real + 24 model updates vs DQN: 48 real updates', 'Real samples, collector, target-update schedule, architecture', 'If gains vanish, extra optimization may explain the apparent model benefit. The current sweep has only 24 real updates in panel DQN.'],
        ['3 · test model reliability', 'Categorical grid predictions / valid-state decoding; reward calibration; short-rollout control', 'Collector and policy training; change one modeling choice at a time', 'Test whether invalid states, reward error, or model compounding drives collapse. Short horizon alone is not proven sufficient.'],
        ['4 · test targeted correction', '4 judges + error specialist vs 4 trained longer vs 5 ordinary judges', 'Real dataset; candidate training compute; held-out validation', 'Does the candidate improve the combined panel on decision-relevant mistakes and improve retained policy return?'],
        ['5 · confirm H1', 'Freeze a finalist; run fresh seeds/layouts; add Plan2Explore', 'Declared task set, budgets, thresholds, sustained criterion', 'Estimate sample ratios, failure rates, and uncertainty without tuning on the final evaluation set.'],
    ])
    confounds = table(['Observation or design detail', 'Why it limits the conclusion'], [
        ['Historical random search is unfinished: 107/150 random configurations', '43 configurations were not saved. All 75 preset repetitions and 10 PPO baselines completed. Missing jobs have no saved outcome.'],
        ['Historical PPO reference: only 2 seeds/task; mean of final 20% of evaluations', 'This is an operational late-return proxy, not proof that PPO reached its true asymptote.'],
        ['Historical panel budgets are 4k–20k, PPO budgets 40k–300k', 'Not reaching the target is censored by the budget. DoorKey’s 20k panel budget is below the ~26k budget needed to test a 5× ratio against its ~130k PPO crossing. The two 100k panel runs provide additional negative evidence.'],
        ['Mean of last 3 evaluations vs final 20% of samples', 'These are different metrics. Earlier summaries used the last 3 for “final”; this report labels both definitions and uses the longer late window for retention.'],
        ['Ten fixed evaluation episodes per checkpoint', 'Limited evaluation precision; successive checkpoints are not independent observations. A success-rate confidence interval cannot be inferred from scalar return logs alone.'],
        ['Evaluation samples are uncharged', 'Training-sample accounting is consistent with the existing code, but repeated evaluation is not free in an expensive real environment. Report evaluation separately for deployment claims.'],
        ['New PPO uses horizon 1, 50% real mixing, and 25 judge updates', 'This is a new configuration, not a direct repetition of the historical presets (100 judge updates). Cross-sweep differences are not single-factor effects.'],
        ['Current J=1 uses uniform acquisition; J>1 uses disagreement', 'Judge count also changes acquisition behavior and training compute. Add same-J query ablations before attributing gains to diversity.'],
        ['PPO uses sampled evaluation; DQN/Dyna use greedy evaluation', 'These measure each deployed policy, but differing evaluation rules complicate optimizer-only comparisons. PPO baseline may overshoot by less than one 1,024-step collection batch.'],
        ['Dyna uses twice the policy updates of panel DQN', 'Current Dyna vs DQN changes both data source and optimization compute. An equal-update control is needed.'],
    ])

    source_paths = [nightly_path, current_path, manifest_path, *long_sources,
                    ROOT/'plan.md', ROOT/'panel/experiments/algo_sweep.py', ROOT/'panel/panel/trainer.py',
                    ROOT/'panel/agent/dqn.py', ROOT/'panel/judges/ensemble.py', Path(__file__)]
    provenance = []
    for path in source_paths:
        if path.exists():
            raw = nightly_bytes if path == nightly_path else current_bytes if path == current_path else path.read_bytes()
            provenance.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=len(raw),
                                   sha256=hashlib.sha256(raw).hexdigest()))
    snapshot = dict(generated_at=now.isoformat(), nightly_summary=summary, algorithm_completed=len(current_good),
                    algorithm_planned=len(manifest), sources=provenance)
    (assets/'snapshot.json').write_text(json.dumps(snapshot, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    sources_table = table(['Source', 'Snapshot bytes', 'SHA-256 prefix'], [
        [f'<a href="../{escape(p["path"])}">{escape(p["path"])}</a>', f'{p["bytes"]:,}', f'<code>{p["sha256"][:16]}</code>'] for p in provenance])
    failed = len({r['id'] for r in current if 'error' in r} - set(current_good))
    html = TEMPLATE.replace('@@STAMP@@', stamp).replace('@@CURRENT@@', str(len(current_good))).replace('@@PLANNED@@', str(len(manifest)))
    replacements = dict(EVIDENCE=evidence, ENV_TABLE=env_table, RETENTION=retention_plot,
                        PRESET_TABLE=preset_table, PRESET_PLOT=preset_plot, CURVES=curves_plot,
                        LONG_TABLE=long_table, SCATTER=scatter_plot, METHODS=methods,
                        COVERAGE=coverage_table, LIVE_TABLE=live_table, CURRENT_PLOT=current_plot,
                        OBSERVATIONS=observations_table, ABLATIONS=ablations,
                        CONFOUNDS=confounds, SOURCES=sources_table, FAILURES=str(failed),
                        HIST_LATE=str(int(old.late_above.sum())), HIST_PANEL=str(len(old)))
    for key, value in replacements.items():
        html = html.replace('@@'+key+'@@', value)
    target = reports/'H1-learnings.html'
    target.write_text(html, encoding='utf-8')
    print(f'{target}\nNightly: {len(old)} panel / 10 PPO; current sweep: {len(current_good)}/{len(manifest)} complete.')


TEMPLATE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>H1 learnings — evidence, components, and next decisions</title>
<style>
:root{--ink:#182e48;--muted:#526478;--bg:#f3f5f7;--line:#dce3e9;--teal:#157b77;--amber:#a25e13}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:75px}body{margin:0;color:var(--ink);background:var(--bg);font:15px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}
header{background:#182e48;color:white;padding:48px max(24px,calc((100vw - 1260px)/2)) 38px}header p{color:#d3dfeb;max-width:850px}h1{font-size:clamp(30px,4vw,49px);line-height:1.12;letter-spacing:-1.6px;margin:12px 0 18px}h2{font-size:27px;letter-spacing:-.7px;margin:0 0 12px;line-height:1.3}h3{font-size:18px;margin:25px 0 8px}.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:12px;font-weight:750}.updated{font-size:13px;color:#bed0e2}
nav{position:sticky;top:0;z-index:2;background:#fffefcf2;border-bottom:1px solid var(--line);padding:12px 24px;display:flex;gap:22px;justify-content:center;overflow-x:auto;white-space:nowrap}nav a{color:var(--ink);font-size:13px;font-weight:650;text-decoration:none}main{max-width:1308px;padding:28px 24px 80px;margin:auto}section{background:white;border:1px solid var(--line);border-radius:12px;padding:28px;margin:0 0 24px}p{max-width:950px;margin:10px 0 16px}a{color:#096a87}code{font-family:ui-monospace,Consolas,monospace;font-size:.9em;background:#edf1f5;padding:2px 5px;border-radius:4px}pre{padding:16px;background:#182e48;color:#e5edf4;border-radius:8px;overflow:auto}pre code{background:none;color:inherit;padding:0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:22px 0}.card{background:#f5f8fa;border:1px solid var(--line);padding:20px;border-radius:9px}.card strong{display:block;font-size:30px;line-height:1.2;margin-bottom:8px}.card span{font-size:13px;color:var(--muted)}.callout{border-left:4px solid var(--teal);background:#edf7f5;padding:16px 20px;margin:20px 0}.callout.warn{border-color:#c17c2c;background:#fff6e9}.small,figcaption{font-size:13px;color:var(--muted)}.badge{display:inline-block;font-size:11px;padding:3px 7px;border-radius:4px;font-weight:700;white-space:nowrap;margin-right:5px}.amber{background:#fff0d8;color:#89520b}.gray{background:#eaf0f5;color:#496078}.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:7px;margin:18px 0}table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.5;font-variant-numeric:tabular-nums}th{text-align:left;background:#edf2f6;color:#3c536b;font-size:11px;letter-spacing:.3px}th,td{padding:12px 13px;vertical-align:top;border-bottom:1px solid #e6ebef;min-width:90px}td:first-child{font-weight:650}tbody tr:last-child td{border-bottom:0}tbody tr:nth-child(even){background:#fafbfd}caption{text-align:left;padding:12px;font-weight:650}figure{margin:24px 0}figure img{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:8px}figcaption{margin-top:9px;max-width:1100px}.flow{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}.flow div{padding:17px;background:#f0f5f8;border-top:3px solid #157b77;border-radius:5px}.flow b{display:block;margin-bottom:5px}.flow span{font-size:13px;color:var(--muted)}li{margin:7px 0;max-width:950px}details{border-top:1px solid var(--line);padding:15px 0;margin-top:20px}summary{cursor:pointer;font-weight:700}footer{font-size:12px;color:var(--muted);padding:0 8px}.tag{display:inline-block;margin-right:8px;padding:4px 9px;border:1px solid #698198;border-radius:4px;font-size:11px;color:#d8e7f5}
@media(max-width:760px){section{padding:20px 16px}main{padding:16px 12px}.cards,.flow{grid-template-columns:1fr}header{padding:32px 20px}nav{justify-content:flex-start;gap:18px}th,td{padding:10px;min-width:110px}h2{font-size:24px}}
@media print{nav{display:none}body{background:white}main{max-width:none;padding:0}section{break-inside:auto;border:0;padding:15px 0}figure{break-inside:avoid}header{padding:20px;background:white;color:#182e48}header p,.updated{color:#425469}.table-wrap{overflow:visible}table{font-size:10px}th,td{min-width:0;padding:6px}a{color:inherit}}
</style></head><body>
<header><div class="eyebrow">Research notebook / hypothesis 01</div><h1>Where the panel helps.<br>Where the evidence stops.</h1>
<p>H1 asks whether judge-guided learning buys substantially more return per real environment sample. We have an early signal on small navigation tasks, a retention problem, and no confirmed result on the harder tasks.</p>
<div><span class="tag">Living report</span><span class="tag">H1 remains open</span><span class="tag">Measured ≠ inferred</span></div><p class="updated">Snapshot: @@STAMP@@ · Rebuild from saved logs to update.</p></header>
<nav aria-label="Report sections"><a href="#verdict">Answer so far</a><a href="#evidence">Measured results</a><a href="#components">What works together</a><a href="#current">Current sweep</a><a href="#protocol">How to answer H1</a><a href="#next">Next decisions</a><a href="#provenance">Sources & updates</a></nav>
<main>
<section id="verdict"><div class="eyebrow">01 / Decision status</div><h2>Keep H1. Do not call it confirmed.</h2>
<div class="callout"><b>Original H1, preserved from <a href="../plan.md">plan.md</a>:</b> Panel-guided querying reaches 80% of the PPO asymptote with ≥5× fewer environment samples than PPO, and ≥2× fewer than Plan2Explore.</div>
<p>We are still working on that hypothesis. The current algorithm sweep is diagnostic: it tests which complete methods learn and retain performance, and which environment/model/optimizer combinations deserve confirmation. It does not yet establish that disagreement-based querying is the source of any gain.</p>
<div class="cards"><div class="card"><strong>7.7–10.7×</strong><span>Historical base preset’s median sample savings at the first target crossing, on the two small navigation tasks. Five panel seeds; two PPO seeds. Not a retained-performance result.</span></div><div class="card"><strong>@@HIST_LATE@@ / @@HIST_PANEL@@</strong><span>Historical panel runs whose final-20%-of-samples mean return met their task’s target. Pool includes different configurations; this is a warning about retention, not an estimate for one method.</span></div><div class="card"><strong>Untested</strong><span>The ≥2× Plan2Explore requirement. A DQN or Dyna comparison cannot satisfy that part of the original hypothesis.</span></div></div>
@@EVIDENCE@@
<p class="small">The original hypothesis does not specify a sustained-performance rule. The protocol below proposes one explicitly; historical first-crossing results remain labeled under their original definition.</p></section>

<section id="evidence"><div class="eyebrow">02 / Completed historical evidence</div><h2>Small tasks show early efficiency. The gain often disappears.</h2>
<p>The October 28 nightly sweep saved <b>192 of 235 planned jobs</b>: 182 panel runs and 10 PPO baselines, with no recorded failures. All 75 named-preset repetitions finished; 43 random-search jobs have no saved outcome. The file ended at 10:57 p.m. Central, before the configured eight-hour window elapsed. The cause of the stop is not in the saved results.</p>
@@ENV_TABLE@@
<p class="small"><b>Ever hit:</b> any evaluation ≥80% of the task’s PPO late-return reference. <b>Late mean:</b> average of recorded evaluations at ≥80% of that run’s final sample count. <b>Historical collapse:</b> peak &gt;0.1 and mean of the last three evaluations &lt;30% of peak, matching the original sweep report. These are different diagnostics. Counts pool five-seed presets and single-seed random configurations.</p>
@@RETENTION@@
<h3>Replication matters more than the best random-search run</h3>
<p>The original two-judge, long-horizon pilot winner was not a universal winner across seeds. The four-judge base preset has the clearest early efficiency signal on small tasks. Sixteen judges improve some Empty 8×8 and LavaGap outcomes, but do not rescue DoorKey and add substantial training cost.</p>
@@PRESET_PLOT@@
<details><summary>All 15 preset/task comparisons: first crossing, late return, and training time</summary>
<p class="small">Base = entropy 0.01, horizon 2, 4 judges; H8/J2 = entropy 0.01, horizon 8, 2 judges; H2/J16 = entropy 0.003, horizon 2, 16 judges. All use imagination-only policy training. A seed that never crosses counts as infinity in the median; “—” means no finite median crossing or ratio. Training seconds exclude evaluation and are not FLOPs.</p>
@@PRESET_TABLE@@</details>
<p><b>Samples are not time:</b> on Empty 5×5 the historical base preset first crosses at a median 1,152 real samples versus PPO’s 12,288, but takes about 30.6 training seconds versus 10.9. The method trades extra computation for real samples; the trade only helps when environment queries are sufficiently expensive.</p>
@@CURVES@@
<h3>DoorKey’s longer run is additional negative evidence</h3>
<p>Two panel seeds completed 100,000 real samples each and finished at zero return. Three PPO seeds completed about 300,000 samples each and finished near 0.65–0.75. The third panel seed is missing. This supports “the current configuration fails here”; it does not establish that all judge-based methods fail on DoorKey.</p>
@@LONG_TABLE@@
@@SCATTER@@
<div class="callout warn"><b>Correction to the earlier pilot report:</b> <a href="H1_01_collapse_eval.html">H1_01_collapse_eval.html</a> is a single-seed historical note, not the current conclusion. Its “horizon is the lever” conclusion was exploratory. Empty tasks also use sparse terminal reward; describing Empty 5×5 as a dense-reward task was inaccurate. The broader evidence supports easier navigation versus harder interaction/hazard tasks, without identifying a single causal property.</div></section>

<section id="components"><div class="eyebrow">03 / Mechanism map</div><h2>The panel is a chain of components, not one intervention.</h2>
<div class="flow"><div><b>1. Collect truth</b><span>Real trajectories and oracle queries supply verified transitions. No model can prioritize a successful interaction it has never seen unless exploration reaches it.</span></div><div><b>2. Fit the panel</b><span>Bootstrapped dynamics/reward/termination models learn from real data; anchor errors determine competence weights.</span></div><div><b>3. Spend samples</b><span>Disagreement selects candidate state/action queries, mixed with random anchors. One judge has no ensemble variance.</span></div><div><b>4. Improve behavior</b><span>PPO learns from rollout batches; DQN learns from real replay; Dyna adds model-generated transitions. Real evaluation checks the result.</span></div></div>
<p>A method can fail at any link: insufficient exploration, shared blind spots, a poor query score, invalid imagined states, misleading reward predictions, or an optimizer exploiting those predictions. Better average prediction error does not automatically mean better decisions.</p>
@@METHODS@@
<h3>Most plausible failure mechanisms to test—not established causes</h3>
<ul><li><b>Invalid imagined states:</b> current models regress numeric encodings of categorical grid objects. Panel PPO averages next-state predictions and feeds the result back into the policy. A numerical average can be an impossible grid. Dyna samples a judge instead of averaging, but that judge still predicts continuous grids.</li>
<li><b>Reward optimism:</b> panel PPO clips imagined reward to [0,1]. Around true zero reward, positive prediction errors survive while negative errors are removed. Measure whether this creates exploitable reward. Dyna’s imagined reward uses [-1,1], so cross-algorithm model handling is not identical.</li>
<li><b>Shared errors:</b> bootstrapped judges can agree and still be wrong. Disagreement can miss a common blind spot; scalar error correlation is only a diagnostic, not a causal test of diversity.</li>
<li><b>Model exploitation and long rollouts:</b> the policy can visit prediction errors outside training support. Short rollouts and real updates may help, but the historical sweep does not prove that either is sufficient.</li></ul>
<h3>Where the proposed error specialist fits</h3>
<p>Score the existing panel on newly verified transitions <em>before</em> training on them. Bin mistakes by movement, turning, pickup, door interaction, reward, and termination; train a candidate on a mixture of ordinary examples and those errors. Validate on separate real transitions. Admit the candidate only when the <em>combined</em> panel improves, including on the events that matter for reward.</p>
<p>Compare a specialist against both “four judges trained longer” and “five ordinary judges.” A global average may dilute a specialist, so its routing or correction rule also needs validation. Deliberate specialization changes the meaning of disagreement; do not treat specialist disagreement as automatically calibrated uncertainty.</p>
<p><b>A practical cap:</b> stop adding judges when several candidate seeds fail to improve held-out panel predictions and retained policy return enough to justify compute. That is a stopping rule for a particular architecture, dataset, and budget—not a universal maximum number of judges.</p></section>

<section id="current"><div class="eyebrow">04 / New sweep in progress</div><h2>@@CURRENT@@ / @@PLANNED@@ jobs have saved results.</h2>
<p>Three algorithms × three tasks × three judge counts × three seeds = 81 panel runs, plus 18 real-only baselines. Intended budgets are 40,000 / 150,000 / 300,000 real samples for Empty / LavaGap / DoorKey. Five workers run jobs from the saved manifest; a time limit stops new starts, and completed jobs can be resumed. This report reads only completed records; it does not inspect partial training inside running workers.</p>
<div class="callout warn"><b>Preliminary snapshot.</b> The completion order is not random. Incomplete seed coverage must not be ranked as a definitive algorithm or environment result. Unresolved saved failures: @@FAILURES@@. The old sweep and this new design are reported separately.</div>
@@COVERAGE@@
@@CURRENT_PLOT@@
<h3>Signals worth following in the completed cells</h3>
@@OBSERVATIONS@@
<p>Instability in real-only DQN means imagined-model errors cannot explain every failure. Panel DQN also changes where real data comes from: about two-thirds of each collection round queries stored states rather than advancing the current trajectory. Reduced exploration or changed replay coverage is another explanation to test. Neither mechanism is isolated by the current design.</p>
<details><summary>Completed new-sweep runs at this snapshot</summary>@@LIVE_TABLE@@</details>
<p>Useful comparisons once matched cells finish: panel DQN versus Dyna for added model updates; real-only DQN versus panel DQN for the combined acquisition/data-collection change; J=1/2/4 within a method for the practical panel-size tradeoff. None alone isolates every mechanism.</p></section>

<section id="protocol"><div class="eyebrow">05 / Decision protocol</div><h2>Define the answer before picking a winner.</h2>
<p><b>Preserve the original claim.</b> Report its first-crossing sample ratio separately. Do not change H1 to match whichever configuration wins. Add retained performance as a declared robustness requirement for the next confirmation experiment.</p>
<ol><li><b>Freeze tasks, settings, and budgets.</b> Keep Empty 5×5 as a positive control and LavaGap/DoorKey as harder tests. Select a candidate using exploratory data, then use fresh training seeds and fresh final-evaluation layouts. These three tasks can support a bounded MiniGrid claim, not the whole planned Procgen/generalization scope.</li>
<li><b>Establish a learned reference.</b> For each task, estimate PPO’s late return across independent seeds and inspect whether it has stabilized. A final-20% average is a proxy. If the baseline has not learned, H1 is not measurable on that task; do not substitute a random-search peak for a confirmatory threshold.</li>
<li><b>Use one target for all methods.</b> Let T = 0.8 × the frozen PPO reference. Report the original first crossing. Proposed retained crossing: first of five consecutive evaluations at or above T, followed by a final-20%-of-samples mean at or above T. Five checks are an operational rule, not a statistical proof; pick a common evaluation interval before running confirmation.</li>
<li><b>Retain non-reaching seeds.</b> Treat their crossing times as right-censored/infinite for the median; show reached/attempted and the full per-seed data. Never calculate the headline median using only successful runs. Report uncertainty from independent training seeds, not from checkpoints treated as independent samples.</li>
<li><b>Compute the two required ratios.</b> PPO samples / panel samples ≥5, and Plan2Explore samples / panel samples ≥2, at the same target. Require the retained-performance result alongside the original first-crossing result. If a comparator or panel never reaches the target, report censoring instead of a fabricated finite speedup.</li>
<li><b>Separate mechanism and cost claims.</b> Run same-J disagreement-versus-uniform acquisition and equal-policy-update DQN/Dyna controls. Report real samples, evaluation samples, model/policy updates, training time, full elapsed time, and eventually FLOPs.</li></ol>
<div class="callout"><b>Current reporting status:</b> the sustained rule above is a proposal, not a retroactive pass/fail label. The tables use the existing first-crossing definition and explicitly labeled late-average diagnostics. No completed experiment yet meets the full confirmation protocol.</div>
<h3>Interpretation rules</h3><ul><li><b>Works only on Empty:</b> report a restricted navigation result; the broad H1 remains unsupported.</li><li><b>DQN alone matches the gain:</b> replay is a sufficient alternative explanation until the panel beats it.</li><li><b>Dyna helps but disagreement does not:</b> support a model-assisted learning result; do not attribute it to judge-guided querying.</li><li><b>First crossing improves but late return collapses:</b> report a transient efficiency effect, not reliable learning.</li><li><b>Several judges help at matched samples but not matched compute:</b> report the explicit compute-for-samples tradeoff.</li></ul></section>

<section id="next"><div class="eyebrow">06 / Next work</div><h2>Complete coverage, isolate the gain, then confirm it.</h2>@@ABLATIONS@@
<p>Prioritize diagnosis over finishing the 43 historical random configurations. Complete the current design or transparently report time-censored coverage. Add the controls needed for the most promising matched comparison before expanding environments or increasing judge counts.</p>
<details><summary>Limitations and confounds to keep visible in future updates</summary>@@CONFOUNDS@@</details></section>

<section id="provenance"><div class="eyebrow">07 / Reproducibility and update log</div><h2>One report, refreshed from saved evidence.</h2>
<p>This HTML contains its figures and needs no network connection. Standalone PNGs and per-run CSVs are in <a href="H1-learnings-assets/snapshot.json">H1-learnings-assets</a>. The snapshot records source hashes so results can be tied to the files read by the generator.</p>
<pre><code>python -m panel.analysis.h1_learnings</code></pre>
<p>Run from the repository root after more jobs finish. The generator refreshes measurements, plots, current coverage, and timestamps. It reads complete JSONL records without modifying the running experiment. Edit <a href="../panel/analysis/h1_learnings.py">panel/analysis/h1_learnings.py</a> when adding new interpretations, datasets, or experiment stages; regeneration alone does not perform scientific review.</p>
<p><a href="H1-learnings-assets/nightly_per_run.csv">Historical per-run metrics</a> · <a href="H1-learnings-assets/algorithm_per_run.csv">New-sweep per-run metrics</a> · <a href="../runs/algo_sweep/console.log">Current runner log</a> · <a href="../run.sh">Sweep launcher</a> · <a href="H1_01_collapse_eval.html">Earlier pilot note</a></p>
<h3>Evidence ledger</h3>
<ul><li><b>October 28, 2026:</b> pilot and historical nightly sweep. Early small-task gains; incomplete random-search coverage; repeated harder-task failure.</li><li><b>October 29, 2026:</b> three-algorithm sweep launched; standalone DQN Empty 5×5 seed 0 learning check reached 0.955 final return at 40k samples. This development check is outside the confirmatory dataset.</li><li><b>This report:</b> separates first crossing, historical last-three collapse, and final-20% retention; preserves original H1 and proposes a prospective sustained-performance rule.</li></ul>
<details><summary>Source inventory and snapshot fingerprints</summary>@@SOURCES@@</details></section>
<footer>Generated @@STAMP@@. Conclusions apply to the saved implementations, budgets, tasks, and seeds shown here. Update the evidence and its interpretation together.</footer>
</main></body></html>'''


if __name__ == '__main__':
    build()
