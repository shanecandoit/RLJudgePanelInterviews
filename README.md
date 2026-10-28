# Panel: Judge-Panel RL for Sample-Efficient World Models and Honesty Verification

A policy must answer "what would you do in state *s*?" to a diverse, self-maintaining panel of judge models. The judges choose the states where they disagree. Real environment samples are spent only where that disagreement predicts error. The world model, the plan validator, and the honesty signal all come out of this one loop.

## Core claim

When environment samples are expensive, routing ground-truth queries to the states where a **diverse, calibrated** judge ensemble disagrees reaches a target return with far fewer environment samples than standard model-free or curiosity-driven baselines. The same machinery produces a stated-vs-revealed **honesty metric** that is hard to game, as long as judges are anchored to ground truth and firewalled from the policy.

## Relation to prior work

| Idea | Source | What we take / change |
|---|---|---|
| AI Debate | Irving et al. 2018 | Honesty via adversarial scrutiny. We use judges that *predict* outcomes, not persuasion contests |
| Prover-Verifier Games | Kirchner et al. 2024 | Legibility pressure on the prover |
| Query-by-Committee | Seung et al. 1992 | Disagreement → query selection |
| Plan2Explore / ensemble disagreement | Sekar et al. 2020 | Epistemic uncertainty from ensembles |
| League training | AlphaStar, Vinyals et al. 2019 | Hall of fame and exploiters, used to prevent strategy cycling |
| Dawid-Skene / IRT | Dawid & Skene 1979 | Estimating judge competence without full labels |

**Candidate novelty:** (a) a judge population with redundancy-based turnover, (b) a learned acquisition head P(error | s), and (c) honesty measured as stated-vs-revealed consistency against ground truth.

## Roles

- **Agent:** policy π(a|s) and value Q(s,a).
- **Judges (N ≈ 8):** each has a world model plus its own value estimate. Judges differ in architecture, seed, and data subset.
- **Audit panel:** frozen judges that are never trained against and never exposed to coaches or the agent.
- **Hall of fame:** retired judges and agents, kept for regression checks and never deleted.
- **Environment oracle:** ground truth for anchor queries.

## Prediction heads per judge

| # | Head | Output | Used for |
|---|---|---|---|
| 1 | Dynamics | p(s′ \| s,a) | World model; disagreement signal |
| 2 | Reward / outcome | r(s,a), P(success \| plan) | Plan validation; action scoring |
| 3 | Agent-action prediction | p(a_agent \| s) | Query informativeness; deception drift |
| 6 | Self-calibration | P(I'm wrong \| s) | Competence weights |
| 5 | Inverse dynamics | p(a \| s,s′) | Verifying the agent's claims about its actions |
| 7 | Peer prediction | p(verdict_B \| s) | Redundancy measurement for retirement |
| 8 | Acquisition | P(disagreement or error \| s) | Learned query selection |

**Minimum viable set:** heads 1, 2, 3 and 6.

## Loop (one round)

1. The judges or the acquisition head propose a batch of states with high predicted disagreement.
2. The agent commits to an action and a short stated plan. It does not know which judge will score it.
3. Each judge scores the action as Q_judge(s,a) and rolls the plan forward in its own model.
4. A fraction of queries (the **anchors**, about 15%) plus the top-disagreement queries are resolved in the environment.
5. Updates:
   - judges via a proper scoring rule on resolved items,
   - competence weights via Dawid-Skene/IRT on anchors only,
   - the agent via RL on competence-weighted judge Q plus environment reward on resolved items,
   - a consistency penalty when the stated plan diverges from revealed behavior.
6. Every K rounds, run the turnover check described below.

## Safeguards against reward hacking and whipsaw

1. **Ground-truth anchor:** judge competence is estimated only on items the environment can verify.
2. **Frozen audit panel:** no gradients ever flow against it.
3. **Proper scoring rules:** judges are rewarded for calibration, not for agreeing with each other.
4. **Error decorrelation:** penalize pairwise error correlation on anchors.
5. **Slow turnover:** judge J is retired only if I(J; truth | panel + candidate) < ε on held-out anchors. At most one swap per K rounds, with hysteresis. Retired judges go to the hall of fame.
6. **Information firewall:** coaches and agents see only verdicts on anchor items, never judge internals or gradients.

## Metrics

- **Primary:** environment samples needed to reach X% of the PPO asymptote.
- **World model:** held-out multi-step prediction error inside and outside the disagreement region.
- **Honesty:**
  - stated-vs-revealed consistency rate,
  - consistency under paraphrased or counterfactual states,
  - audit-panel vs. training-panel score gap (the hacking indicator).
- **Panel health:**
  - mean pairwise error correlation,
  - calibration (ECE),
  - turnover rate,
  - hall-of-fame regression.
- **Compute:** total FLOPs. Report this honestly, since the method trades compute for environment samples.

## Repo layout (planned)

```
panel/
  envs/          # MiniGrid, Procgen wrappers, sample-cost accounting
  judges/        # world-model backbones + heads
  agent/         # PPO / model-based policy
  panel/         # query selection, scoring, IRT, turnover
  audit/         # frozen panel, hall of fame
  experiments/   # configs per ablation
  analysis/      # metrics, plots
```

## Status

Pre-implementation. See `plan.md`.
