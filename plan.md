# Plan: Judge-Panel RL Experiment

**Scope:** single GPU, small environments, and a clean ablation story. The paper is the set of ablations, not peak performance.

## Hypotheses

- **H1 (efficiency):** Panel-guided querying reaches 80% of the PPO asymptote with ≥5× fewer environment samples than PPO, and ≥2× fewer than Plan2Explore.
- **H2 (anchor necessity):** Removing ground-truth anchors causes panel drift: a rising gap between audit-panel and training-panel scores, and judge consensus that is confidently wrong.
- **H3 (firewall necessity):** Giving coaches or the agent access to judge internals produces adversarial exploitation. Training-panel scores go up while audit-panel scores stay flat or fall.
- **H4 (diversity):** The efficiency gain shrinks toward zero as judge error correlation goes to 1.
- **H5 (honesty):** The stated-vs-revealed consistency metric detects an injected deceptive agent with AUROC ≥ 0.9.
- **H6 (acquisition head, stretch):** A learned P(error | s) matches exhaustive disagreement search at less than 20% of its compute.

## Environments

1. **MiniGrid** (DoorKey, KeyCorridor, MultiRoom). The state is discrete, and plans and dynamics are easy to inspect. Used for development and the honesty experiments.
2. **Procgen** (CoinRun, Maze, Heist; 200 levels). Tests generalization and makes the sample-efficiency claim credible.
3. **Optional:** a synthetic environment with a tunable "expensive sample" cost, to show where the method crosses over against cheap simulation.

**Accounting:** every environment step goes through one counter, including anchor queries and rollouts used for judge training.

## Architecture

- **Judge backbone:** small latent world model (DreamerV3-lite or an RSSM of about 5M parameters). Diversity across the 8 judges comes from:
  - architecture: half CNN encoders, half ViT-tiny,
  - seeds,
  - bootstrap data subsets,
  - latent dimension.
- **Heads:** dynamics, reward, agent-action prediction, and self-calibration in v1. Inverse dynamics, peer prediction, and acquisition in v2.
- **Agent:** PPO baseline policy. Alternative is actor-critic trained in imagination, using competence-weighted rollouts from the panel.
- **Competence model:** a 2-parameter IRT model (judge ability, item difficulty), fit by EM on anchor items.

## Phases

### Phase 0: Infrastructure (1–2 weeks)
- Environment wrappers with sample counting, plus a config system and logging (wandb or local).
- Run PPO and Plan2Explore baselines on 3 MiniGrid tasks and 3 Procgen games, with 3 seeds each.
- **Exit:** baseline curves reproduce published numbers within about 10%.

### Phase 1: Static panel (2–3 weeks)
- 8 judges with heads 1, 2, 3 and 6, trained on a shared replay buffer.
- Query selection by variance of judge Q-estimates; the top-k plus 15% random anchors go to the environment.
- No turnover.
- **Exit:** H1 measured on MiniGrid.
- **Kill criterion:** if the gain is less than 1.5× fewer samples, diagnose judge error correlation before continuing.

### Phase 2: Honesty (2 weeks)
- The agent emits a stated plan as a k-step action sequence before acting.
- Consistency metric:
  - compare the stated plan with revealed actions,
  - re-check with paraphrased states (augmentations, symmetric layouts) and counterfactual states.
- Inject a deceptive agent that says one plan and does another under specific triggers.
- **Exit:** H5 measured.

### Phase 3: Safeguard ablations (2–3 weeks)

This phase is the core of the paper.

| Ablation | Change | Expected |
|---|---|---|
| A0 | Full system | Baseline |
| A1 | No anchors (judges scored by panel consensus) | Drift (H2) |
| A2 | No firewall (agent gets judge gradients) | Exploitation (H3) |
| A3 | Homogeneous judges (same architecture and data) | Gain shrinks (H4) |
| A4 | Fast turnover, no hysteresis | Whipsaw: oscillating scores, hall-of-fame regression |
| A5 | Panel approval as reward (persuasion objective) | Hacking: audit gap |

Use 3 seeds minimum per cell.

### Phase 4: Dynamic panel (2–3 weeks)
- Turnover rule: retire J if conditional mutual information on held-out anchors is below ε. At most one swap per K rounds; retired judges go to the hall of fame.
- Candidate judges come from distilled agent world models. This tests the idea that new agents replace the judges they make redundant.
- League variant: add exploiter agents that search for judge blind spots, and fold the states they find into the anchor pool.
- **Exit:** stable panel health over at least 10 turnover cycles, with no hall-of-fame regression.

### Phase 5: Procgen scale-up and acquisition head (3 weeks)
- Repeat A0 and the key ablations on Procgen.
- Add head 8, the learned acquisition function, and test H6.

### Phase 6: Write-up (2–3 weeks)
- Framing: a sample-efficiency and oversight paper. The safeguard ablations are the main contribution.
- Plots:
  - environment samples vs. return,
  - audit gap over time per ablation,
  - efficiency vs. judge error correlation,
  - honesty AUROC.

**Total:** about 15–19 weeks.

## Compute budget (rough)

- MiniGrid: a full grid of 6 ablations × 3 tasks × 3 seeds is feasible on one consumer GPU in days.
- Procgen: 8 judges cost roughly 8× the world-model training compute. Expect 1–2 weeks of GPU time for the core cells. Cut to 2 games if needed.
- Always report total FLOPs next to environment samples.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Judge errors stay correlated despite diversity tricks | High | Measure correlation first; add explicit decorrelation loss; vary training data harder |
| Gains vanish once compute is counted | Medium | Frame the result explicitly for expensive-sample regimes; include the synthetic cost environment |
| Honesty metric is gameable by consistent-but-wrong agents | Medium | Consistency is necessary, not sufficient; pair it with anchor accuracy |
| Turnover criterion is noisy at small N | Medium | Larger held-out anchor pool; bootstrap CIs on mutual information |
| Scope creep (coaches, mentors, teams) | High | League variant only in Phase 4; coaches deferred to future work |

## Out of scope for v1
- Language-model judges and agents.
- Coach and mentor roles beyond the exploiter league.
- Real-robot or human-in-the-loop experiments.

## First tasks
1. Scaffold the repo and the environment sample counter.
2. Run PPO and Plan2Explore baselines on MiniGrid DoorKey-8x8.
3. Train a single judge world model with all four v1 heads, and check its calibration.
4. Train 8 judges and measure their pairwise error correlation before building anything else, since H4 gates the whole project.
