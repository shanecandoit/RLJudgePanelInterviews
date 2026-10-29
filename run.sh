#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# 99 runs: 3 algorithms x 3 environments x 3 judge counts x 3 seeds,
# plus real-only PPO/DQN baselines once per environment/seed.
# Equal real-sample budgets: Empty=40k, LavaGap=150k, DoorKey=300k.
# Stops STARTING jobs after 8 hours; in-flight jobs finish. Run again to resume.
# Examples:
#   bash run.sh --dry-run
#   bash run.sh --hours 0             # finish every remaining run
#   bash run.sh --steps 384 --seeds 0 --eval-episodes 1 --out runs/algo_smoke
# Set PYTHON to choose a Python executable. Extra arguments override defaults.
exec "${PYTHON:-python}" -u -m panel.experiments.algo_sweep \
  --algos ppo dqn dyna --envs empty5x5 lavagap doorkey \
  --judges 1 2 4 --seeds 0 1 2 --workers 5 --hours 8 \
  --out runs/algo_sweep "$@"
