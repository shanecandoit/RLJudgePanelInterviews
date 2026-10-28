"""Common run log so PPO and the panel agent are compared on identical columns.

Columns every algorithm must write per iteration:
  iteration   one collect+update cycle (PPO: rollout+update; panel: query round)
  env_steps   total charged environment samples so far
  wall_s      training wall time in seconds (evaluation time excluded)
  grad_steps  optimizer steps taken by the policy and (for the panel) the judges
  eval_return mean greedy return on uncounted evaluation episodes
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

REQUIRED = ("iteration", "env_steps", "wall_s", "grad_steps", "eval_return")


class RunLog:
    def __init__(self, algo: str, env_id: str, seed: int) -> None:
        self.algo, self.env_id, self.seed = algo, env_id, seed
        self.rows: List[Dict[str, Any]] = []

    def add(self, **row: Any) -> None:
        missing = [k for k in REQUIRED if k not in row]
        if missing:
            raise KeyError(f"log row missing {missing}")
        self.rows.append(row)

    def column(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.rows], dtype=float)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"algo": self.algo, "env_id": self.env_id, "seed": self.seed, "rows": self.rows}, f)

    @classmethod
    def load(cls, path: str) -> "RunLog":
        with open(path) as f:
            d = json.load(f)
        log = cls(d["algo"], d["env_id"], d["seed"])
        log.rows = d["rows"]
        return log
