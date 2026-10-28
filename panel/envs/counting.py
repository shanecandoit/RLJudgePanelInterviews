"""MiniGrid wrappers with a single environment-sample counter.

Every real `env.step` goes through one `SampleCounter`, including PPO rollouts,
panel anchor queries and seed rollouts. Evaluation envs are built without a
counter and are not charged (standard practice; reported separately).
"""
from __future__ import annotations

import copy
import warnings
from typing import Optional, Tuple

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

import gymnasium as gym
import minigrid  # noqa: F401  (registers MiniGrid-* envs)
import numpy as np
from minigrid.wrappers import FullyObsWrapper


class SampleCounter:
    def __init__(self) -> None:
        self.n = 0

    def add(self, k: int = 1) -> None:
        self.n += k


class FlatFullObs(gym.ObservationWrapper):
    """Fully observed grid -> flat float vector in [0, 1]."""

    SCALE = np.array([10.0, 5.0, 3.0], dtype=np.float32)  # object, color, state/direction

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        shape = env.observation_space["image"].shape
        self.observation_space = gym.spaces.Box(0.0, 1.0, (int(np.prod(shape)),), np.float32)

    def observation(self, obs):
        return (obs["image"].astype(np.float32) / self.SCALE).reshape(-1)


class CountedEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, counter: SampleCounter) -> None:
        super().__init__(env)
        self.counter = counter

    def step(self, action):
        self.counter.add(1)
        return self.env.step(action)


def make_minigrid(env_id: str, counter: Optional[SampleCounter] = None) -> gym.Env:
    env = FlatFullObs(FullyObsWrapper(gym.make(env_id)))
    return CountedEnv(env, counter) if counter is not None else env


def snapshot(env: gym.Env) -> gym.Env:
    """Independent copy of an env's full state. A CountedEnv copy keeps charging the same counter."""
    memo = {}
    counter = getattr(env, "counter", None)
    if counter is not None:
        memo[id(counter)] = counter
    return copy.deepcopy(env, memo)


class Oracle:
    """Ground-truth query: step a copy of a stored state with a chosen action.

    The input snapshot is never mutated, so it can be queried again. Each query
    costs exactly one environment sample on the shared counter.
    """

    def __init__(self, env_id: str, counter: SampleCounter) -> None:
        self.env = make_minigrid(env_id, counter)
        self.counter = counter
        self.n_actions = int(self.env.action_space.n)
        self.obs_dim = int(np.prod(self.env.observation_space.shape))

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, gym.Env]:
        obs, _ = self.env.reset(seed=seed)
        return obs, snapshot(self.env)

    def query(self, snap: gym.Env, action: int):
        """Returns (obs, reward, terminated, truncated, next_snap)."""
        nxt = snapshot(snap)
        obs, r, term, trunc, _ = nxt.step(int(action))
        return obs, float(r), bool(term), bool(trunc), nxt
