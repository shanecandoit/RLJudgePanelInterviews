import numpy as np

from panel.envs.counting import Oracle, SampleCounter, make_minigrid, snapshot

ENV = "MiniGrid-DoorKey-5x5-v0"


def test_obs_is_flat_and_normalised():
    env = make_minigrid(ENV)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (5 * 5 * 3,) and obs.dtype == np.float32
    assert obs.min() >= 0.0 and obs.max() <= 1.0


def test_every_step_is_counted_across_envs():
    c = SampleCounter()
    envs = [make_minigrid(ENV, c) for _ in range(3)]
    for e in envs:
        e.reset(seed=0)
        for _ in range(4):
            e.step(e.action_space.sample())
    assert c.n == 12


def test_reset_is_free():
    c = SampleCounter()
    make_minigrid(ENV, c).reset(seed=0)
    assert c.n == 0


def test_snapshot_is_independent_and_shares_counter():
    c = SampleCounter()
    env = make_minigrid(ENV, c)
    env.reset(seed=0)
    snap = snapshot(env)
    assert snap.counter is c
    before = snap.unwrapped.agent_pos, snap.unwrapped.agent_dir
    for a in (1, 2, 2, 0):
        env.step(a)
    assert (snap.unwrapped.agent_pos, snap.unwrapped.agent_dir) == before
    assert c.n == 4  # snapshotting charged nothing


def test_oracle_query_costs_one_sample_and_does_not_mutate_input():
    c = SampleCounter()
    oracle = Oracle(ENV, c)
    _, snap = oracle.reset(seed=1)
    pose = snap.unwrapped.agent_pos, snap.unwrapped.agent_dir
    o1, *_, nxt1 = oracle.query(snap, 1)  # turn right
    o2, *_, nxt2 = oracle.query(snap, 1)  # same state, same action -> same result
    assert c.n == 2
    np.testing.assert_array_equal(o1, o2)
    assert nxt1 is not nxt2 and nxt1 is not snap
    assert (snap.unwrapped.agent_pos, snap.unwrapped.agent_dir) == pose
    assert nxt1.unwrapped.agent_dir != pose[1]
