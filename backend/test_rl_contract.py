from __future__ import annotations

import json
import os
from pathlib import Path
import sys

VENV_PYTHON = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
if ".venv" not in sys.executable and VENV_PYTHON.exists():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

import numpy as np

from urban_flighter_rl import UrbanFlighterEnv
from urban_flighter_rl.planner import _grid_astar
from urban_flighter_rl.rollout import run_deterministic_baseline_rollout
from urban_flighter_rl.wind import UrbanWindField
from urban_flighter_rl.world import Building, UrbanWorld


def test_policy_observation_contract_excludes_privileged_flow() -> None:
    env = UrbanFlighterEnv(max_steps=3)
    obs, info = env.reset(seed=11, options={"randomize_mission": True})
    spec = env.spec()
    contract = spec["policy_observation_contract"]

    assert contract["privileged_flow_access"] is False
    assert "absolute_position" not in contract["fields"]
    assert "full_flow_grid" in contract["forbidden"]
    assert len(obs) == spec["observation_space"]["shape"][0]
    assert info["policy_had_privileged_flow_access"] is False


def test_seeded_multi_drone_rollout_serializes_policy_frames() -> None:
    first = run_deterministic_baseline_rollout(seed=11, max_steps=5, n_drones=3, randomize_missions=True)
    second = run_deterministic_baseline_rollout(seed=11, max_steps=5, n_drones=3, randomize_missions=True)
    frame = first["drones"][0]["trajectory"][0]

    assert first["missions"] == second["missions"]
    assert len(first["missions"]) == 3
    assert "policy_observation" in frame
    assert "info" not in frame
    assert frame["policy_had_privileged_flow_access"] is False
    assert len(frame["policy_observation"]) == 18


def test_boundary_violation_is_not_building_collision() -> None:
    env = UrbanFlighterEnv(max_steps=1, start=[99.9, 99.9, 20.0], goal=[10.0, 10.0, 20.0])
    env.reset(seed=3)
    result = env.step([1.0, 1.0, 0.0])

    assert result.info["reward_terms"]["boundary_cost"] < 0.0
    assert result.info["reward_terms"]["collision_cost"] == 0.0
    assert env.metrics()["boundary_violations"] == 1
    assert env.metrics()["collisions"] == 0


def test_swept_motion_blocks_building_tunneling() -> None:
    env = UrbanFlighterEnv(max_steps=1, start=[14.0, 35.0, 10.0], goal=[90.0, 35.0, 10.0], dt=1.0)
    env.max_speed = 100.0
    env.reset(seed=4)
    env.vel[:] = (25.0, 0.0, 0.0)
    result = env.step([0.0, 0.0, 0.0])

    assert result.info["collision"] is True
    assert env.metrics()["collisions"] == 1
    assert env.pos.tolist() == [14.0, 35.0, 10.0]


def test_planner_fallback_edges_do_not_tunnel_through_buildings() -> None:
    world = UrbanWorld(
        bounds=(0.0, 100.0, 0.0, 100.0, 0.0, 60.0),
        buildings=[Building(np.array([88.0, 12.0]), np.array([10.0, 10.0]), 30.0)],
    )
    env = UrbanFlighterEnv(
        world=world,
        wind=UrbanWindField(world),
        start=[20.0, 20.0, 10.0],
        goal=[98.0, 2.0, 10.0],
        max_steps=1,
    )
    env.reset(seed=1)

    path = _grid_astar(env, spacing=20.0, z_levels=(10.0,))

    assert len(path) > 2
    for index in range(1, len(path)):
        assert not world.segment_hits_building(path[index - 1], path[index], margin=3.0)


def test_planner_does_not_emit_unsafe_overflight_when_ceiling_is_too_low() -> None:
    world = UrbanWorld(
        bounds=(0.0, 100.0, 0.0, 100.0, 0.0, 35.0),
        buildings=[Building(np.array([50.0, 50.0]), np.array([20.0, 90.0]), 34.0)],
    )
    env = UrbanFlighterEnv(
        world=world,
        wind=UrbanWindField(world),
        start=[10.0, 50.0, 10.0],
        goal=[90.0, 50.0, 10.0],
        max_steps=1,
    )
    env.reset(seed=2)

    path = _grid_astar(env, spacing=20.0, z_levels=(10.0,))

    for index in range(1, len(path)):
        assert not world.segment_hits_building(path[index - 1], path[index], margin=3.0)


if __name__ == "__main__":
    test_policy_observation_contract_excludes_privileged_flow()
    test_seeded_multi_drone_rollout_serializes_policy_frames()
    test_boundary_violation_is_not_building_collision()
    test_swept_motion_blocks_building_tunneling()
    test_planner_fallback_edges_do_not_tunnel_through_buildings()
    test_planner_does_not_emit_unsafe_overflight_when_ceiling_is_too_low()
    print(json.dumps({"status": "ok", "tests": 6}))
