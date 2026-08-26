from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch


VENV_PYTHON = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
if ".venv" not in sys.executable and VENV_PYTHON.exists():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

import numpy as np

from urbanflow_gym import (
    ACTION_DIM,
    ACTOR_OBSERVATION_DIM,
    BASELINE_ORDER,
    CFDWindProvider2DAdapter,
    UrbanFlowConfig,
    UrbanFlowEnv,
    relative_air_energy_step,
    run_baseline_evaluation,
    urbanflow_contract_payload,
)
from urbanflow_gym.evaluation import _evaluation_source, validate_evaluation_inputs
from urbanflow_gym.geometry import AxisAlignedPrism, UrbanGeometry
from urbanflow_gym.scenario import make_custom_scenario
from urbanflow_gym.schemas import leakage_guard_report
from urbanflow_gym.train import _require_training_extras
from urbanflow_gym.wind import ConstantWindProvider


def _custom_scenario(
    *,
    scenario_id: str,
    bounds_xy: tuple[float, float, float, float],
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    prisms: tuple[AxisAlignedPrism, ...] = (),
):
    geometry = UrbanGeometry(bounds_xy, prisms, flight_altitude_m=18.0)
    return make_custom_scenario(
        scenario_id=scenario_id,
        seed=31,
        geometry=geometry,
        start_xy=np.asarray(start_xy, dtype=float),
        goal_xy=np.asarray(goal_xy, dtype=float),
        wind_provider=ConstantWindProvider(np.zeros(2, dtype=float)),
    )


def test_seeded_reset_and_step_are_deterministic() -> None:
    actions = (
        np.array([0.75, 0.10]),
        np.array([0.65, -0.20]),
        np.array([0.50, 0.35]),
        np.array([0.90, 0.00]),
    )
    first = UrbanFlowEnv(UrbanFlowConfig(max_steps=20))
    second = UrbanFlowEnv(UrbanFlowConfig(max_steps=20))
    first_observation, first_info = first.reset(seed=2718)
    second_observation, second_info = second.reset(seed=2718)

    np.testing.assert_array_equal(first_observation, second_observation)
    assert first_info == second_info
    for index in range(12):
        first_step = first.step(actions[index % len(actions)])
        second_step = second.step(actions[index % len(actions)])
        np.testing.assert_array_equal(first_step[0], second_step[0])
        assert first_step[1:] == second_step[1:]
        if first_step[2] or first_step[3]:
            break

    assert first.metrics() == second.metrics()
    assert first.trajectory == second.trajectory
    repeated_observation, repeated_info = first.reset(seed=2718)
    np.testing.assert_array_equal(repeated_observation, first_observation)
    assert repeated_info == first_info


def test_numpy_spaces_and_actor_leakage_guard() -> None:
    env = UrbanFlowEnv(UrbanFlowConfig(max_steps=2))
    observation, reset_info = env.reset(seed=7)

    assert env.observation_space.shape == (ACTOR_OBSERVATION_DIM,) == (49,)
    assert env.action_space.shape == (ACTION_DIM,) == (2,)
    assert env.observation_space.contains(observation)
    assert env.action_space.contains(np.array([1.0, -1.0], dtype=np.float32))
    assert not env.action_space.contains(np.array([1.01, 0.0], dtype=np.float32))
    assert not env.observation_space.contains(np.full(49, np.nan, dtype=np.float32))

    guard = leakage_guard_report()
    contract = urbanflow_contract_payload()
    assert guard["status"] == "passed"
    assert guard["full_flow_access"] is False
    assert contract["labels"]["full_flow_access"] == "NO"
    assert contract["status"]["trained_policy_available"] is False
    assert contract["status"]["real_3d_navier_stokes_validated"] is False
    assert reset_info["policy_had_privileged_flow_access"] is False
    assert reset_info["privileged_critic_returned_to_actor"] is False
    assert reset_info["full_flow_access"] is False

    next_observation, _, _, _, step_info = env.step(np.zeros(2, dtype=float))
    assert env.observation_space.contains(next_observation)
    assert step_info["policy_had_privileged_flow_access"] is False
    assert step_info["privileged_critic_returned_to_actor"] is False
    for forbidden in (
        "privileged_critic_state",
        "hidden_local_wind_xy",
        "full_flow_field",
        "wind_provider",
    ):
        assert forbidden not in reset_info
        assert forbidden not in step_info


def test_tailwind_requires_less_relative_air_energy_than_headwind() -> None:
    ground_velocity = np.array([6.0, 0.0])
    tailwind = np.array([3.5, 0.0])
    headwind = np.array([-3.5, 0.0])

    tailwind_energy = relative_air_energy_step(ground_velocity, tailwind, 1.0)
    headwind_energy = relative_air_energy_step(ground_velocity, headwind, 1.0)

    assert tailwind_energy == 6.25
    assert headwind_energy == 90.25
    assert tailwind_energy < headwind_energy


def test_swept_collision_success_and_time_limit_termination() -> None:
    blocking_prism = AxisAlignedPrism(
        "blocking",
        np.array([8.0, 4.0]),
        np.array([10.0, 16.0]),
        30.0,
    )
    collision_scenario = _custom_scenario(
        scenario_id="swept-collision",
        bounds_xy=(0.0, 20.0, 0.0, 20.0),
        start_xy=(3.0, 10.0),
        goal_xy=(17.0, 10.0),
        prisms=(blocking_prism,),
    )
    collision_config = UrbanFlowConfig(
        dt_s=1.0,
        max_steps=4,
        max_ground_speed_mps=8.0,
        max_acceleration_mps2=20.0,
        velocity_tracking_time_s=0.1,
        wind_drag_gain_per_s=0.0,
        agent_radius_m=1.0,
        goal_radius_m=1.0,
    )
    collision_env = UrbanFlowEnv(collision_config, fixed_scenario=collision_scenario)
    collision_env.reset(seed=31)
    _, _, terminated, truncated, info = collision_env.step([1.0, 0.0])

    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "collision"
    assert collision_env.metrics()["collision_count"] == 1
    np.testing.assert_array_equal(collision_env.position_xy, collision_scenario.start_xy)
    try:
        collision_env.step([0.0, 0.0])
    except RuntimeError as exc:
        assert "call reset" in str(exc)
    else:
        raise AssertionError("step after termination was accepted")

    success_scenario = _custom_scenario(
        scenario_id="short-success",
        bounds_xy=(0.0, 30.0, 0.0, 30.0),
        start_xy=(5.0, 15.0),
        goal_xy=(9.0, 15.0),
    )
    success_env = UrbanFlowEnv(
        UrbanFlowConfig(max_steps=20, wind_drag_gain_per_s=0.0),
        fixed_scenario=success_scenario,
    )
    success_env.reset(seed=31)
    for _ in range(20):
        _, _, terminated, truncated, info = success_env.step([1.0, 0.0])
        if terminated or truncated:
            break
    assert terminated is True
    assert truncated is False
    assert info["termination_reason"] == "success"
    assert success_env.metrics()["success"] is True

    timeout_scenario = _custom_scenario(
        scenario_id="one-step-timeout",
        bounds_xy=(0.0, 30.0, 0.0, 30.0),
        start_xy=(5.0, 5.0),
        goal_xy=(25.0, 25.0),
    )
    timeout_env = UrbanFlowEnv(
        UrbanFlowConfig(max_steps=1),
        fixed_scenario=timeout_scenario,
    )
    timeout_env.reset(seed=31)
    _, _, terminated, truncated, info = timeout_env.step([0.0, 0.0])
    assert terminated is False
    assert truncated is True
    assert info["termination_reason"] == "time_limit"


def test_baselines_artifact_determinism_and_bounded_runtime() -> None:
    with TemporaryDirectory() as temporary_directory:
        artifact_path = Path(temporary_directory) / "evaluation.json"
        started = perf_counter()
        first = run_baseline_evaluation(
            seeds=[10_007],
            max_steps=50,
            artifact_path=artifact_path,
        )
        first_bytes = artifact_path.read_bytes()
        elapsed_s = perf_counter() - started
        second = run_baseline_evaluation(
            seeds=[10_007],
            max_steps=50,
            artifact_path=artifact_path,
        )

        assert elapsed_s < 10.0
        assert first == second
        assert artifact_path.read_bytes() == first_bytes
        parsed = json.loads(first_bytes)
        assert parsed["evaluation_id"] == first["evaluation_id"]

    assert tuple(first["baselines"]) == BASELINE_ORDER
    assert first["policy_status"] == "not_trained"
    assert first["policy_had_privileged_flow_access"] is False
    assert first["real_cfd_validation_status"] == "not_run_interface_only"
    assert first["dynamics_source"]["navier_stokes_cfd"] is False
    for baseline in first["baselines"].values():
        assert baseline["uses_hidden_flow"] is False
        assert baseline["aggregate"]["episodes"] == 1
        assert baseline["episodes"][0]["metrics"]["steps"] <= 50

    assert validate_evaluation_inputs([0], 50) == ((0,), 50)
    for seeds, max_steps in (([], 50), (range(6), 50), ([1], 49), ([1], 501)):
        try:
            validate_evaluation_inputs(seeds, max_steps)
        except ValueError:
            pass
        else:
            raise AssertionError("unbounded evaluation input was accepted")


def test_future_external_cfd_adapter_makes_no_validation_claim() -> None:
    class UserSuppliedDataset:
        inlet_velocity_xyz = np.array([4.0, 1.0, 0.0])

        def __init__(self) -> None:
            self.last_position: np.ndarray | None = None

        def sample_velocity_xyz(self, position_xyz: np.ndarray, time_s: float) -> np.ndarray:
            self.last_position = np.asarray(position_xyz, dtype=float)
            return np.array([3.0 + 0.1 * time_s, 0.5, 0.2])

        def dataset_metadata(self) -> dict:
            return {
                "solver_family": "navier_stokes_3d",
                "provenance_verified": True,
                "origin": "user supplied later",
            }

    dataset = UserSuppliedDataset()
    adapter = CFDWindProvider2DAdapter(dataset, flight_altitude_m=18.0)
    np.testing.assert_array_equal(adapter.inlet_velocity, np.array([4.0, 1.0]))
    np.testing.assert_allclose(adapter.velocity_at(np.array([7.0, 9.0]), 2.0), [3.2, 0.5])
    np.testing.assert_array_equal(dataset.last_position, np.array([7.0, 9.0, 18.0]))

    source = _evaluation_source(dataset)
    assert source["dataset_declares_navier_stokes_3d"] is True
    assert source["navier_stokes_cfd"] is False
    assert source["provenance_verified_by_urbanflow"] is False
    assert source["real_cfd_validation_status"] == (
        "external_dataset_executed_not_independently_validated"
    )


def test_training_entrypoint_reports_missing_optional_extras() -> None:
    with patch("urbanflow_gym.train.importlib.util.find_spec", return_value=None):
        try:
            _require_training_extras()
        except SystemExit as exc:
            message = str(exc)
        else:
            raise AssertionError("missing training extras were accepted")

    assert "gymnasium, stable_baselines3, torch" in message
    assert "backend/requirements-urbanflow-train.txt" in message


if __name__ == "__main__":
    tests = (
        test_seeded_reset_and_step_are_deterministic,
        test_numpy_spaces_and_actor_leakage_guard,
        test_tailwind_requires_less_relative_air_energy_than_headwind,
        test_swept_collision_success_and_time_limit_termination,
        test_baselines_artifact_determinism_and_bounded_runtime,
        test_future_external_cfd_adapter_makes_no_validation_claim,
        test_training_entrypoint_reports_missing_optional_extras,
    )
    for test in tests:
        test()
    print(json.dumps({"status": "ok", "tests": len(tests)}))
