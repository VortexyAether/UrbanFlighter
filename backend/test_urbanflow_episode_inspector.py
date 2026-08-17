from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys


VENV_PYTHON = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
if ".venv" not in sys.executable and VENV_PYTHON.exists():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

import main
from test_urbanflow_live_scenario import _field, _record, _request_json, _weather
from urbanflow_gym.inspector import (
    INSPECTOR_DEFAULT_MAX_STEPS,
    INSPECTOR_MAX_BATCH_STEPS,
    INSPECTOR_MAX_STEPS,
    InspectorSessionManager,
    StaleInspectorScenarioError,
    UnknownInspectorSessionError,
    estimate_inspector_minimum_required,
)
from urbanflow_gym.env import UrbanFlowConfig
from urbanflow_gym.live_scenario import LiveScenarioRegistry, live_scenario_registry
from urbanflow_gym.schemas import ACTOR_OBSERVATION_FIELDS


def _all_keys(value) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(_all_keys(child))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for child in value:
            keys.update(_all_keys(child))
        return keys
    return set()


def _register_api_record():
    main.inspector_session_manager.clear()
    live_scenario_registry.clear()
    record = _record()
    live_scenario_registry.register(record)
    return record


def test_inspector_horizon_covers_representative_live_route_lower_bound() -> None:
    config = UrbanFlowConfig()
    # The deterministic mission endpoints in a representative registered 800 m
    # live world. Even their unobstructed straight-line lower bound exceeds the
    # untouched core-environment horizon before obstacle detours are considered.
    representative_start_xy_m = (-395.75, -395.75)
    representative_goal_xy_m = (395.75, 395.75)
    representative_route_distance_m = math.dist(
        representative_start_xy_m,
        representative_goal_xy_m,
    )
    minimum_steps, minimum_time_s = estimate_inspector_minimum_required(
        representative_route_distance_m,
        config,
    )
    expected_time_s = (
        representative_route_distance_m / config.max_ground_speed_mps
        + config.max_ground_speed_mps / config.max_acceleration_mps2
    )
    assert math.isclose(minimum_time_s, expected_time_s, rel_tol=0.0, abs_tol=1e-12)
    assert minimum_steps == math.ceil(expected_time_s / config.dt_s)
    assert minimum_steps > config.max_steps == 360
    assert INSPECTOR_DEFAULT_MAX_STEPS == 1_200
    assert INSPECTOR_MAX_STEPS == 1_600
    assert INSPECTOR_DEFAULT_MAX_STEPS >= minimum_steps * 2
    assert INSPECTOR_DEFAULT_MAX_STEPS * config.dt_s == 300.0

    record = _register_api_record()
    status, created = _request_json(
        "POST",
        "/urbanflow-gym/inspector/sessions",
        {
            "scenario_id": record.scenario_id,
            "seed": 10_007,
            "baseline": "shortest_path",
        },
    )
    assert status == 200
    assert created["limits"]["max_steps"] == INSPECTOR_DEFAULT_MAX_STEPS
    assert created["limits"]["dt_s"] == config.dt_s
    assert created["limits"]["simulated_max_s"] == 300.0
    assert created["limits"]["max_batch_steps"] == INSPECTOR_MAX_BATCH_STEPS
    assert created["frame"]["simulated_elapsed_s"] == 0.0
    assert created["frame"]["simulated_max_s"] == 300.0


def test_same_live_world_seed_and_baseline_produce_identical_frames() -> None:
    registry = LiveScenarioRegistry(max_entries=2)
    record = _record()
    registry.register(record)
    manager = InspectorSessionManager(registry=registry, max_sessions=4)

    first = manager.create(
        scenario_id=record.scenario_id,
        seed=10_007,
        baseline_id="wind_aware_inlet",
        max_steps=8,
    )
    second = manager.create(
        scenario_id=record.scenario_id,
        seed=10_007,
        baseline_id="wind_aware_inlet",
        max_steps=8,
    )
    assert first["session_id"] != second["session_id"]
    assert first["world"] == second["world"]
    assert first["frame"] == second["frame"]

    for _ in range(4):
        first_step = manager.step(first["session_id"])
        second_step = manager.step(second["session_id"])
        assert first_step["frame"] == second_step["frame"]
        if not first_step["session_active"]:
            break

    reset_session = manager.create(
        scenario_id=record.scenario_id,
        seed=2718,
        baseline_id="shortest_path",
        max_steps=8,
    )
    initial_frame = reset_session["frame"]
    manager.step(reset_session["session_id"])
    reset = manager.reset(reset_session["session_id"])
    assert reset["frame"] == initial_frame
    assert reset["limits"]["reset_count"] == 1


def test_batch_matches_serial_steps_for_baseline_and_action_override() -> None:
    registry = LiveScenarioRegistry(max_entries=2)
    record = _record()
    registry.register(record)
    manager = InspectorSessionManager(registry=registry, max_sessions=8)

    for baseline_id, action in (
        ("shortest_path", None),
        ("direct_goal", [0.35, -0.2]),
    ):
        serial = manager.create(
            scenario_id=record.scenario_id,
            seed=10_007,
            baseline_id=baseline_id,
            max_steps=20,
        )
        batched = manager.create(
            scenario_id=record.scenario_id,
            seed=10_007,
            baseline_id=baseline_id,
            max_steps=20,
        )
        serial_batch_reward = 0.0
        serial_final = None
        for _ in range(7):
            serial_final = manager.step(serial["session_id"], action)
            serial_batch_reward += serial_final["frame"]["reward"]["step_total"]
        assert serial_final is not None

        batch_final = manager.step(batched["session_id"], action, repeat=7)
        assert batch_final["requested_steps"] == 7
        assert batch_final["executed_steps"] == 7
        assert batch_final["batch_reward"] == serial_batch_reward
        assert batch_final["frame"] == serial_final["frame"]
        assert batch_final["frame"]["reward"]["episode_total"] == (
            serial_final["frame"]["reward"]["episode_total"]
        )
        assert len(batch_final["frame"]["trajectory_xy_m"]) == 8

    serial = manager.create(
        scenario_id=record.scenario_id,
        seed=10_007,
        baseline_id="direct_goal",
        max_steps=100,
    )
    batched = manager.create(
        scenario_id=record.scenario_id,
        seed=10_007,
        baseline_id="direct_goal",
        max_steps=100,
    )
    serial_reward = 0.0
    serial_steps = 0
    serial_final = None
    while serial_steps < INSPECTOR_MAX_BATCH_STEPS:
        serial_final = manager.step(serial["session_id"], [-1.0, 0.0])
        serial_steps += 1
        serial_reward += serial_final["frame"]["reward"]["step_total"]
        if not serial_final["session_active"]:
            break
    assert serial_final is not None
    assert serial_final["frame"]["termination_reason"] == "collision"
    assert serial_steps < INSPECTOR_MAX_BATCH_STEPS

    batch_final = manager.step(
        batched["session_id"],
        [-1.0, 0.0],
        repeat=INSPECTOR_MAX_BATCH_STEPS,
    )
    assert batch_final["requested_steps"] == INSPECTOR_MAX_BATCH_STEPS
    assert batch_final["executed_steps"] == serial_steps
    assert batch_final["batch_reward"] == serial_reward
    assert batch_final["frame"] == serial_final["frame"]
    assert batch_final["cleanup"] == "terminal_session_deleted"


def test_terminal_mid_batch_stops_and_preserves_cleanup() -> None:
    record = _register_api_record()
    status, created = _request_json(
        "POST",
        "/urbanflow-gym/inspector/sessions",
        {
            "scenario_id": record.scenario_id,
            "seed": 1,
            "baseline": "direct_goal",
            "max_steps": 3,
        },
    )
    assert status == 200
    session_id = created["session_id"]
    status, final = _request_json(
        "POST",
        f"/urbanflow-gym/inspector/sessions/{session_id}/step",
        {"repeat": 5, "action": [0.0, 0.0]},
    )
    assert status == 200
    assert final["requested_steps"] == 5
    assert final["executed_steps"] == 3
    assert final["session_active"] is False
    assert final["cleanup"] == "terminal_session_deleted"
    assert final["frame"]["step_index"] == 3
    assert final["frame"]["truncated"] is True
    assert final["frame"]["termination_reason"] == "time_limit"
    assert final["frame"]["simulated_elapsed_s"] == 0.75
    assert len(final["frame"]["trajectory_xy_m"]) == 4
    for method, suffix in (("POST", "step"), ("POST", "reset"), ("DELETE", "")):
        path = f"/urbanflow-gym/inspector/sessions/{session_id}"
        if suffix:
            path += f"/{suffix}"
        missing_status, _ = _request_json(method, path)
        assert missing_status == 404


def test_step_and_action_validation_does_not_advance_on_rejection() -> None:
    record = _register_api_record()
    status, created = _request_json(
        "POST",
        "/urbanflow-gym/inspector/sessions",
        {
            "scenario_id": record.scenario_id,
            "seed": 10_007,
            "baseline": "shortest_path",
            "max_steps": 5,
        },
    )
    assert status == 200
    session_id = created["session_id"]

    invalid_actions = (
        {"action": [1.01, 0.0]},
        {"action": [-1.01, 0.0]},
        {"action": [0.0]},
        {"action": [0.0, 0.0, 0.0]},
        {"action": [True, 0.0]},
        {"action": "forward"},
        {"action": [0.0, 0.0], "unexpected": True},
    )
    for payload in invalid_actions:
        invalid_status, _ = _request_json(
            "POST",
            f"/urbanflow-gym/inspector/sessions/{session_id}/step",
            payload,
        )
        assert invalid_status == 422, payload

    status, stepped = _request_json(
        "POST",
        f"/urbanflow-gym/inspector/sessions/{session_id}/step",
        {"action": [0.0, 0.0]},
    )
    assert status == 200
    assert stepped["frame"]["step_index"] == 1
    assert stepped["frame"]["local_guidance_action"]["source"] == "validated_actor_override"

    status, baseline_step = _request_json(
        "POST",
        f"/urbanflow-gym/inspector/sessions/{session_id}/step",
    )
    assert status == 200
    assert baseline_step["frame"]["step_index"] == 2
    assert baseline_step["frame"]["local_guidance_action"]["source"] == "deterministic_baseline"


def test_repeat_validation_rejects_abuse_without_advancing() -> None:
    record = _register_api_record()
    status, created = _request_json(
        "POST",
        "/urbanflow-gym/inspector/sessions",
        {
            "scenario_id": record.scenario_id,
            "seed": 10_007,
            "baseline": "shortest_path",
            "max_steps": 8,
        },
    )
    assert status == 200
    session_id = created["session_id"]
    for repeat in (0, 65, True, 1.0):
        invalid_status, _ = _request_json(
            "POST",
            f"/urbanflow-gym/inspector/sessions/{session_id}/step",
            {"repeat": repeat},
        )
        assert invalid_status == 422, repeat

    status, stepped = _request_json(
        "POST",
        f"/urbanflow-gym/inspector/sessions/{session_id}/step",
        {"repeat": 1},
    )
    assert status == 200
    assert stepped["requested_steps"] == stepped["executed_steps"] == 1
    assert stepped["frame"]["step_index"] == 1
    status, upper_bound = _request_json(
        "POST",
        f"/urbanflow-gym/inspector/sessions/{session_id}/step",
        {"repeat": INSPECTOR_MAX_BATCH_STEPS},
    )
    assert status == 200
    assert upper_bound["requested_steps"] == INSPECTOR_MAX_BATCH_STEPS
    assert upper_bound["executed_steps"] == 7
    assert upper_bound["frame"]["step_index"] == 8
    assert upper_bound["cleanup"] == "terminal_session_deleted"


def test_actor_frame_contains_visual_contract_without_flow_field_leakage() -> None:
    record = _register_api_record()
    status, created = _request_json(
        "POST",
        "/api/urbanflow-gym/live/inspector/sessions",
        {
            "scenario_id": record.scenario_id,
            "seed": 10_007,
            "baseline": "direct_goal",
            "max_steps": 6,
        },
    )
    assert status == 200
    frame = created["frame"]
    world = created["world"]
    assert frame["scenario_id"] == record.scenario_id == world["scenario_id"]
    assert world["source"] == "exact_registered_live_osm_scenario"
    assert world["synthetic_fixture"] is False
    assert world["structure_count"] == 2 == len(world["buildings"])
    assert sorted(building["height_m"] for building in world["buildings"]) == [17.5, 37.5]
    assert len(frame["actor_lidar"]["rays"]) == 16
    assert len(frame["actor_observation"]["vector"]) == 33
    assert len(frame["actor_observation"]["fields"]) == 10
    for field, specification in zip(
        frame["actor_observation"]["fields"],
        ACTOR_OBSERVATION_FIELDS,
        strict=True,
    ):
        assert field["name"] == specification.name
        assert field["units"] == specification.units
        assert field["source"] == specification.source
        assert len(field["values"]) == specification.size
    assert frame["trajectory_xy_m"] == [frame["start_xy_m"]]
    assert frame["dt_s"] == 0.25
    assert frame["simulated_elapsed_s"] == 0.0
    assert frame["simulated_max_s"] == 1.5
    assert frame["distance_to_goal_m"] > 0.0
    assert frame["estimated_minimum_steps"] > 0
    assert frame["estimated_minimum_time_s"] > 0.0
    assert frame["flags"] == {
        "policy_status": "not_trained",
        "policy_had_privileged_flow_access": False,
        "full_flow_access": False,
        "training_executed": False,
        "browser_motor_training": False,
        "navier_stokes_cfd": False,
        "real_cfd_validation_run": False,
        "real_cfd_adapter_status": "interface_only_not_executed",
        "synthetic_fixture": False,
    }

    forbidden = {
        "ux",
        "uy",
        "mask",
        "flow_field",
        "full_flow_field",
        "grid_shape",
        "grid_digest_sha256",
        "hidden_flow",
        "hidden_local_wind_xy",
        "hidden_wake_delta_xy",
        "privileged_critic_state",
        "wind_provider",
    }
    assert not (_all_keys(created) & forbidden)
    status, batched = _request_json(
        "POST",
        f"/urbanflow-gym/inspector/sessions/{created['session_id']}/step",
        {"repeat": 3},
    )
    assert status == 200
    assert batched["executed_steps"] == 3
    assert len(batched["frame"]["trajectory_xy_m"]) == 4
    assert not (_all_keys(batched) & forbidden)


def test_terminal_cleanup_hard_limits_lru_and_ttl() -> None:
    record = _register_api_record()
    status, created = _request_json(
        "POST",
        "/urbanflow-gym/inspector/sessions",
        {
            "scenario_id": record.scenario_id,
            "seed": 1,
            "baseline": "direct_goal",
            "max_steps": 1,
        },
    )
    assert status == 200
    session_id = created["session_id"]
    status, final = _request_json(
        "POST",
        f"/urbanflow-gym/inspector/sessions/{session_id}/step",
    )
    assert status == 200
    assert final["session_active"] is False
    assert final["cleanup"] == "terminal_session_deleted"
    assert final["frame"]["terminated"] or final["frame"]["truncated"]
    assert final["frame"]["step_index"] == 1
    assert len(final["frame"]["trajectory_xy_m"]) == 2
    for method, suffix in (("POST", "step"), ("POST", "reset"), ("DELETE", "")):
        path = f"/urbanflow-gym/inspector/sessions/{session_id}"
        if suffix:
            path += f"/{suffix}"
        missing_status, _ = _request_json(method, path)
        assert missing_status == 404

    for max_steps in (0, 1_601, 1.0, True):
        invalid_status, _ = _request_json(
            "POST",
            "/urbanflow-gym/inspector/sessions",
            {
                "scenario_id": record.scenario_id,
                "seed": 1,
                "baseline": "direct_goal",
                "max_steps": max_steps,
            },
        )
        assert invalid_status == 422

    clock = [100.0]
    registry = LiveScenarioRegistry(max_entries=2)
    registry.register(record)
    ids = iter(("session-a", "session-b", "session-c", "session-d"))
    manager = InspectorSessionManager(
        registry=registry,
        max_sessions=2,
        ttl_s=10.0,
        clock=lambda: clock[0],
        id_factory=lambda: next(ids),
    )
    first = manager.create(
        scenario_id=record.scenario_id,
        seed=1,
        baseline_id="direct_goal",
        max_steps=3,
    )
    second = manager.create(
        scenario_id=record.scenario_id,
        seed=2,
        baseline_id="direct_goal",
        max_steps=3,
    )
    manager.reset(first["session_id"])
    manager.create(
        scenario_id=record.scenario_id,
        seed=3,
        baseline_id="direct_goal",
        max_steps=3,
    )
    try:
        manager.step(second["session_id"])
    except UnknownInspectorSessionError:
        pass
    else:
        raise AssertionError("least-recently-used inspector session was not evicted")
    clock[0] += 10.0
    assert len(manager) == 0
    try:
        manager.reset(first["session_id"])
    except UnknownInspectorSessionError:
        pass
    else:
        raise AssertionError("expired inspector session remained available")


def test_stale_and_unknown_scenario_ids_are_rejected() -> None:
    record = _register_api_record()
    unknown_id = f"urbanflow-live-v1-{'f' * 24}"
    status, payload = _request_json(
        "POST",
        "/urbanflow-gym/inspector/sessions",
        {
            "scenario_id": unknown_id,
            "seed": 1,
            "baseline": "direct_goal",
            "max_steps": 3,
        },
    )
    assert status == 404
    assert "stale" in payload["detail"]

    registry = LiveScenarioRegistry(max_entries=1)
    registry.register(record)
    manager = InspectorSessionManager(registry=registry, max_sessions=2)
    created = manager.create(
        scenario_id=record.scenario_id,
        seed=1,
        baseline_id="direct_goal",
        max_steps=3,
    )
    replacement = _record(
        weather=_weather(speed_mps=4.0),
        inlet_xy=(4.0, 0.0),
        field=_field(inlet_xy=(4.0, 0.0)),
    )
    registry.register(replacement)
    try:
        manager.step(created["session_id"])
    except StaleInspectorScenarioError:
        pass
    else:
        raise AssertionError("session continued after its registered scenario was evicted")
    assert len(manager) == 0


def test_inspector_imports_no_learning_stack() -> None:
    assert "gymnasium" not in sys.modules
    assert "stable_baselines3" not in sys.modules
    assert "torch" not in sys.modules
    assert "urbanflow_gym.gym_adapter" not in sys.modules
    assert "urbanflow_gym.train" not in sys.modules


if __name__ == "__main__":
    tests = (
        test_inspector_horizon_covers_representative_live_route_lower_bound,
        test_same_live_world_seed_and_baseline_produce_identical_frames,
        test_batch_matches_serial_steps_for_baseline_and_action_override,
        test_terminal_mid_batch_stops_and_preserves_cleanup,
        test_step_and_action_validation_does_not_advance_on_rejection,
        test_repeat_validation_rejects_abuse_without_advancing,
        test_actor_frame_contains_visual_contract_without_flow_field_leakage,
        test_terminal_cleanup_hard_limits_lru_and_ttl,
        test_stale_and_unknown_scenario_ids_are_rejected,
        test_inspector_imports_no_learning_stack,
    )
    for test in tests:
        test()
    print(json.dumps({"status": "ok", "tests": len(tests)}))
