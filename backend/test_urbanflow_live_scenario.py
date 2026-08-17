from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch


VENV_PYTHON = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
if ".venv" not in sys.executable and VENV_PYTHON.exists():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

import numpy as np

import main
from urbanflow_gym.geometry import PolygonPrism
from urbanflow_gym.live_scenario import (
    LiveScenarioRegistry,
    UnknownLiveScenarioError,
    build_live_scenario_record,
    live_scenario_registry,
    make_live_scenario,
    snapshot_to_geometry,
)


def _field(
    *,
    inlet_xy: tuple[float, float] = (3.0, 0.0),
    half_span_m: float = 100.0,
    cells: int = 21,
) -> dict:
    count = cells * cells
    return {
        "nx": cells,
        "ny": cells,
        "cell_size_m": (half_span_m * 2.0) / (cells - 1),
        "bounds": {
            "min_x": -half_span_m,
            "max_x": half_span_m,
            "min_y": -half_span_m,
            "max_y": half_span_m,
        },
        "ux": [inlet_xy[0]] * count,
        "uy": [inlet_xy[1]] * count,
        "mask": [0] * count,
        "stats": {
            "mean_speed_mps": float(np.linalg.norm(inlet_xy)),
            "max_speed_mps": float(np.linalg.norm(inlet_xy)),
            "blocked_fraction": 0.0,
        },
    }


def _weather(
    *,
    speed_mps: float = 3.0,
    direction_deg: float = 270.0,
    timestamp: str = "2026-07-12T03:00Z",
) -> dict:
    return {
        "wind_speed": speed_mps,
        "wind_deg": direction_deg,
        "description": "Deterministic current-condition fixture",
        "units": {"wind_speed": "m/s", "wind_deg": "degrees_from_north"},
        "source": {
            "provider": "fixture-weather",
            "kind": "deterministic_current_conditions",
            "observation_time": timestamp,
            "variable_height_m": 10.0,
        },
        "fallback": {"used": False, "reason": None},
    }


def _buildings() -> list[dict]:
    return [
        {
            "building_id": "way:101:part:0",
            "height": 37.5,
            "height_source": "osm:height",
            "footprint": [[-8.0, -24.0], [8.0, -24.0], [8.0, 24.0], [-8.0, 24.0], [-8.0, -24.0]],
            "source": {
                "element_id": "way:101",
                "projected_crs": "EPSG:32652",
            },
        },
        {
            "building_id": "way:202:part:0",
            "height": 17.5,
            "height_source": "osm:building:levels_estimate_3.5m_per_level",
            "footprint": [[34.0, 24.0], [48.0, 24.0], [48.0, 40.0], [34.0, 40.0]],
            "source": {
                "element_id": "way:202",
                "projected_crs": "EPSG:32652",
            },
        },
    ]


def _record(
    *,
    buildings: list[dict] | None = None,
    weather: dict | None = None,
    inlet_xy: tuple[float, float] = (3.0, 0.0),
    field: dict | None = None,
):
    return build_live_scenario_record(
        lat=37.451448,
        lon=126.6515423,
        geometry_radius_m=100.0,
        solve_radius_m=200.0,
        buildings=buildings or _buildings(),
        weather=weather or _weather(),
        inlet_velocity_xy=inlet_xy,
        field=field or _field(inlet_xy=inlet_xy),
        flow_source={
            "kind": "POTENTIAL-FLOW CFD-LITE B fixture",
            "model": "deterministic-test-grid",
            "navier_stokes_cfd": False,
        },
    )


def _request_json(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    sent: list[dict] = []
    requests_to_receive = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive() -> dict:
        if requests_to_receive:
            return requests_to_receive.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    headers = [(b"host", b"testserver")]
    if payload is not None:
        headers.extend(
            [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ]
        )
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "root_path": "",
    }
    asyncio.run(main.app(scope, receive, send))
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return status, json.loads(response_body)


def _all_keys(value) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(_all_keys(child))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for child in value:
            result.update(_all_keys(child))
        return result
    return set()


def test_canonical_id_is_stable_and_registry_returns_independent_objects() -> None:
    first = _record()
    reordered = _buildings()[::-1]
    reordered[1] = {
        **reordered[1],
        # Same first polygon with a different start vertex and winding.
        "footprint": [[8.0, 24.0], [8.0, -24.0], [-8.0, -24.0], [-8.0, 24.0]],
    }
    second = _record(buildings=reordered)

    assert first.scenario_id == second.scenario_id
    assert first.canonical_snapshot_bytes == second.canonical_snapshot_bytes
    registry = LiveScenarioRegistry(max_entries=2)
    returned = registry.register(first)
    returned["location"]["selected_lat_deg"] = 0.0
    returned["inlet"]["velocity_xy_mps"][0] = 999.0
    snapshot = registry.snapshot(first.scenario_id)
    snapshot["buildings"][0]["footprint_xy_m"][0][0] = 999.0

    clean = registry.current_summary()
    assert clean["location"]["selected_lat_deg"] == 37.451448
    assert clean["inlet"]["velocity_xy_mps"] == [3.0, 0.0]
    assert registry.snapshot(first.scenario_id)["buildings"][0]["footprint_xy_m"][0][0] != 999.0
    try:
        first.flow_field.ux[0] = 9.0
    except ValueError:
        pass
    else:
        raise AssertionError("registered hidden flow array was mutable")


def test_changed_inlet_or_geometry_changes_content_addressed_id() -> None:
    baseline = _record()
    changed_inlet = _record(
        weather=_weather(speed_mps=4.0),
        inlet_xy=(4.0, 0.0),
        field=_field(inlet_xy=(4.0, 0.0)),
    )
    changed_geometry = _buildings()
    changed_geometry[0] = {
        **changed_geometry[0],
        "footprint": [[-9.0, -24.0], [8.0, -24.0], [8.0, 24.0], [-9.0, 24.0]],
    }
    geometry_record = _record(buildings=changed_geometry)

    assert baseline.scenario_id != changed_inlet.scenario_id
    assert baseline.scenario_id != geometry_record.scenario_id
    assert baseline.snapshot()["content_hash_sha256"] != changed_inlet.snapshot()["content_hash_sha256"]
    assert baseline.snapshot()["content_hash_sha256"] != geometry_record.snapshot()["content_hash_sha256"]


def test_polygon_height_world_conversion_and_coordinate_semantics() -> None:
    record = _record()
    snapshot = record.snapshot()
    geometry = snapshot_to_geometry(snapshot)
    assert snapshot["coordinate_frame"]["x_axis"] == "east"
    assert snapshot["coordinate_frame"]["y_axis"] == "north"
    assert snapshot["coordinate_frame"]["horizontal_units"] == "m"
    assert snapshot["coordinate_frame"]["projected_crs"] == "EPSG:32652"
    assert all(isinstance(prism, PolygonPrism) for prism in geometry.prisms)
    assert sorted(prism.height_m for prism in geometry.prisms) == [17.5, 37.5]

    main_prism = next(prism for prism in geometry.prisms if prism.height_m == 37.5)
    assert set(map(tuple, main_prism.footprint_xy.tolist())) == {
        (-8.0, -24.0),
        (8.0, -24.0),
        (8.0, 24.0),
        (-8.0, 24.0),
    }
    ranges = geometry.lidar_ranges(
        np.array([-30.0, 0.0]),
        np.array([0.0]),
        max_range_m=80.0,
    )
    np.testing.assert_allclose(ranges, [22.0], atol=1e-9)
    assert geometry.segment_collides(np.array([-30.0, 0.0]), np.array([30.0, 0.0]), 1.25)
    assert not geometry.segment_collides(np.array([-30.0, -40.0]), np.array([30.0, -40.0]), 1.25)

    scenario = make_live_scenario(record, seed=10007)
    np.testing.assert_array_equal(scenario.known_inlet_velocity_xy, np.array([3.0, 0.0]))
    np.testing.assert_array_equal(
        scenario.wind_provider.velocity_at(np.array([-50.0, -50.0]), 17.0),
        np.array([3.0, 0.0]),
    )


def test_no_live_scenario_has_clear_api_response() -> None:
    live_scenario_registry.clear()
    status, payload = _request_json("GET", "/urbanflow-gym/live-scenarios/current")
    assert status == 409
    assert "no live UrbanFlow scenario is loaded" in payload["detail"]
    assert "POST /flow-fields/2d" in payload["detail"]

    status, payload = _request_json(
        "POST",
        "/urbanflow-gym/live/evaluate",
        {"seeds": [10007], "max_steps": 50, "save_artifact": False},
    )
    assert status == 409
    assert "no live UrbanFlow scenario is loaded" in payload["detail"]


def test_flow_response_registers_exact_live_world_and_evaluation_has_no_leakage() -> None:
    live_scenario_registry.clear()
    buildings = _buildings()
    weather = _weather()
    field = _field()
    request = {
        "lat": 37.451448,
        "lon": 126.6515423,
        "geometry_radius_m": 100.0,
        "solve_radius_m": 200.0,
        "grid_size_m": 10.0,
        "use_real_weather": True,
    }
    with (
        patch.object(main, "fetch_buildings", return_value=buildings) as fetch,
        patch.object(main, "get_real_weather", return_value=weather) as get_weather,
        patch.object(main, "compute_cfd_lite_b_flow_2d", return_value=field) as solve,
    ):
        status, flow_response = _request_json("POST", "/flow-fields/2d", request)

    assert status == 200
    fetch.assert_called_once()
    get_weather.assert_called_once()
    solve.assert_called_once()
    live = flow_response["live_scenario"]
    scenario_id = live["scenario_id"]
    assert live["schema_id"] == "urbanflow.live_scenario.v1"
    assert live["structure_count"] == len(buildings) == len(flow_response["buildings"])
    assert live["location"]["selected_lat_deg"] == request["lat"]
    assert live["location"]["selected_lon_deg"] == request["lon"]
    np.testing.assert_allclose(live["inlet"]["velocity_xy_mps"], [3.0, 0.0], atol=1e-6)
    assert live["hidden_flow"]["grid_digest_sha256"]

    status, current = _request_json("GET", "/urbanflow-gym/live-scenarios/current")
    assert status == 200
    assert current["scenario_id"] == scenario_id

    status, evaluation = _request_json(
        "POST",
        "/urbanflow-gym/live/evaluate",
        {
            "scenario_id": scenario_id,
            "seeds": [10007],
            "max_steps": 50,
            "save_artifact": False,
        },
    )
    assert status == 200
    assert evaluation["scenario_kind"] == "live_osm_current_inlet"
    assert evaluation["scenario_id"] == scenario_id
    assert evaluation["live_scenario"]["scenario_id"] == scenario_id
    assert evaluation["live_scenario"]["structure_count"] == len(buildings)
    assert evaluation["live_scenario"]["location"]["selected_lat_deg"] == request["lat"]
    np.testing.assert_allclose(
        evaluation["live_scenario"]["inlet"]["velocity_xy_mps"],
        [3.0, 0.0],
        atol=1e-6,
    )
    assert evaluation["policy_status"] == "not_trained"
    assert evaluation["policy_had_privileged_flow_access"] is False
    assert evaluation["policy_full_flow_access"] is False
    assert evaluation["synthetic_hidden_flow"] is True
    assert evaluation["real_cfd_validation_run"] is False
    assert evaluation["dynamics_source"]["navier_stokes_cfd"] is False
    assert "ux" not in _all_keys(evaluation)
    assert "uy" not in _all_keys(evaluation)
    assert "buildings" not in _all_keys(evaluation)
    for baseline in evaluation["baselines"].values():
        assert baseline["uses_hidden_flow"] is False
        assert "full_flow_field" not in baseline["allowed_inputs"]


def test_invalid_stale_ids_and_mission_bounds_are_rejected() -> None:
    current_record = _record()
    live_scenario_registry.clear()
    live_scenario_registry.register(current_record)

    status, payload = _request_json(
        "POST",
        "/urbanflow-gym/live/evaluate",
        {
            "scenario_id": f"urbanflow-live-v1-{'f' * 24}",
            "seeds": [10007],
            "max_steps": 50,
            "save_artifact": False,
        },
    )
    assert status == 404
    assert "invalid, stale, or no longer cached" in payload["detail"]

    for override in (
        {"start_xy": [999.0, 0.0], "goal_xy": [0.0, 80.0]},
        {"start_xy": [0.0, 80.0]},
        {"start_xy": [4_000.0, 0.0], "goal_xy": [0.0, 80.0]},
    ):
        status, response = _request_json(
            "POST",
            "/urbanflow-gym/live/evaluate",
            {
                "scenario_id": current_record.scenario_id,
                "seeds": [10007],
                "max_steps": 50,
                "save_artifact": False,
                **override,
            },
        )
        assert status == 422, response

    registry = LiveScenarioRegistry(max_entries=1)
    first = _record()
    second = _record(
        weather=_weather(speed_mps=4.0),
        inlet_xy=(4.0, 0.0),
        field=_field(inlet_xy=(4.0, 0.0)),
    )
    registry.register(first)
    registry.register(second)
    try:
        registry.get_record(first.scenario_id)
    except UnknownLiveScenarioError:
        pass
    else:
        raise AssertionError("evicted scenario id remained live")


def test_api_and_live_evaluation_import_no_training_stack_and_run_no_training() -> None:
    assert "gymnasium" not in sys.modules
    assert "stable_baselines3" not in sys.modules
    assert "torch" not in sys.modules
    assert "urbanflow_gym.gym_adapter" not in sys.modules
    assert "urbanflow_gym.train" not in sys.modules
    assert main.urbanflow_contract_payload()["status"]["training_executed_on_this_machine"] is False


if __name__ == "__main__":
    tests = (
        test_canonical_id_is_stable_and_registry_returns_independent_objects,
        test_changed_inlet_or_geometry_changes_content_addressed_id,
        test_polygon_height_world_conversion_and_coordinate_semantics,
        test_no_live_scenario_has_clear_api_response,
        test_flow_response_registers_exact_live_world_and_evaluation_has_no_leakage,
        test_invalid_stale_ids_and_mission_bounds_are_rejected,
        test_api_and_live_evaluation_import_no_training_stack_and_run_no_training,
    )
    for test in tests:
        test()
    print(json.dumps({"status": "ok", "tests": len(tests)}))
