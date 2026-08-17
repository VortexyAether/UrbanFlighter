from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
from unittest.mock import Mock, patch


VENV_PYTHON = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
if ".venv" not in sys.executable and VENV_PYTHON.exists():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

import requests
import numpy as np
from pydantic import ValidationError

import main
from services import wind
from services.geometry import _building_height


def _response(payload: dict) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = payload
    return response


def _post_json(path: str, payload: dict) -> tuple[int, bytes]:
    body = json.dumps(payload).encode("utf-8")
    sent: list[dict] = []
    requests_to_receive = [
        {
            "type": "http.request",
            "body": body,
            "more_body": False,
        },
    ]

    async def receive() -> dict:
        if requests_to_receive:
            return requests_to_receive.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "root_path": "",
    }
    asyncio.run(main.app(scope, receive, send))
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    response_body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return status, response_body


def test_open_meteo_request_and_success_metadata() -> None:
    payload = {
        "current": {
            "time": "2026-07-12T04:15",
            "wind_speed_10m": 7.25,
            "wind_direction_10m": 382.0,
        },
        "current_units": {
            "wind_speed_10m": "m/s",
            "wind_direction_10m": "°",
        },
    }
    with patch.object(wind.requests, "get", return_value=_response(payload)) as get:
        weather = wind.get_real_weather(37.45, 126.65)

    get.assert_called_once()
    endpoint = get.call_args.args[0]
    kwargs = get.call_args.kwargs
    assert endpoint == wind.OPEN_METEO_FORECAST_URL
    assert kwargs["params"]["wind_speed_unit"] == "ms"
    assert kwargs["params"]["current"] == "wind_speed_10m,wind_direction_10m"
    assert weather["wind_speed"] == 7.25
    assert weather["wind_deg"] == 22.0
    assert weather["units"]["wind_speed"] == "m/s"
    assert weather["source"]["provider"] == "open-meteo"
    assert weather["source"]["kind"] == "forecast_model_current_conditions"
    assert weather["fallback"] == {"used": False, "reason": None}


def test_weather_fallback_is_deterministic_and_labelled() -> None:
    with patch.object(wind.requests, "get", side_effect=requests.Timeout("offline")):
        first = wind.get_real_weather(37.45, 126.65)
        second = wind.get_real_weather(37.45, 126.65)

    assert first == second
    assert first["wind_speed"] == wind.FALLBACK_WIND_SPEED_MPS
    assert first["wind_deg"] == wind.FALLBACK_WIND_DIRECTION_DEG
    assert first["units"]["wind_speed"] == "m/s"
    assert first["source"]["kind"] == "deterministic_fallback"
    assert first["fallback"] == {
        "used": True,
        "reason": "weather_service_unavailable",
    }

    malformed = {"current": {"wind_speed_10m": float("nan"), "wind_direction_10m": 10.0}}
    with patch.object(wind.requests, "get", return_value=_response(malformed)):
        invalid = wind.get_real_weather(37.45, 126.65)
    assert invalid["wind_speed"] == wind.FALLBACK_WIND_SPEED_MPS
    assert invalid["fallback"]["used"] is True
    assert invalid["fallback"]["reason"] == "invalid_weather_response"


def test_coupled_grid_validation_accepts_cockpit_and_rejects_impractical_grid() -> None:
    request = main.FlowField2DRequest(solve_radius_m=400.0, grid_size_m=2.5)
    axis_cells = main.estimate_flow_field_2d_grid_axis_cells(
        request.solve_radius_m,
        request.grid_size_m,
    )
    assert axis_cells == 321
    assert axis_cells * axis_cells == 103_041
    assert axis_cells <= main.FLOW_FIELD_2D_MAX_GRID_AXIS_CELLS
    assert axis_cells * axis_cells <= main.FLOW_FIELD_2D_MAX_GRID_TOTAL_CELLS

    try:
        main.FlowField2DRequest(solve_radius_m=3000.0, grid_size_m=0.11)
    except ValidationError as exc:
        message = str(exc)
        assert "derived 2D solver grid" in message
        assert "exceeds production limits" in message
    else:
        raise AssertionError("impractical derived grid was accepted")

    for name, value in (("lat", float("nan")), ("lon", float("inf")), ("lat", 91.0), ("lon", -181.0)):
        try:
            main.FlowField2DRequest(**{name: value})
        except ValidationError:
            pass
        else:
            raise AssertionError(f"invalid {name} was accepted")

    try:
        main.FlowField2DRequest(geometry_radius_m=500.0, solve_radius_m=400.0)
    except ValidationError as exc:
        assert "geometry_radius_m cannot exceed solve_radius_m" in str(exc)
    else:
        raise AssertionError("geometry outside solver bounds was accepted")


def test_grid_estimator_and_meteorological_cardinals_match_solver_coordinates() -> None:
    for radius_m, grid_size_m in (
        (200.0, 80.0),
        (200.0, 32.0),
        (400.0, 2.5),
        (333.0, 7.3),
        (3000.0, 80.0),
    ):
        actual_grid = np.arange(
            -radius_m,
            radius_m + grid_size_m * 0.5,
            grid_size_m,
            dtype=np.float32,
        )
        assert main.estimate_flow_field_2d_grid_axis_cells(radius_m, grid_size_m) == len(actual_grid)

    for direction_from_north, expected_flow_to in (
        (0.0, (0.0, -5.0)),
        (90.0, (-5.0, 0.0)),
        (180.0, (0.0, 5.0)),
        (270.0, (5.0, 0.0)),
    ):
        inlet = main.wind_dir_to_inlet_vector(5.0, direction_from_north)
        np.testing.assert_allclose(inlet, expected_flow_to, atol=1e-6)


def test_http_validation_rejects_oversized_grid_before_external_work() -> None:
    with patch.object(main, "fetch_buildings") as fetch:
        status, response_body = _post_json(
            "/flow-fields/2d",
            {
                "lat": 37.45,
                "lon": 126.65,
                "geometry_radius_m": 400.0,
                "solve_radius_m": 3000.0,
                "grid_size_m": 0.11,
            },
        )

    assert status == 422
    assert b"derived 2D solver grid" in response_body
    fetch.assert_not_called()


def test_flow_endpoint_preserves_weather_provenance_without_allocating() -> None:
    weather = {
        "wind_speed": 6.5,
        "wind_deg": 90.0,
        "description": "Open-Meteo forecast-model current wind at 10 m",
        "units": {"wind_speed": "m/s", "wind_deg": "degrees_from_north"},
        "source": {
            "provider": "open-meteo",
            "kind": "forecast_model_current_conditions",
            "variable_height_m": 10.0,
        },
        "fallback": {"used": False, "reason": None},
    }
    field = {"nx": 1, "ny": 1, "ux": [-6.5], "uy": [0.0], "mask": [0]}
    request = main.FlowField2DRequest(
        solve_radius_m=400.0,
        grid_size_m=2.5,
        use_real_weather=True,
    )
    with (
        patch.object(main, "fetch_buildings", return_value=[]) as fetch,
        patch.object(main, "get_real_weather", return_value=weather) as get_weather,
        patch.object(main, "compute_cfd_lite_b_flow_2d", return_value=field) as solve,
    ):
        response = main.create_flow_field_2d(request)

    fetch.assert_called_once()
    get_weather.assert_called_once()
    solve.assert_called_once()
    assert response["weather"]["wind_speed"] == 6.5
    assert response["weather"]["units"]["wind_speed"] == "m/s"
    assert response["weather"]["source"]["provider"] == "open-meteo"
    assert response["weather"]["fallback"] == {"used": False, "reason": None}
    assert response["field"] is field


def test_configured_wind_is_not_mislabelled_as_live_or_fallback() -> None:
    request = main.FlowField2DRequest(use_real_weather=False)
    with (
        patch.object(main, "fetch_buildings", return_value=[]),
        patch.object(main, "get_real_weather") as get_weather,
        patch.object(main, "compute_cfd_lite_b_flow_2d", return_value={}),
    ):
        response = main.create_flow_field_2d(request)

    get_weather.assert_not_called()
    assert response["weather"]["source"]["kind"] == "configured_baseline"
    assert response["weather"]["fallback"]["used"] is False
    assert response["weather"]["description"] == "Configured deterministic baseline wind"


def test_osm_height_sources_are_preserved_or_deterministically_labelled() -> None:
    assert _building_height({"height": "42.5 m"}) == (42.5, "osm:height")
    assert _building_height({"building:levels": "5"}) == (
        17.5,
        "osm:building:levels_estimate_3.5m_per_level",
    )
    first = _building_height({})
    second = _building_height({})
    assert first == second == (10.0, "deterministic_default_missing_osm_height")


if __name__ == "__main__":
    tests = (
        test_open_meteo_request_and_success_metadata,
        test_weather_fallback_is_deterministic_and_labelled,
        test_coupled_grid_validation_accepts_cockpit_and_rejects_impractical_grid,
        test_grid_estimator_and_meteorological_cardinals_match_solver_coordinates,
        test_http_validation_rejects_oversized_grid_before_external_work,
        test_flow_endpoint_preserves_weather_provenance_without_allocating,
        test_configured_wind_is_not_mislabelled_as_live_or_fallback,
        test_osm_height_sources_are_preserved_or_deterministically_labelled,
    )
    for test in tests:
        test()
    print(json.dumps({"status": "ok", "tests": len(tests)}))
