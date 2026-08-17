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

import main


def _request_json(method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    sent: list[dict] = []
    requests_to_receive = [
        {
            "type": "http.request",
            "body": body,
            "more_body": False,
        }
    ]

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
    status = next(
        message["status"] for message in sent if message["type"] == "http.response.start"
    )
    response_body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    return status, json.loads(response_body)


def test_versioned_spec_and_alias_are_explicit_about_status_boundaries() -> None:
    for path in ("/urbanflow-gym/spec", "/api/urbanflow-gym/spec"):
        status, payload = _request_json("GET", path)
        assert status == 200
        assert payload["contract_version"] == "1.0.0"
        assert payload["environment_id"] == "UrbanFlowGym-v1"
        assert payload["labels"]["primary"] == "URBANFLOW GYM / LIVE OSM WORLD · NOT TRAINED"
        assert payload["labels"]["world"] == "LIVE OSM WORLD"
        assert payload["labels"]["full_flow_access"] == "NO"
        assert payload["action"]["frame"] == "vehicle_local_forward_left"
        assert payload["status"]["trained_policy_available"] is False
        assert payload["status"]["training_executed_on_this_machine"] is False
        assert payload["status"]["real_3d_navier_stokes_validated"] is False
        assert payload["leakage_guard"]["status"] == "passed"
        assert payload["future_external_cfd_evaluation"]["validation_claim"] == "none"


def test_bounded_evaluate_returns_aggregate_comparison_without_trajectories() -> None:
    status, payload = _request_json(
        "POST",
        "/urbanflow-gym/evaluate",
        {"seeds": [10_007], "max_steps": 50, "save_artifact": False},
    )

    assert status == 200
    assert payload["status"] == "ok"
    assert payload["scenario_kind"] == "synthetic_fixture"
    assert payload["scenario_id"] is None
    assert payload["contract_version"] == "1.0.0"
    assert payload["artifact_path"] is None
    assert payload["evaluation_config"]["seeds"] == [10_007]
    assert payload["evaluation_config"]["max_steps"] == 50
    assert payload["policy_status"] == "not_trained"
    assert payload["policy_had_privileged_flow_access"] is False
    assert payload["policy_full_flow_access"] is False
    assert payload["synthetic_hidden_flow"] is True
    assert payload["real_cfd_validation_status"] == "not_run_interface_only"
    assert payload["dynamics_source"]["navier_stokes_cfd"] is False
    assert set(payload["baselines"]) == {
        "direct_goal",
        "shortest_path",
        "wind_aware_inlet",
    }
    for baseline in payload["baselines"].values():
        assert baseline["uses_hidden_flow"] is False
        assert baseline["aggregate"]["episodes"] == 1
        assert "episodes" not in baseline
        assert "trajectory" not in baseline


def test_evaluate_validation_rejects_unbounded_or_coerced_requests_before_work() -> None:
    invalid_payloads = (
        {"seeds": [], "max_steps": 50},
        {"seeds": [1, 2, 3, 4, 5, 6], "max_steps": 50},
        {"seeds": [-1], "max_steps": 50},
        {"seeds": [True], "max_steps": 50},
        {"seeds": [1], "max_steps": 49},
        {"seeds": [1], "max_steps": 501},
        {"seeds": [1], "max_steps": 50.0},
        {"seeds": [1], "max_steps": 50, "unexpected": "field"},
    )
    with patch.object(main, "run_baseline_evaluation") as evaluate:
        for payload in invalid_payloads:
            status, response = _request_json(
                "POST",
                "/urbanflow-gym/evaluate",
                payload,
            )
            assert status == 422, response
    evaluate.assert_not_called()


def test_legacy_rl_routes_remain_available() -> None:
    for path in ("/rl/spec", "/api/rl/spec"):
        status, payload = _request_json("GET", path)
        assert status == 200
        assert payload["policy_had_privileged_flow_access"] is False
        assert "environment" in payload


if __name__ == "__main__":
    tests = (
        test_versioned_spec_and_alias_are_explicit_about_status_boundaries,
        test_bounded_evaluate_returns_aggregate_comparison_without_trajectories,
        test_evaluate_validation_rejects_unbounded_or_coerced_requests_before_work,
        test_legacy_rl_routes_remain_available,
    )
    for test in tests:
        test()
    print(json.dumps({"status": "ok", "tests": len(tests)}))
