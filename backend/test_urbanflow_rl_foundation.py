from __future__ import annotations

import json
import os
from pathlib import Path
import sys


VENV_PYTHON = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
if ".venv" not in sys.executable and VENV_PYTHON.exists():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])

import numpy as np

from urbanflow_gym.env import UrbanFlowConfig, UrbanFlowEnv
from urbanflow_gym.eval_policy import evaluate_baseline, evaluate_random_policy
from urbanflow_gym.geometry import AxisAlignedPrism, UrbanGeometry
from urbanflow_gym.observation import RADAR_RAY_COUNT
from urbanflow_gym.scenario import make_custom_scenario
from urbanflow_gym.schemas import ACTOR_OBSERVATION_DIM, leakage_guard_report
from urbanflow_gym.wind import ConstantWindProvider


class _SplitInletWind:
    def __init__(self, inlet: np.ndarray, hidden: np.ndarray) -> None:
        self._inlet = np.asarray(inlet, dtype=float)
        self._hidden = np.asarray(hidden, dtype=float)

    @property
    def inlet_velocity(self) -> np.ndarray:
        return self._inlet.copy()

    def velocity_at(self, position_xy: np.ndarray, time_s: float) -> np.ndarray:
        del position_xy, time_s
        return self._hidden.copy()

    def source_metadata(self) -> dict:
        return {"kind": "split_inlet_hidden_test"}


def _scenario(inlet: np.ndarray, hidden: np.ndarray, scenario_id: str):
    geometry = UrbanGeometry((0.0, 40.0, 0.0, 40.0), (), flight_altitude_m=18.0)
    return make_custom_scenario(
        scenario_id=scenario_id,
        seed=19,
        geometry=geometry,
        start_xy=np.array([6.0, 8.0]),
        goal_xy=np.array([32.0, 30.0]),
        wind_provider=_SplitInletWind(inlet, hidden),
    )


def test_actor_cannot_recover_hidden_local_wind() -> None:
    inlet = np.array([4.0, 0.5])
    first = UrbanFlowEnv(
        UrbanFlowConfig(max_steps=4),
        fixed_scenario=_scenario(inlet, np.array([4.0, 0.5]), "same-inlet-a"),
    )
    second = UrbanFlowEnv(
        UrbanFlowConfig(max_steps=4),
        fixed_scenario=_scenario(inlet, np.array([-2.0, 3.0]), "same-inlet-b"),
    )
    first_obs, _ = first.reset(seed=19)
    second_obs, _ = second.reset(seed=19)
    np.testing.assert_array_equal(first_obs, second_obs)

    action = np.array([0.4, -0.2])
    first.step(action)
    second.step(action)
    first_air = first.actor_snapshot().relative_air_velocity_estimate_xy
    second_air = second.actor_snapshot().relative_air_velocity_estimate_xy
    np.testing.assert_allclose(
        first_air,
        first.ground_velocity_xy - inlet,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        second_air,
        second.ground_velocity_xy - inlet,
        atol=1e-9,
    )
    assert first.metrics()["relative_air_speed_energy"] != second.metrics()[
        "relative_air_speed_energy"
    ]


def test_radar_is_present_and_honest() -> None:
    env = UrbanFlowEnv(UrbanFlowConfig(max_steps=2))
    observation, _ = env.reset(seed=11)
    snapshot = env.actor_snapshot()
    assert observation.shape == (ACTOR_OBSERVATION_DIM,) == (49,)
    assert snapshot.radar_ranges_m.shape == (RADAR_RAY_COUNT,)
    assert snapshot.radar_range_rate_mps.shape == (RADAR_RAY_COUNT,)
    assert leakage_guard_report()["dimension"] == 49


def test_quadratic_drag_metrics_and_random_eval() -> None:
    blocking = AxisAlignedPrism(
        "wall",
        np.array([16.0, 10.0]),
        np.array([18.0, 30.0]),
        30.0,
    )
    geometry = UrbanGeometry((0.0, 40.0, 0.0, 40.0), (blocking,), flight_altitude_m=18.0)
    scenario = make_custom_scenario(
        scenario_id="foundation-eval",
        seed=10007,
        geometry=geometry,
        start_xy=np.array([6.0, 20.0]),
        goal_xy=np.array([34.0, 20.0]),
        wind_provider=ConstantWindProvider(np.array([3.0, 0.0])),
    )
    config = UrbanFlowConfig(max_steps=40)
    env = UrbanFlowEnv(config, fixed_scenario=scenario)
    random_metrics = evaluate_random_policy(env, seed=10007, max_steps=40)
    planned = evaluate_baseline(
        UrbanFlowEnv(config, fixed_scenario=scenario),
        "shortest_path",
        40,
    )
    assert "parasite_energy_j" in random_metrics
    assert random_metrics["drag_model_id"] == "quadratic-air-relative-v1"
    assert planned["collision_count"] == 0
    assert planned["success"] is True


if __name__ == "__main__":
    tests = (
        test_actor_cannot_recover_hidden_local_wind,
        test_radar_is_present_and_honest,
        test_quadratic_drag_metrics_and_random_eval,
    )
    for test in tests:
        test()
    print(json.dumps({"status": "ok", "tests": len(tests)}))
