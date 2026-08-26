from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from urban_flighter_physics.quadratic_air_drag import (
    QUAD_AIR_DRAG,
    evaluate_physical_drag_power,
    integrate_quadratic_air_drag,
    parasite_drag_per_m,
)


def test_hover_drifts_with_wind() -> None:
    ground = np.zeros(2)
    wind = np.array([8.0, 0.0])
    for _ in range(720):
        ground = integrate_quadratic_air_drag(ground, wind, 1.0 / 120.0)
    assert abs(ground[0] - 8.0) < 1.2
    assert abs(ground[1]) < 1e-9


def test_headwind_is_slower_and_costlier() -> None:
    def cruise(wind_x: float) -> tuple[float, float]:
        ground = np.zeros(2)
        wind = np.array([wind_x, 0.0])
        energy = 0.0
        for _ in range(360):
            ground = ground + np.array([8.0, 0.0]) / 120.0
            ground = integrate_quadratic_air_drag(ground, wind, 1.0 / 120.0)
            energy += evaluate_physical_drag_power(ground, wind)["total_power_w"] / 120.0
        return float(np.linalg.norm(ground)), energy

    tail_speed, tail_energy = cruise(6.0)
    head_speed, head_energy = cruise(-6.0)
    assert tail_speed > head_speed + 1.5
    assert head_energy > tail_energy * 1.05


def test_induced_falls_parasite_rises() -> None:
    hover = evaluate_physical_drag_power(np.zeros(3), np.zeros(3))
    cruise = evaluate_physical_drag_power(np.array([11.0, 0.0, 0.0]), np.zeros(3))
    assert hover["induced_power_w"] > cruise["induced_power_w"]
    assert cruise["parasite_power_w"] > hover["parasite_power_w"] + 10
    assert math.isfinite(parasite_drag_per_m())
    assert QUAD_AIR_DRAG["model_id"] == "quadratic-air-relative-v1"


if __name__ == "__main__":
    test_hover_drifts_with_wind()
    test_headwind_is_slower_and_costlier()
    test_induced_falls_parasite_rises()
    print("ok quadratic-air-relative-v1")
