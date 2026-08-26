from __future__ import annotations

from typing import Mapping

import numpy as np

QUAD_AIR_DRAG: dict[str, float | int | str] = {
    "air_density_kg_m3": 1.225,
    "drag_coefficient": 1.05,
    "frontal_area_m2": 0.18,
    "mass_kg": 2.5,
    "gravity_mps2": 9.81,
    "rotor_count": 4,
    "propeller_diameter_m": 0.15,
    "induced_power_factor": 1.15,
    "avionics_power_w": 18.0,
    "sensor_power_w": 8.0,
    "energy_unit_scale": 0.03,
    "linear_air_drag_per_s": 0.28,
    "model_id": "quadratic-air-relative-v1",
    "honesty": (
        "QUADRATIC AIR-RELATIVE DRAG · MOMENTUM-THEORY INDUCED · "
        "NOT BLADE-ELEMENT / NOT NS"
    ),
}


def _cfg(overrides: Mapping[str, float | int | str] | None = None) -> dict:
    merged = dict(QUAD_AIR_DRAG)
    if overrides:
        merged.update(overrides)
    return merged


def parasite_drag_per_m(config: Mapping[str, float | int | str] | None = None) -> float:
    cfg = _cfg(config)
    return float(
        0.5
        * cfg["air_density_kg_m3"]
        * cfg["drag_coefficient"]
        * cfg["frontal_area_m2"]
        / max(float(cfg["mass_kg"]), 1e-9)
    )


def rotor_disk_area_m2(config: Mapping[str, float | int | str] | None = None) -> float:
    cfg = _cfg(config)
    radius = float(cfg["propeller_diameter_m"]) * 0.5
    return float(cfg["rotor_count"]) * np.pi * radius * radius


def hover_induced_velocity_mps(config: Mapping[str, float | int | str] | None = None) -> float:
    cfg = _cfg(config)
    weight_n = float(cfg["mass_kg"]) * float(cfg["gravity_mps2"])
    return float(np.sqrt(weight_n / (2.0 * float(cfg["air_density_kg_m3"]) * max(rotor_disk_area_m2(cfg), 1e-9))))


def relative_air_velocity(ground: np.ndarray, wind: np.ndarray) -> np.ndarray:
    ground_v = np.asarray(ground, dtype=float)
    wind_v = np.asarray(wind, dtype=float)
    if ground_v.shape != wind_v.shape or ground_v.ndim != 1:
        raise ValueError("ground and wind must be 1-D vectors of the same shape")
    if not np.all(np.isfinite(ground_v)) or not np.all(np.isfinite(wind_v)):
        raise ValueError("ground and wind must be finite")
    return ground_v - wind_v


def integrate_quadratic_air_drag(
    ground: np.ndarray,
    wind: np.ndarray,
    dt_s: float,
    *,
    k_per_m: float | None = None,
    linear_per_s: float | None = None,
    config: Mapping[str, float | int | str] | None = None,
) -> np.ndarray:
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        return np.asarray(ground, dtype=float).copy()
    cfg = _cfg(config)
    k = parasite_drag_per_m(cfg) if k_per_m is None else float(k_per_m)
    linear = float(cfg["linear_air_drag_per_s"] if linear_per_s is None else linear_per_s)
    k = k if np.isfinite(k) and k > 0.0 else 0.0
    linear = linear if np.isfinite(linear) and linear > 0.0 else 0.0
    ground_v = np.asarray(ground, dtype=float)
    wind_v = np.asarray(wind, dtype=float)
    if k == 0.0 and linear == 0.0:
        return ground_v.copy()
    air = relative_air_velocity(ground_v, wind_v)
    air_speed = float(np.linalg.norm(air))
    denom = 1.0 + dt_s * (k * air_speed + linear)
    return wind_v + air / denom


def evaluate_physical_drag_power(
    ground: np.ndarray,
    wind: np.ndarray,
    config: Mapping[str, float | int | str] | None = None,
) -> dict[str, float]:
    cfg = _cfg(config)
    air = relative_air_velocity(ground, wind)
    relative_air_speed = float(np.linalg.norm(air))
    drag_force_n = float(
        0.5
        * cfg["air_density_kg_m3"]
        * cfg["drag_coefficient"]
        * cfg["frontal_area_m2"]
        * relative_air_speed
        * relative_air_speed
    )
    parasite_power_w = drag_force_n * relative_air_speed
    weight_n = float(cfg["mass_kg"]) * float(cfg["gravity_mps2"])
    induced_hover = hover_induced_velocity_mps(cfg)
    induced_hover_w = float(cfg["induced_power_factor"]) * weight_n * induced_hover
    induced_power_w = induced_hover_w / np.sqrt(1.0 + (relative_air_speed / max(induced_hover, 1e-6)) ** 2)
    climb = float(ground[1]) if np.asarray(ground).shape == (3,) else 0.0
    climb_power_w = max(0.0, climb) * weight_n
    total_power_w = (
        float(cfg["avionics_power_w"])
        + float(cfg["sensor_power_w"])
        + parasite_power_w
        + float(induced_power_w)
        + climb_power_w
    )
    return {
        "relative_air_speed": relative_air_speed,
        "drag_force_n": drag_force_n,
        "parasite_power_w": parasite_power_w,
        "induced_power_w": float(induced_power_w),
        "climb_power_w": climb_power_w,
        "total_power_w": total_power_w,
        "hover_induced_velocity_mps": induced_hover,
    }
