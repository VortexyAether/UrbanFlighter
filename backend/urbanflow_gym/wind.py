from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

import numpy as np

from .geometry import UrbanGeometry


@runtime_checkable
class WindProvider(Protocol):
    """Hidden dynamics boundary used by both synthetic and offline CFD fields."""

    @property
    def inlet_velocity(self) -> np.ndarray:
        ...

    def velocity_at(self, position_xy: np.ndarray, time_s: float) -> np.ndarray:
        ...

    def source_metadata(self) -> dict:
        ...


@dataclass(frozen=True)
class ConstantWindProvider:
    velocity_xy: np.ndarray

    def __post_init__(self) -> None:
        velocity = np.asarray(self.velocity_xy, dtype=float)
        if velocity.shape != (2,) or not np.all(np.isfinite(velocity)):
            raise ValueError("constant wind must be a finite 2D vector")
        object.__setattr__(self, "velocity_xy", velocity)

    @property
    def inlet_velocity(self) -> np.ndarray:
        return self.velocity_xy.copy()

    def velocity_at(self, position_xy: np.ndarray, time_s: float) -> np.ndarray:
        del position_xy, time_s
        return self.velocity_xy.copy()

    def source_metadata(self) -> dict:
        return {
            "kind": "constant_test_or_configured_wind",
            "inlet_velocity_mps": self.velocity_xy.tolist(),
            "navier_stokes_cfd": False,
            "hidden_from_actor_except_known_inlet": True,
        }


class SyntheticWakeWindProvider:
    """Cheap deterministic inlet + potential-flow-ish deflection + wake proxy.

    This is deliberately inexpensive training dynamics. It is neither a
    Navier--Stokes solver nor a claim of real CFD validation.
    """

    def __init__(
        self,
        geometry: UrbanGeometry,
        inlet_velocity_xy: np.ndarray,
        seed: int,
        wake_strength: float = 0.48,
        gust_amplitude_mps: float = 0.22,
    ) -> None:
        inlet = np.asarray(inlet_velocity_xy, dtype=float)
        if inlet.shape != (2,) or not np.all(np.isfinite(inlet)):
            raise ValueError("inlet_velocity_xy must be a finite 2D vector")
        self.geometry = geometry
        self._inlet_velocity = inlet
        self.seed = int(seed)
        self.wake_strength = float(np.clip(wake_strength, 0.0, 0.85))
        self.gust_amplitude_mps = float(np.clip(gust_amplitude_mps, 0.0, 1.0))
        self._phase = (self.seed % 10_007) * 0.017453292519943295

    @property
    def inlet_velocity(self) -> np.ndarray:
        return self._inlet_velocity.copy()

    def velocity_at(self, position_xy: np.ndarray, time_s: float) -> np.ndarray:
        point = np.asarray(position_xy, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("wind sample position must be a finite 2D vector")

        inlet_speed = float(np.linalg.norm(self._inlet_velocity))
        if inlet_speed < 1e-9:
            direction = np.array([1.0, 0.0], dtype=float)
            cross_direction = np.array([0.0, 1.0], dtype=float)
        else:
            direction = self._inlet_velocity / inlet_speed
            cross_direction = np.array([-direction[1], direction[0]], dtype=float)

        velocity = self._inlet_velocity.copy()
        for prism in self.geometry.relevant_prisms():
            if np.all(point >= prism.min_xy) and np.all(point <= prism.max_xy):
                return np.zeros(2, dtype=float)

            relative = point - prism.center_xy
            along_center = float(np.dot(relative, direction))
            cross_center = float(np.dot(relative, cross_direction))
            half_size = prism.size_xy * 0.5
            half_along = float(np.dot(np.abs(direction), half_size))
            half_cross = float(np.dot(np.abs(cross_direction), half_size))
            equivalent_radius = max(2.0, math.sqrt(float(np.prod(prism.size_xy))) * 0.42)
            radial_distance = max(float(np.linalg.norm(relative)), equivalent_radius * 1.04)

            if radial_distance < equivalent_radius * 4.5 and inlet_speed > 0.0:
                x = along_center
                y = cross_center
                radius_squared = equivalent_radius * equivalent_radius
                radius_fourth = radial_distance**4
                potential_along = inlet_speed * (
                    -radius_squared * (x * x - y * y) / max(radius_fourth, 1e-9)
                )
                potential_cross = inlet_speed * (
                    -2.0 * radius_squared * x * y / max(radius_fourth, 1e-9)
                )
                blend = 0.34 * (1.0 - radial_distance / (equivalent_radius * 4.5))
                velocity += blend * (
                    potential_along * direction + potential_cross * cross_direction
                )

            downstream = along_center - half_along
            if downstream > 0.0 and inlet_speed > 0.0:
                wake_width = max(3.0, half_cross * 1.25 + 0.10 * downstream)
                centerline = math.exp(-((cross_center / wake_width) ** 2))
                decay = math.exp(-downstream / max(18.0, 3.2 * half_along))
                deficit = self.wake_strength * centerline * decay
                velocity -= direction * inlet_speed * deficit
                side_sign = 1.0 if cross_center >= 0.0 else -1.0
                velocity += cross_direction * side_sign * inlet_speed * deficit * 0.08

        if self.gust_amplitude_mps > 0.0:
            gust_phase = (
                self._phase
                + point[0] * 0.031
                - point[1] * 0.023
                + float(time_s) * 0.41
            )
            gust = self.gust_amplitude_mps * np.array(
                [math.sin(gust_phase), 0.55 * math.cos(gust_phase * 0.83 + 0.4)],
                dtype=float,
            )
            velocity += gust

        speed_limit = max(2.0, inlet_speed * 2.25 + self.gust_amplitude_mps)
        speed = float(np.linalg.norm(velocity))
        if speed > speed_limit:
            velocity *= speed_limit / speed
        return velocity

    def source_metadata(self) -> dict:
        return {
            "kind": "synthetic_potential_flowish_wake_proxy",
            "model": "uniform_inlet_plus_low_cost_deflection_wake_and_deterministic_gust",
            "inlet_velocity_mps": self._inlet_velocity.tolist(),
            "seed": self.seed,
            "navier_stokes_cfd": False,
            "real_cfd_validation": False,
            "hidden_from_actor_except_known_inlet": True,
            "allowed_uses": ["dynamics", "reward", "metrics", "privileged_critic"],
        }
