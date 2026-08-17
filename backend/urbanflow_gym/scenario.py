from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .geometry import AxisAlignedPrism, UrbanGeometry
from .wind import SyntheticWakeWindProvider, WindProvider


DEFAULT_HELD_OUT_SEEDS = (10_007, 10_009, 10_037)
TRAINING_SEED_RANGE = (0, 9_999)
HELD_OUT_SEED_MIN = 10_000


@dataclass(frozen=True)
class PublicMissionContext:
    scenario_id: str
    seed: int
    geometry: UrbanGeometry
    start_xy: np.ndarray
    goal_xy: np.ndarray
    initial_heading_rad: float
    known_inlet_velocity_xy: np.ndarray


@dataclass(frozen=True)
class UrbanFlowScenario:
    scenario_id: str
    seed: int
    geometry: UrbanGeometry
    start_xy: np.ndarray
    goal_xy: np.ndarray
    initial_heading_rad: float
    known_inlet_velocity_xy: np.ndarray
    wind_provider: WindProvider
    randomization: dict

    def __post_init__(self) -> None:
        start = np.asarray(self.start_xy, dtype=float)
        goal = np.asarray(self.goal_xy, dtype=float)
        inlet = np.asarray(self.known_inlet_velocity_xy, dtype=float)
        if start.shape != (2,) or goal.shape != (2,) or inlet.shape != (2,):
            raise ValueError("scenario start, goal, and inlet must be 2D vectors")
        if not np.all(np.isfinite(np.concatenate([start, goal, inlet]))):
            raise ValueError("scenario vectors must be finite")
        if not self.geometry.point_is_free(start, agent_radius_m=1.25, margin_m=1.0):
            raise ValueError("scenario start is not collision-free")
        if not self.geometry.point_is_free(goal, agent_radius_m=1.25, margin_m=1.0):
            raise ValueError("scenario goal is not collision-free")
        if not np.allclose(self.wind_provider.inlet_velocity, inlet, atol=1e-9):
            raise ValueError("wind provider inlet does not match known scenario inlet")
        object.__setattr__(self, "start_xy", start)
        object.__setattr__(self, "goal_xy", goal)
        object.__setattr__(self, "known_inlet_velocity_xy", inlet)
        object.__setattr__(self, "initial_heading_rad", float(self.initial_heading_rad))
        object.__setattr__(self, "randomization", dict(self.randomization))

    def public_context(self) -> PublicMissionContext:
        """Return the complete baseline/actor-visible mission context only."""

        return PublicMissionContext(
            scenario_id=self.scenario_id,
            seed=self.seed,
            geometry=self.geometry,
            start_xy=self.start_xy.copy(),
            goal_xy=self.goal_xy.copy(),
            initial_heading_rad=self.initial_heading_rad,
            known_inlet_velocity_xy=self.known_inlet_velocity_xy.copy(),
        )

    def to_public_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "split": "held_out" if self.seed >= HELD_OUT_SEED_MIN else "training_or_development",
            "start_xy": self.start_xy.tolist(),
            "goal_xy": self.goal_xy.tolist(),
            "initial_heading_rad": self.initial_heading_rad,
            "known_inlet_velocity_mps": self.known_inlet_velocity_xy.tolist(),
            "geometry": self.geometry.to_dict(),
            "randomization": self.randomization,
        }


def make_seeded_scenario(seed: int, randomize_domain: bool = True) -> UrbanFlowScenario:
    """Build the legacy rectangular synthetic fixture (never used by live endpoints)."""
    seed = int(seed)
    rng = np.random.default_rng(seed)
    bounds = (0.0, 120.0, 0.0, 120.0)
    base_prisms = (
        ("west_block", (24.0, 15.0), (44.0, 55.0), 34.0),
        ("central_block", (48.0, 48.0), (70.0, 72.0), 46.0),
        ("east_block", (78.0, 18.0), (98.0, 58.0), 30.0),
        ("northwest_block", (20.0, 78.0), (42.0, 105.0), 27.0),
        ("northeast_block", (68.0, 82.0), (100.0, 102.0), 41.0),
    )
    prisms: list[AxisAlignedPrism] = []
    for obstacle_id, minimum, maximum, height in base_prisms:
        min_xy = np.asarray(minimum, dtype=float)
        max_xy = np.asarray(maximum, dtype=float)
        if randomize_domain:
            center_jitter = rng.uniform(-1.8, 1.8, size=2)
            size_scale = rng.uniform(0.94, 1.06, size=2)
            center = (min_xy + max_xy) * 0.5 + center_jitter
            half_size = (max_xy - min_xy) * 0.5 * size_scale
            min_xy = center - half_size
            max_xy = center + half_size
            height = float(height * rng.uniform(0.92, 1.08))
        prisms.append(AxisAlignedPrism(obstacle_id, min_xy, max_xy, height))
    geometry = UrbanGeometry(bounds, prisms, flight_altitude_m=18.0)

    mission_pairs = (
        (np.array([8.0, 20.0]), np.array([112.0, 100.0])),
        (np.array([8.0, 105.0]), np.array([112.0, 15.0])),
        (np.array([7.0, 61.0]), np.array([113.0, 61.0])),
        (np.array([59.0, 7.0]), np.array([59.0, 113.0])),
        (np.array([10.0, 10.0]), np.array([110.0, 110.0])),
        (np.array([110.0, 10.0]), np.array([10.0, 110.0])),
    )
    mission_index = int(rng.integers(0, len(mission_pairs))) if randomize_domain else 0
    start, goal = (point.copy() for point in mission_pairs[mission_index])
    if randomize_domain:
        start += rng.uniform(-2.2, 2.2, size=2)
        goal += rng.uniform(-2.2, 2.2, size=2)

    inlet_angle = float(rng.uniform(-math.pi, math.pi)) if randomize_domain else 0.0
    inlet_speed = float(rng.uniform(2.5, 5.5)) if randomize_domain else 4.0
    inlet = inlet_speed * np.array([math.cos(inlet_angle), math.sin(inlet_angle)], dtype=float)
    wake_strength = float(rng.uniform(0.38, 0.62)) if randomize_domain else 0.48
    gust_amplitude = float(rng.uniform(0.08, 0.32)) if randomize_domain else 0.16
    initial_heading = math.atan2(goal[1] - start[1], goal[0] - start[0])
    if randomize_domain:
        initial_heading += float(rng.uniform(-0.18, 0.18))

    provider = SyntheticWakeWindProvider(
        geometry=geometry,
        inlet_velocity_xy=inlet,
        seed=seed,
        wake_strength=wake_strength,
        gust_amplitude_mps=gust_amplitude,
    )
    return UrbanFlowScenario(
        scenario_id=f"urbanflow-synthetic-fixture-{seed}",
        seed=seed,
        geometry=geometry,
        start_xy=start,
        goal_xy=goal,
        initial_heading_rad=initial_heading,
        known_inlet_velocity_xy=inlet,
        wind_provider=provider,
        randomization={
            "enabled": bool(randomize_domain),
            "mission_index": mission_index,
            "building_center_jitter_max_m": 1.8 if randomize_domain else 0.0,
            "building_size_scale_range": [0.94, 1.06] if randomize_domain else [1.0, 1.0],
            "inlet_speed_range_mps": [2.5, 5.5] if randomize_domain else [4.0, 4.0],
            "inlet_direction_range_rad": [-math.pi, math.pi] if randomize_domain else [0.0, 0.0],
            "wake_strength": wake_strength,
            "gust_amplitude_mps": gust_amplitude,
        },
    )


def make_custom_scenario(
    *,
    scenario_id: str,
    seed: int,
    geometry: UrbanGeometry,
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    wind_provider: WindProvider,
    initial_heading_rad: float | None = None,
) -> UrbanFlowScenario:
    start = np.asarray(start_xy, dtype=float)
    goal = np.asarray(goal_xy, dtype=float)
    heading = initial_heading_rad
    if heading is None:
        heading = math.atan2(goal[1] - start[1], goal[0] - start[0])
    return UrbanFlowScenario(
        scenario_id=scenario_id,
        seed=int(seed),
        geometry=geometry,
        start_xy=start,
        goal_xy=goal,
        initial_heading_rad=float(heading),
        known_inlet_velocity_xy=wind_provider.inlet_velocity,
        wind_provider=wind_provider,
        randomization={"enabled": False, "custom": True},
    )
