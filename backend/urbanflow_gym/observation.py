from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .geometry import UrbanGeometry
from .schemas import (
    ACTOR_OBSERVATION_DIM,
    ACTOR_OBSERVATION_HIGH,
    ACTOR_OBSERVATION_LOW,
    LIDAR_RAY_COUNT,
    PRIVILEGED_CRITIC_DIM,
)


LIDAR_LOCAL_ANGLES_RAD = (
    np.arange(LIDAR_RAY_COUNT, dtype=float) / LIDAR_RAY_COUNT * (2.0 * math.pi)
)


@dataclass(frozen=True)
class ActorObservableState:
    """The only state object accepted by the actor observation builder."""

    position_xy: np.ndarray
    ground_velocity_xy: np.ndarray
    heading_rad: float
    relative_air_velocity_estimate_xy: np.ndarray
    known_inlet_velocity_xy: np.ndarray
    goal_xy: np.ndarray
    previous_action: np.ndarray

    def __post_init__(self) -> None:
        vectors = {
            "position_xy": self.position_xy,
            "ground_velocity_xy": self.ground_velocity_xy,
            "relative_air_velocity_estimate_xy": self.relative_air_velocity_estimate_xy,
            "known_inlet_velocity_xy": self.known_inlet_velocity_xy,
            "goal_xy": self.goal_xy,
            "previous_action": self.previous_action,
        }
        for name, value in vectors.items():
            vector = np.asarray(value, dtype=float)
            if vector.shape != (2,) or not np.all(np.isfinite(vector)):
                raise ValueError(f"{name} must be a finite 2D vector")
            object.__setattr__(self, name, vector)
        if not math.isfinite(self.heading_rad):
            raise ValueError("heading_rad must be finite")
        object.__setattr__(self, "heading_rad", float(self.heading_rad))


@dataclass(frozen=True)
class ActorSnapshot:
    """Decoded actor-visible state for deterministic non-learning baselines."""

    position_xy: np.ndarray
    ground_velocity_xy: np.ndarray
    heading_rad: float
    relative_air_velocity_estimate_xy: np.ndarray
    known_inlet_velocity_xy: np.ndarray
    goal_xy: np.ndarray
    lidar_ranges_m: np.ndarray
    previous_action: np.ndarray


def build_actor_observation(
    observable: ActorObservableState,
    geometry: UrbanGeometry,
    *,
    max_ground_speed_mps: float,
    relative_air_speed_limit_mps: float,
    lidar_range_m: float,
) -> tuple[np.ndarray, ActorSnapshot]:
    """Build the v1 actor vector without accepting any hidden-provider object."""

    goal_delta = observable.goal_xy - observable.position_xy
    goal_distance = float(np.linalg.norm(goal_delta))
    goal_direction_world = goal_delta / goal_distance if goal_distance > 1e-9 else np.zeros(2)
    forward = np.array([math.cos(observable.heading_rad), math.sin(observable.heading_rad)])
    left = np.array([-forward[1], forward[0]])
    goal_direction_local = np.array(
        [float(np.dot(goal_direction_world, forward)), float(np.dot(goal_direction_world, left))]
    )
    world_angles = observable.heading_rad + LIDAR_LOCAL_ANGLES_RAD
    lidar_ranges = geometry.lidar_ranges(observable.position_xy, world_angles, lidar_range_m)
    values = np.concatenate(
        [
            geometry.normalize_position(observable.position_xy),
            observable.ground_velocity_xy / max(max_ground_speed_mps, 1e-9),
            np.array([math.sin(observable.heading_rad), math.cos(observable.heading_rad)]),
            observable.relative_air_velocity_estimate_xy / max(relative_air_speed_limit_mps, 1e-9),
            observable.known_inlet_velocity_xy / max(relative_air_speed_limit_mps, 1e-9),
            goal_delta / max(geometry.diagonal_m, 1e-9),
            np.array([goal_distance / max(geometry.diagonal_m, 1e-9)]),
            goal_direction_local,
            lidar_ranges / max(lidar_range_m, 1e-9),
            observable.previous_action,
        ]
    )
    if values.shape != (ACTOR_OBSERVATION_DIM,):
        raise RuntimeError(
            f"actor observation shape drifted to {values.shape}; expected {(ACTOR_OBSERVATION_DIM,)}"
        )
    observation = np.clip(values, ACTOR_OBSERVATION_LOW, ACTOR_OBSERVATION_HIGH).astype(np.float32)
    snapshot = ActorSnapshot(
        position_xy=observable.position_xy.copy(),
        ground_velocity_xy=observable.ground_velocity_xy.copy(),
        heading_rad=observable.heading_rad,
        relative_air_velocity_estimate_xy=observable.relative_air_velocity_estimate_xy.copy(),
        known_inlet_velocity_xy=observable.known_inlet_velocity_xy.copy(),
        goal_xy=observable.goal_xy.copy(),
        lidar_ranges_m=lidar_ranges.copy(),
        previous_action=observable.previous_action.copy(),
    )
    return observation, snapshot


def build_privileged_critic_state(
    actor_observation: np.ndarray,
    *,
    exact_local_wind_xy: np.ndarray,
    known_inlet_velocity_xy: np.ndarray,
    exact_clearance_m: float,
    lidar_range_m: float,
    elapsed_steps: int,
    max_steps: int,
    relative_air_speed_limit_mps: float,
) -> np.ndarray:
    actor = np.asarray(actor_observation, dtype=float)
    local = np.asarray(exact_local_wind_xy, dtype=float)
    inlet = np.asarray(known_inlet_velocity_xy, dtype=float)
    state = np.concatenate(
        [
            actor,
            np.clip(local / max(relative_air_speed_limit_mps, 1e-9), -1.0, 1.0),
            np.clip((local - inlet) / max(relative_air_speed_limit_mps, 1e-9), -1.0, 1.0),
            np.array([np.clip(exact_clearance_m / max(lidar_range_m, 1e-9), 0.0, 1.0)]),
            np.array([np.clip(elapsed_steps / max(max_steps, 1), 0.0, 1.0)]),
        ]
    ).astype(np.float32)
    if state.shape != (PRIVILEGED_CRITIC_DIM,):
        raise RuntimeError(
            f"privileged critic state shape drifted to {state.shape}; expected {(PRIVILEGED_CRITIC_DIM,)}"
        )
    return state
