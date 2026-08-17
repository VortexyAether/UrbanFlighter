from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .spaces import BoxSpec


CONTRACT_VERSION = "1.0.0"
ENVIRONMENT_ID = "UrbanFlowGym-v1"
ACTOR_OBSERVATION_SCHEMA_ID = "urbanflow.actor_observation.v1"
PRIVILEGED_CRITIC_SCHEMA_ID = "urbanflow.privileged_critic_state.v1"
ACTION_SCHEMA_ID = "urbanflow.local_guidance_action.v1"
REWARD_SCHEMA_ID = "urbanflow.reward_terms.v1"
METRICS_SCHEMA_ID = "urbanflow.episode_metrics.v1"
LIDAR_RAY_COUNT = 16


@dataclass(frozen=True)
class VectorFieldSpec:
    name: str
    size: int
    low: float
    high: float
    units: str
    source: str
    description: str

    def to_dict(self, offset: int) -> dict:
        return {
            "name": self.name,
            "offset": offset,
            "size": self.size,
            "low": self.low,
            "high": self.high,
            "units": self.units,
            "source": self.source,
            "description": self.description,
        }


ACTOR_OBSERVATION_FIELDS = (
    VectorFieldSpec(
        "position_xy_normalized",
        2,
        -1.0,
        1.0,
        "domain_fraction",
        "own_odometry",
        "Own planar odometry normalized to the known domain bounds.",
    ),
    VectorFieldSpec(
        "ground_velocity_xy_normalized",
        2,
        -1.0,
        1.0,
        "max_ground_speed_fraction",
        "own_kinematics",
        "Onboard ground-velocity estimate.",
    ),
    VectorFieldSpec(
        "heading_sin_cos",
        2,
        -1.0,
        1.0,
        "unitless",
        "own_kinematics",
        "Sine and cosine of onboard heading.",
    ),
    VectorFieldSpec(
        "relative_air_velocity_xy_normalized",
        2,
        -1.0,
        1.0,
        "relative_air_speed_limit_fraction",
        "onboard_relative_air_velocity_estimate",
        "Onboard relative-air-velocity estimate: ground velocity minus sensed local air motion.",
    ),
    VectorFieldSpec(
        "known_inlet_velocity_xy_normalized",
        2,
        -1.0,
        1.0,
        "relative_air_speed_limit_fraction",
        "known_inlet",
        "Mission-known inlet velocity, not a spatial flow sample.",
    ),
    VectorFieldSpec(
        "goal_delta_xy_normalized",
        2,
        -1.0,
        1.0,
        "domain_diagonal_fraction",
        "goal_route_context",
        "Goal displacement from own odometry.",
    ),
    VectorFieldSpec(
        "goal_distance_normalized",
        1,
        0.0,
        1.0,
        "domain_diagonal_fraction",
        "goal_route_context",
        "Scalar goal range.",
    ),
    VectorFieldSpec(
        "goal_direction_local_xy",
        2,
        -1.0,
        1.0,
        "unitless",
        "goal_route_context",
        "Unit goal direction expressed in the vehicle-local frame.",
    ),
    VectorFieldSpec(
        "lidar_ranges_normalized",
        LIDAR_RAY_COUNT,
        0.0,
        1.0,
        "sensor_range_fraction",
        "deterministic_geometry_lidar",
        "Fixed-angle deterministic ranges against the registered polygon geometry and domain boundary.",
    ),
    VectorFieldSpec(
        "previous_action",
        2,
        -1.0,
        1.0,
        "normalized_guidance_command",
        "recent_observable_history",
        "Most recent actor command for action smoothness and short observable history.",
    ),
)

ACTOR_OBSERVATION_DIM = sum(field.size for field in ACTOR_OBSERVATION_FIELDS)

PRIVILEGED_CRITIC_FIELDS = (
    VectorFieldSpec(
        "actor_observation",
        ACTOR_OBSERVATION_DIM,
        -1.0,
        1.0,
        "mixed_normalized",
        "actor_observation",
        "The complete honest actor observation.",
    ),
    VectorFieldSpec(
        "hidden_local_wind_xy_normalized",
        2,
        -1.0,
        1.0,
        "relative_air_speed_limit_fraction",
        "training_only_hidden_dynamics",
        "Exact local wind sample available only to an optional asymmetric critic.",
    ),
    VectorFieldSpec(
        "hidden_wake_delta_xy_normalized",
        2,
        -1.0,
        1.0,
        "relative_air_speed_limit_fraction",
        "training_only_hidden_dynamics",
        "Exact local deviation from the known inlet, never returned by reset or step.",
    ),
    VectorFieldSpec(
        "exact_clearance_normalized",
        1,
        0.0,
        1.0,
        "lidar_range_fraction",
        "training_only_geometry_metric",
        "Exact simulator clearance for critic/value learning only.",
    ),
    VectorFieldSpec(
        "episode_time_fraction",
        1,
        0.0,
        1.0,
        "unitless",
        "training_only_episode_state",
        "Fraction of the configured episode horizon consumed.",
    ),
)

PRIVILEGED_CRITIC_DIM = sum(field.size for field in PRIVILEGED_CRITIC_FIELDS)
ACTION_DIM = 2

REWARD_TERMS = (
    {
        "name": "progress",
        "weight": 2.0,
        "formula": "2.0 * (previous_goal_distance - goal_distance)",
    },
    {
        "name": "relative_air_speed_energy",
        "weight": -0.005,
        "formula": "-0.005 * ||ground_velocity - hidden_local_wind||^2 * dt",
    },
    {
        "name": "path_length",
        "weight": -0.015,
        "formula": "-0.015 * step_path_length",
    },
    {
        "name": "time",
        "weight": -0.02,
        "formula": "-0.02 per step",
    },
    {
        "name": "clearance",
        "weight": -0.08,
        "formula": "-0.08 * max(0, (5m-clearance)/5m)^2",
    },
    {
        "name": "smoothness",
        "weight": -0.08,
        "formula": "-0.08 * ||action - previous_action||^2",
    },
    {
        "name": "collision",
        "weight": -75.0,
        "formula": "-75 on swept rectangle or boundary collision",
    },
    {
        "name": "success",
        "weight": 100.0,
        "formula": "+100 on reaching the goal radius",
    },
)

EPISODE_METRICS = (
    {"name": "success", "units": "boolean", "description": "Goal reached before collision or timeout."},
    {"name": "collision_count", "units": "count", "description": "Swept geometry/boundary collisions."},
    {"name": "path_length_m", "units": "m", "description": "Integrated ground path length."},
    {
        "name": "relative_air_speed_energy",
        "units": "(m/s)^2*s",
        "description": "Integral of squared relative air speed; an energy proxy, not battery watt-hours.",
    },
    {"name": "time_s", "units": "s", "description": "Simulated elapsed time."},
    {"name": "min_clearance_m", "units": "m", "description": "Minimum obstacle/boundary clearance."},
    {"name": "score", "units": "reward", "description": "Sum of the versioned reward terms."},
)

ALLOWED_ACTOR_SOURCES = frozenset(field.source for field in ACTOR_OBSERVATION_FIELDS)
FORBIDDEN_ACTOR_CONCEPTS = (
    "hidden_full_flow_field",
    "exact_wake_state",
    "future_wind",
    "simulator_global_field_array",
    "privileged_critic_state",
)


def _bounds(fields: Iterable[VectorFieldSpec]) -> tuple[np.ndarray, np.ndarray]:
    low: list[float] = []
    high: list[float] = []
    for field in fields:
        low.extend([field.low] * field.size)
        high.extend([field.high] * field.size)
    return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)


ACTOR_OBSERVATION_LOW, ACTOR_OBSERVATION_HIGH = _bounds(ACTOR_OBSERVATION_FIELDS)
ACTOR_OBSERVATION_SPACE = BoxSpec(ACTOR_OBSERVATION_LOW, ACTOR_OBSERVATION_HIGH)
ACTION_SPACE = BoxSpec(
    np.full(ACTION_DIM, -1.0, dtype=np.float32),
    np.full(ACTION_DIM, 1.0, dtype=np.float32),
)


def _field_payload(fields: Iterable[VectorFieldSpec]) -> list[dict]:
    result: list[dict] = []
    offset = 0
    for field in fields:
        result.append(field.to_dict(offset))
        offset += field.size
    return result


def actor_observation_contract() -> dict:
    return {
        "schema_id": ACTOR_OBSERVATION_SCHEMA_ID,
        "version": CONTRACT_VERSION,
        "dimension": ACTOR_OBSERVATION_DIM,
        "dtype": "float32",
        "space": ACTOR_OBSERVATION_SPACE.to_dict(),
        "fields": _field_payload(ACTOR_OBSERVATION_FIELDS),
        "full_flow_access": False,
        "direct_hidden_provider_state_exposed": False,
        "onboard_relative_air_velocity_estimate_available": True,
        "forbidden": list(FORBIDDEN_ACTOR_CONCEPTS),
        "note": (
            "Relative air velocity is an onboard observable; the provider object, wake state, "
            "future samples, and spatial field remain hidden."
        ),
    }


def privileged_critic_contract() -> dict:
    low, high = _bounds(PRIVILEGED_CRITIC_FIELDS)
    return {
        "schema_id": PRIVILEGED_CRITIC_SCHEMA_ID,
        "version": CONTRACT_VERSION,
        "dimension": PRIVILEGED_CRITIC_DIM,
        "dtype": "float32",
        "space": BoxSpec(low, high).to_dict(),
        "fields": _field_payload(PRIVILEGED_CRITIC_FIELDS),
        "training_only": True,
        "returned_by_reset_or_step": False,
        "used_by_bundled_ppo_or_sac_entrypoint": False,
    }


def action_contract() -> dict:
    return {
        "schema_id": ACTION_SCHEMA_ID,
        "version": CONTRACT_VERSION,
        "dimension": ACTION_DIM,
        "dtype": "float32",
        "space": ACTION_SPACE.to_dict(),
        "frame": "vehicle_local_forward_left",
        "semantics": "bounded desired 2D ground-velocity guidance; the environment does not expose motor commands",
        "motor_control": False,
    }


def reward_contract() -> dict:
    return {
        "schema_id": REWARD_SCHEMA_ID,
        "version": CONTRACT_VERSION,
        "terms": list(REWARD_TERMS),
    }


def metrics_contract() -> dict:
    return {
        "schema_id": METRICS_SCHEMA_ID,
        "version": CONTRACT_VERSION,
        "metrics": list(EPISODE_METRICS),
    }


def leakage_guard_report() -> dict:
    """Fail closed if an actor field is added from an unapproved source."""

    approved_sources = {
        "own_odometry",
        "own_kinematics",
        "onboard_relative_air_velocity_estimate",
        "known_inlet",
        "goal_route_context",
        "deterministic_geometry_lidar",
        "recent_observable_history",
    }
    unapproved = sorted(ALLOWED_ACTOR_SOURCES - approved_sources)
    names = {field.name for field in ACTOR_OBSERVATION_FIELDS}
    forbidden_names = sorted(names.intersection(FORBIDDEN_ACTOR_CONCEPTS))
    passed = not unapproved and not forbidden_names and ACTOR_OBSERVATION_DIM == 33
    if not passed:
        raise RuntimeError(
            "actor observation leakage guard failed: "
            f"unapproved_sources={unapproved}, forbidden_fields={forbidden_names}, "
            f"dimension={ACTOR_OBSERVATION_DIM}"
        )
    return {
        "status": "passed",
        "schema_id": ACTOR_OBSERVATION_SCHEMA_ID,
        "dimension": ACTOR_OBSERVATION_DIM,
        "full_flow_access": False,
        "approved_sources": sorted(approved_sources),
    }
