from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .observation import (
    ActorObservableState,
    ActorSnapshot,
    build_actor_observation,
    build_privileged_critic_state,
)
from .scenario import UrbanFlowScenario, make_seeded_scenario
from .schemas import (
    ACTION_SCHEMA_ID,
    ACTION_SPACE,
    ACTOR_OBSERVATION_SCHEMA_ID,
    ACTOR_OBSERVATION_SPACE,
    ENVIRONMENT_ID,
    METRICS_SCHEMA_ID,
    PRIVILEGED_CRITIC_SCHEMA_ID,
    REWARD_SCHEMA_ID,
    leakage_guard_report,
)


@dataclass(frozen=True)
class UrbanFlowConfig:
    dt_s: float = 0.25
    max_steps: int = 360
    max_ground_speed_mps: float = 9.0
    max_acceleration_mps2: float = 7.0
    velocity_tracking_time_s: float = 0.65
    wind_drag_gain_per_s: float = 0.14
    max_turn_rate_rad_s: float = 2.8
    agent_radius_m: float = 1.25
    goal_radius_m: float = 2.5
    lidar_range_m: float = 35.0
    relative_air_speed_limit_mps: float = 20.0
    clearance_reward_threshold_m: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.max_steps, bool) or not isinstance(
            self.max_steps, (int, np.integer)
        ):
            raise ValueError("max_steps must be an integer")
        positive = {
            "dt_s": self.dt_s,
            "max_steps": self.max_steps,
            "max_ground_speed_mps": self.max_ground_speed_mps,
            "max_acceleration_mps2": self.max_acceleration_mps2,
            "velocity_tracking_time_s": self.velocity_tracking_time_s,
            "max_turn_rate_rad_s": self.max_turn_rate_rad_s,
            "agent_radius_m": self.agent_radius_m,
            "goal_radius_m": self.goal_radius_m,
            "lidar_range_m": self.lidar_range_m,
            "relative_air_speed_limit_mps": self.relative_air_speed_limit_mps,
            "clearance_reward_threshold_m": self.clearance_reward_threshold_m,
        }
        for name, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if (
            not math.isfinite(float(self.wind_drag_gain_per_s))
            or float(self.wind_drag_gain_per_s) < 0.0
        ):
            raise ValueError("wind_drag_gain_per_s must be non-negative and finite")


def relative_air_energy_step(
    ground_velocity_xy: np.ndarray,
    local_wind_xy: np.ndarray,
    dt_s: float,
) -> float:
    ground_velocity = np.asarray(ground_velocity_xy, dtype=float)
    local_wind = np.asarray(local_wind_xy, dtype=float)
    duration = float(dt_s)
    if (
        ground_velocity.shape != (2,)
        or local_wind.shape != (2,)
        or not np.all(np.isfinite(ground_velocity))
        or not np.all(np.isfinite(local_wind))
    ):
        raise ValueError("ground velocity and local wind must be finite 2D vectors")
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("dt_s must be non-negative and finite")
    relative_air_velocity = ground_velocity - local_wind
    return float(np.dot(relative_air_velocity, relative_air_velocity) * duration)


class UrbanFlowEnv:
    """Deterministic, headless, NumPy-only UrbanFlow Gym core.

    ``reset`` and ``step`` use the Gymnasium return convention without requiring
    Gymnasium itself. The returned actor observation never contains exact local
    wind, wake state, a future sample, or a full field array.
    """

    metadata = {
        "name": ENVIRONMENT_ID,
        "render_modes": [],
        "headless": True,
        "policy_status": "not_trained",
    }

    def __init__(
        self,
        config: UrbanFlowConfig | None = None,
        *,
        scenario_factory: Callable[[int, bool], UrbanFlowScenario] = make_seeded_scenario,
        fixed_scenario: UrbanFlowScenario | None = None,
    ) -> None:
        self.config = config or UrbanFlowConfig()
        self.scenario_factory = scenario_factory
        self.fixed_scenario = fixed_scenario
        self.observation_space = ACTOR_OBSERVATION_SPACE
        self.action_space = ACTION_SPACE
        leakage_guard_report()
        self.scenario: UrbanFlowScenario
        self.seed_value = 0
        self.np_random = np.random.default_rng(0)
        self._last_actor_snapshot: ActorSnapshot | None = None
        self._done = False
        self.reset(seed=fixed_scenario.seed if fixed_scenario is not None else 0)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        options = dict(options or {})
        if seed is None:
            seed = self.seed_value
        self.seed_value = int(seed)
        self.np_random = np.random.default_rng(self.seed_value)
        supplied_scenario = options.pop("scenario", None)
        randomize_domain = bool(options.pop("randomize_domain", True))
        if options:
            unknown = ", ".join(sorted(options))
            raise ValueError(f"unknown reset options: {unknown}")
        if supplied_scenario is not None and not isinstance(supplied_scenario, UrbanFlowScenario):
            raise TypeError("reset option 'scenario' must be an UrbanFlowScenario")
        if supplied_scenario is not None:
            self.scenario = supplied_scenario
            self.seed_value = supplied_scenario.seed
            self.np_random = np.random.default_rng(self.seed_value)
        elif self.fixed_scenario is not None:
            self.scenario = self.fixed_scenario
            self.seed_value = self.fixed_scenario.seed
            self.np_random = np.random.default_rng(self.seed_value)
        else:
            self.scenario = self.scenario_factory(self.seed_value, randomize_domain)

        self.position_xy = self.scenario.start_xy.copy()
        self.ground_velocity_xy = np.zeros(2, dtype=float)
        self.heading_rad = float(self.scenario.initial_heading_rad)
        self.previous_action = np.zeros(2, dtype=float)
        self.steps = 0
        self.time_s = 0.0
        self.path_length_m = 0.0
        self.relative_air_speed_energy = 0.0
        self.collision_count = 0
        self.score = 0.0
        self.success = False
        self.termination_reason: str | None = None
        self._done = False
        self.reward_terms_total = {
            "progress": 0.0,
            "relative_air_speed_energy": 0.0,
            "path_length": 0.0,
            "time": 0.0,
            "clearance": 0.0,
            "smoothness": 0.0,
            "collision": 0.0,
            "success": 0.0,
        }
        initial_clearance = self.scenario.geometry.clearance(
            self.position_xy, self.config.agent_radius_m
        )
        self.min_clearance_m = float(initial_clearance)
        observation, snapshot = self._actor_observation()
        self._last_actor_snapshot = snapshot
        self.trajectory = [
            self._trajectory_frame(
                observation=observation,
                action=None,
                reward=0.0,
                terminated=False,
                truncated=False,
                clearance_m=initial_clearance,
            )
        ]
        return observation, self._info(
            reward_terms={name: 0.0 for name in self.reward_terms_total},
            clearance_m=initial_clearance,
            terminated=False,
            truncated=False,
        )

    def step(self, action: np.ndarray | list[float]) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._done:
            raise RuntimeError("step called after episode end; call reset before stepping again")
        command = np.asarray(action, dtype=float)
        if command.shape != (2,) or not np.all(np.isfinite(command)):
            raise ValueError("action must be a finite vector with shape (2,)")
        command = np.clip(command, -1.0, 1.0)
        command_norm = float(np.linalg.norm(command))
        if command_norm > 1.0:
            command = command / command_norm

        old_position = self.position_xy.copy()
        old_goal_distance = float(np.linalg.norm(self.scenario.goal_xy - old_position))
        old_action = self.previous_action.copy()
        forward = np.array([math.cos(self.heading_rad), math.sin(self.heading_rad)], dtype=float)
        left = np.array([-forward[1], forward[0]], dtype=float)
        desired_ground_velocity = self.config.max_ground_speed_mps * (
            command[0] * forward + command[1] * left
        )
        local_wind_before = self.scenario.wind_provider.velocity_at(old_position, self.time_s)
        tracking_acceleration = (
            desired_ground_velocity - self.ground_velocity_xy
        ) / self.config.velocity_tracking_time_s
        wind_drag_acceleration = -self.config.wind_drag_gain_per_s * (
            self.ground_velocity_xy - local_wind_before
        )
        acceleration = tracking_acceleration + wind_drag_acceleration
        acceleration_norm = float(np.linalg.norm(acceleration))
        if acceleration_norm > self.config.max_acceleration_mps2:
            acceleration *= self.config.max_acceleration_mps2 / acceleration_norm
        next_velocity = self.ground_velocity_xy + acceleration * self.config.dt_s
        velocity_limit = self.config.max_ground_speed_mps * 1.35
        next_speed = float(np.linalg.norm(next_velocity))
        if next_speed > velocity_limit:
            next_velocity *= velocity_limit / next_speed
        candidate_position = old_position + next_velocity * self.config.dt_s

        target_heading_vector = desired_ground_velocity
        if float(np.linalg.norm(target_heading_vector)) > 0.1:
            target_heading = math.atan2(target_heading_vector[1], target_heading_vector[0])
            heading_delta = _wrap_angle(target_heading - self.heading_rad)
            max_heading_delta = self.config.max_turn_rate_rad_s * self.config.dt_s
            self.heading_rad = _wrap_angle(
                self.heading_rad + float(np.clip(heading_delta, -max_heading_delta, max_heading_delta))
            )

        collision = self.scenario.geometry.segment_collides(
            old_position,
            candidate_position,
            self.config.agent_radius_m,
        )
        if collision:
            self.position_xy = old_position
            self.ground_velocity_xy = np.zeros(2, dtype=float)
            self.collision_count += 1
        else:
            self.position_xy = candidate_position
            self.ground_velocity_xy = next_velocity

        self.steps += 1
        self.time_s = self.steps * self.config.dt_s
        segment_length = float(np.linalg.norm(self.position_xy - old_position))
        self.path_length_m += segment_length
        local_wind_after = self.scenario.wind_provider.velocity_at(self.position_xy, self.time_s)
        energy_step = relative_air_energy_step(
            self.ground_velocity_xy,
            local_wind_after,
            self.config.dt_s,
        )
        self.relative_air_speed_energy += energy_step
        clearance = self.scenario.geometry.clearance(
            self.position_xy,
            self.config.agent_radius_m,
        )
        if collision:
            clearance = min(clearance, 0.0)
        self.min_clearance_m = min(self.min_clearance_m, float(clearance))

        goal_distance = float(np.linalg.norm(self.scenario.goal_xy - self.position_xy))
        reached_goal = bool(goal_distance <= self.config.goal_radius_m and not collision)
        terminated = bool(collision or reached_goal)
        truncated = bool(self.steps >= self.config.max_steps and not terminated)
        if collision:
            self.termination_reason = "collision"
        elif reached_goal:
            self.termination_reason = "success"
            self.success = True
        elif truncated:
            self.termination_reason = "time_limit"

        normalized_clearance_deficit = max(
            0.0,
            (self.config.clearance_reward_threshold_m - max(float(clearance), 0.0))
            / self.config.clearance_reward_threshold_m,
        )
        reward_terms = {
            "progress": 2.0 * (old_goal_distance - goal_distance),
            "relative_air_speed_energy": -0.005 * energy_step,
            "path_length": -0.015 * segment_length,
            "time": -0.02,
            "clearance": -0.08 * normalized_clearance_deficit**2,
            "smoothness": -0.08 * float(np.dot(command - old_action, command - old_action)),
            "collision": -75.0 * float(collision),
            "success": 100.0 * float(reached_goal),
        }
        reward = float(sum(reward_terms.values()))
        self.score += reward
        for name, value in reward_terms.items():
            self.reward_terms_total[name] += float(value)
        self.previous_action = command.copy()
        self._done = bool(terminated or truncated)

        observation, snapshot = self._actor_observation(local_wind_xy=local_wind_after)
        self._last_actor_snapshot = snapshot
        info = self._info(
            reward_terms=reward_terms,
            clearance_m=clearance,
            terminated=terminated,
            truncated=truncated,
        )
        self.trajectory.append(
            self._trajectory_frame(
                observation=observation,
                action=command,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                clearance_m=clearance,
            )
        )
        return observation, reward, terminated, truncated, info

    def actor_snapshot(self) -> ActorSnapshot:
        if self._last_actor_snapshot is None:
            raise RuntimeError("environment has not been reset")
        snapshot = self._last_actor_snapshot
        return ActorSnapshot(
            position_xy=snapshot.position_xy.copy(),
            ground_velocity_xy=snapshot.ground_velocity_xy.copy(),
            heading_rad=snapshot.heading_rad,
            relative_air_velocity_estimate_xy=snapshot.relative_air_velocity_estimate_xy.copy(),
            known_inlet_velocity_xy=snapshot.known_inlet_velocity_xy.copy(),
            goal_xy=snapshot.goal_xy.copy(),
            lidar_ranges_m=snapshot.lidar_ranges_m.copy(),
            previous_action=snapshot.previous_action.copy(),
        )

    def privileged_critic_state(self) -> np.ndarray:
        """Explicit training-only channel; never included in reset/step output."""

        actor_observation, _ = self._actor_observation()
        exact_local_wind = self.scenario.wind_provider.velocity_at(self.position_xy, self.time_s)
        clearance = self.scenario.geometry.clearance(
            self.position_xy, self.config.agent_radius_m
        )
        return build_privileged_critic_state(
            actor_observation,
            exact_local_wind_xy=exact_local_wind,
            known_inlet_velocity_xy=self.scenario.known_inlet_velocity_xy,
            exact_clearance_m=clearance,
            lidar_range_m=self.config.lidar_range_m,
            elapsed_steps=self.steps,
            max_steps=self.config.max_steps,
            relative_air_speed_limit_mps=self.config.relative_air_speed_limit_mps,
        )

    def metrics(self) -> dict:
        return {
            "schema_id": METRICS_SCHEMA_ID,
            "success": bool(self.success),
            "collision_count": int(self.collision_count),
            "path_length_m": float(self.path_length_m),
            "relative_air_speed_energy": float(self.relative_air_speed_energy),
            "time_s": float(self.time_s),
            "min_clearance_m": float(self.min_clearance_m),
            "score": float(self.score),
            "steps": int(self.steps),
            "final_goal_distance_m": float(
                np.linalg.norm(self.scenario.goal_xy - self.position_xy)
            ),
            "termination_reason": self.termination_reason,
            "reward_terms_total": self.reward_terms_total.copy(),
            "actor_full_flow_access": False,
        }

    def _actor_observation(
        self,
        local_wind_xy: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ActorSnapshot]:
        if local_wind_xy is None:
            local_wind_xy = self.scenario.wind_provider.velocity_at(
                self.position_xy, self.time_s
            )
        relative_air_estimate = self.ground_velocity_xy - np.asarray(local_wind_xy, dtype=float)
        observable = ActorObservableState(
            position_xy=self.position_xy,
            ground_velocity_xy=self.ground_velocity_xy,
            heading_rad=self.heading_rad,
            relative_air_velocity_estimate_xy=relative_air_estimate,
            known_inlet_velocity_xy=self.scenario.known_inlet_velocity_xy,
            goal_xy=self.scenario.goal_xy,
            previous_action=self.previous_action,
        )
        return build_actor_observation(
            observable,
            self.scenario.geometry,
            max_ground_speed_mps=self.config.max_ground_speed_mps,
            relative_air_speed_limit_mps=self.config.relative_air_speed_limit_mps,
            lidar_range_m=self.config.lidar_range_m,
        )

    def _info(
        self,
        *,
        reward_terms: dict[str, float],
        clearance_m: float,
        terminated: bool,
        truncated: bool,
    ) -> dict:
        return {
            "scenario_id": self.scenario.scenario_id,
            "seed": self.seed_value,
            "actor_observation_schema_id": ACTOR_OBSERVATION_SCHEMA_ID,
            "action_schema_id": ACTION_SCHEMA_ID,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "policy_had_privileged_flow_access": False,
            "full_flow_access": False,
            "relative_air_velocity_estimate_mps": (
                self._last_actor_snapshot.relative_air_velocity_estimate_xy.tolist()
                if self._last_actor_snapshot is not None
                else [0.0, 0.0]
            ),
            "clearance_m": float(clearance_m),
            "reward_terms": {name: float(value) for name, value in reward_terms.items()},
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reason": self.termination_reason,
            "privileged_critic_schema_id": PRIVILEGED_CRITIC_SCHEMA_ID,
            "privileged_critic_returned_to_actor": False,
            "hidden_dynamics_source": self.scenario.wind_provider.source_metadata()["kind"],
        }

    def _trajectory_frame(
        self,
        *,
        observation: np.ndarray,
        action: np.ndarray | None,
        reward: float,
        terminated: bool,
        truncated: bool,
        clearance_m: float,
    ) -> dict:
        snapshot = self._last_actor_snapshot
        relative_air = (
            snapshot.relative_air_velocity_estimate_xy.tolist()
            if snapshot is not None
            else [0.0, 0.0]
        )
        return {
            "step": int(self.steps),
            "time_s": float(self.time_s),
            "position_xy": self.position_xy.tolist(),
            "ground_velocity_xy": self.ground_velocity_xy.tolist(),
            "relative_air_velocity_estimate_xy": relative_air,
            "heading_rad": float(self.heading_rad),
            "action": None if action is None else np.asarray(action, dtype=float).tolist(),
            "reward": float(reward),
            "clearance_m": float(clearance_m),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "actor_observation_schema_id": ACTOR_OBSERVATION_SCHEMA_ID,
            "actor_observation_checksum": float(np.sum(observation, dtype=np.float64)),
            "policy_had_privileged_flow_access": False,
        }


def _wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)
