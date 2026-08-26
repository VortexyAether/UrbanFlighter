from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import secrets
from threading import RLock
import time
from typing import Callable, Iterable

import numpy as np

from .baselines import BASELINE_ORDER, GuidanceBaseline, make_baseline
from .env import UrbanFlowConfig, UrbanFlowEnv
from .live_scenario import (
    LiveScenarioRecord,
    LiveScenarioRegistry,
    UnknownLiveScenarioError,
    live_scenario_registry,
    make_live_scenario,
)
from .observation import LIDAR_LOCAL_ANGLES_RAD, RADAR_LOCAL_ANGLES_RAD, RADAR_RANGE_M
from .schemas import (
    ACTION_SCHEMA_ID,
    ACTOR_OBSERVATION_FIELDS,
    ACTOR_OBSERVATION_SCHEMA_ID,
    REWARD_SCHEMA_ID,
)


INSPECTOR_FRAME_SCHEMA_ID = "urbanflow.episode_inspector_frame.v1"
INSPECTOR_WORLD_SCHEMA_ID = "urbanflow.episode_inspector_world.v1"
INSPECTOR_MAX_SESSIONS = 12
INSPECTOR_SESSION_TTL_S = 15 * 60
INSPECTOR_MAX_STEPS = 1_600
INSPECTOR_DEFAULT_MAX_STEPS = 1_200
INSPECTOR_MAX_BATCH_STEPS = 64


class UnknownInspectorSessionError(LookupError):
    pass


class StaleInspectorScenarioError(LookupError):
    pass


@dataclass
class InspectorSession:
    session_id: str
    scenario_id: str
    seed: int
    baseline_id: str
    max_steps: int
    record: LiveScenarioRecord
    env: UrbanFlowEnv
    baseline: GuidanceBaseline
    created_at: float
    last_accessed_at: float
    reset_count: int = 0


def _validated_action(action: Iterable[float] | np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(tuple(action), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("action must be a finite [forward, left] vector") from exc
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("action must be a finite [forward, left] vector")
    if np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("action components must be within [-1, 1]")
    return values


def estimate_inspector_minimum_required(
    distance_to_goal_m: float,
    config: UrbanFlowConfig,
    *,
    current_ground_speed_mps: float = 0.0,
) -> tuple[int, float]:
    """Return a conservative straight-line lower-bound estimate.

    The estimate assumes unobstructed travel at the configured maximum speed and
    adds the remaining zero-to-cruise acceleration time as a margin. It does not
    account for buildings, turns, wind, or terminal slowdown, so the inspector
    horizon must remain comfortably larger than this value.
    """

    distance = float(distance_to_goal_m)
    speed = float(current_ground_speed_mps)
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError("distance_to_goal_m must be non-negative and finite")
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("current_ground_speed_mps must be non-negative and finite")
    if distance <= 1e-12:
        return 0, 0.0
    cruise_speed = float(config.max_ground_speed_mps)
    acceleration_margin_s = max(
        0.0,
        (cruise_speed - min(speed, cruise_speed))
        / float(config.max_acceleration_mps2),
    )
    minimum_time_s = distance / cruise_speed + acceleration_margin_s
    minimum_steps = int(math.ceil(minimum_time_s / float(config.dt_s)))
    return minimum_steps, float(minimum_time_s)


def _world_payload(record: LiveScenarioRecord) -> dict:
    snapshot = record.snapshot()
    return {
        "schema_id": INSPECTOR_WORLD_SCHEMA_ID,
        "scenario_id": snapshot["scenario_id"],
        "content_hash_sha256": snapshot["content_hash_sha256"],
        "coordinate_frame": {
            "horizontal_units": "m",
            "x_axis": "east",
            "y_axis": "north",
            "display_orientation": "north_up",
        },
        "bounds": dict(snapshot["bounds"]),
        "start_goal_source": "deterministic_safe_route_in_registered_geometry",
        "structure_count": int(snapshot["structure_count"]),
        "buildings": [
            {
                "building_id": building["building_id"],
                "height_m": float(building["height_m"]),
                "height_source": building["height_source"],
                "footprint_xy_m": [point[:] for point in building["footprint_xy_m"]],
            }
            for building in snapshot["buildings"]
        ],
        "known_inlet": {
            "velocity_xy_mps": list(snapshot["inlet"]["velocity_xy_mps"]),
            "speed_mps": float(snapshot["inlet"]["speed_mps"]),
            "direction_from_north_deg": float(
                snapshot["inlet"]["direction_from_north_deg"]
            ),
            "timestamp": snapshot["inlet"]["timestamp"],
            "source": snapshot["inlet"]["source"],
            "fallback": snapshot["inlet"]["fallback"],
        },
        "source": "exact_registered_live_osm_scenario",
        "synthetic_fixture": False,
    }


def _observation_fields(observation: np.ndarray) -> list[dict]:
    values = np.asarray(observation, dtype=np.float32)
    offset = 0
    fields: list[dict] = []
    for specification in ACTOR_OBSERVATION_FIELDS:
        next_offset = offset + specification.size
        fields.append(
            {
                "name": specification.name,
                "values": [float(value) for value in values[offset:next_offset]],
                "units": specification.units,
                "source": specification.source,
            }
        )
        offset = next_offset
    return fields


def _lidar_payload(env: UrbanFlowEnv) -> dict:
    snapshot = env.actor_snapshot()
    rays = []
    for local_angle, distance in zip(
        LIDAR_LOCAL_ANGLES_RAD,
        snapshot.lidar_ranges_m,
        strict=True,
    ):
        world_angle = snapshot.heading_rad + float(local_angle)
        endpoint = snapshot.position_xy + float(distance) * np.array(
            [math.cos(world_angle), math.sin(world_angle)],
            dtype=float,
        )
        rays.append(
            {
                "local_angle_rad": float(local_angle),
                "distance_m": float(distance),
                "endpoint_xy_m": endpoint.tolist(),
                "hit": bool(float(distance) < env.config.lidar_range_m - 1e-6),
            }
        )
    return {
        "ray_count": len(rays),
        "max_range_m": float(env.config.lidar_range_m),
        "frame": "vehicle_local_counter_clockwise_from_forward",
        "rays": rays,
    }


def _radar_payload(env: UrbanFlowEnv) -> dict:
    snapshot = env.actor_snapshot()
    rays = []
    for local_angle, distance, range_rate in zip(
        RADAR_LOCAL_ANGLES_RAD,
        snapshot.radar_ranges_m,
        snapshot.radar_range_rate_mps,
        strict=True,
    ):
        world_angle = snapshot.heading_rad + float(local_angle)
        endpoint = snapshot.position_xy + float(distance) * np.array(
            [math.cos(world_angle), math.sin(world_angle)],
            dtype=float,
        )
        rays.append(
            {
                "local_angle_rad": float(local_angle),
                "distance_m": float(distance),
                "range_rate_mps": float(range_rate),
                "endpoint_xy_m": endpoint.tolist(),
                "hit": bool(float(distance) < RADAR_RANGE_M - 1e-6),
            }
        )
    return {
        "ray_count": len(rays),
        "max_range_m": float(RADAR_RANGE_M),
        "half_fov_deg": 60.0,
        "model_id": "sim-range-doppler-proxy-v1",
        "rf_hardware": False,
        "frame": "vehicle_local_forward_fan",
        "rays": rays,
    }


def _frame_payload(
    session: InspectorSession,
    *,
    observation: np.ndarray,
    action: np.ndarray,
    action_phase: str,
    action_source: str,
    reward_components: dict[str, float],
    step_reward: float,
    clearance_m: float,
    collision: bool,
    terminated: bool,
    truncated: bool,
) -> dict:
    env = session.env
    scenario = env.scenario
    actor = env.actor_snapshot()
    bounds = scenario.geometry.bounds_xy
    distance_to_goal_m = float(
        np.linalg.norm(scenario.goal_xy - actor.position_xy)
    )
    estimated_minimum_steps, estimated_minimum_time_s = (
        (0, 0.0)
        if distance_to_goal_m <= env.config.goal_radius_m
        else estimate_inspector_minimum_required(
            distance_to_goal_m,
            env.config,
            current_ground_speed_mps=float(
                np.linalg.norm(actor.ground_velocity_xy)
            ),
        )
    )
    if terminated:
        status = "success" if env.success else "collision"
    elif truncated:
        status = "time_limit"
    elif env.steps == 0:
        status = "ready"
    else:
        status = "running"
    return {
        "schema_id": INSPECTOR_FRAME_SCHEMA_ID,
        "scenario_id": session.scenario_id,
        "seed": session.seed,
        "baseline": {
            "baseline_id": session.baseline.baseline_id,
            "label": session.baseline.label,
            "uses_full_flow": False,
        },
        "world_bounds": {
            "min_x_m": float(bounds[0]),
            "max_x_m": float(bounds[1]),
            "min_y_m": float(bounds[2]),
            "max_y_m": float(bounds[3]),
        },
        "drone": {
            "position_xy_m": actor.position_xy.tolist(),
            "heading_rad": float(actor.heading_rad),
            "ground_velocity_xy_mps": actor.ground_velocity_xy.tolist(),
        },
        "start_xy_m": scenario.start_xy.tolist(),
        "goal_xy_m": scenario.goal_xy.tolist(),
        "trajectory_xy_m": [
            list(trajectory_frame["position_xy"])
            for trajectory_frame in env.trajectory
        ],
        "actor_lidar": _lidar_payload(env),
        "actor_radar": _radar_payload(env),
        "local_guidance_action": {
            "schema_id": ACTION_SCHEMA_ID,
            "frame": "vehicle_local_forward_left",
            "vector": action.tolist(),
            "forward": float(action[0]),
            "left": float(action[1]),
            "phase": action_phase,
            "source": action_source,
        },
        "actor_observation": {
            "schema_id": ACTOR_OBSERVATION_SCHEMA_ID,
            "vector": [float(value) for value in observation],
            "fields": _observation_fields(observation),
        },
        "air_relative_velocity_xy_mps": actor.relative_air_velocity_estimate_xy.tolist(),
        "reward": {
            "schema_id": REWARD_SCHEMA_ID,
            "components": {
                name: float(value) for name, value in reward_components.items()
            },
            "step_total": float(step_reward),
            "episode_total": float(env.score),
        },
        "clearance_m": float(clearance_m),
        "collision": bool(collision),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "status": status,
        "termination_reason": env.termination_reason,
        "step_index": int(env.steps),
        "max_steps": int(session.max_steps),
        "dt_s": float(env.config.dt_s),
        "simulated_elapsed_s": float(env.time_s),
        "simulated_max_s": float(session.max_steps * env.config.dt_s),
        "distance_to_goal_m": distance_to_goal_m,
        "estimated_minimum_steps": estimated_minimum_steps,
        "estimated_minimum_time_s": estimated_minimum_time_s,
        "flags": {
            "policy_status": "not_trained",
            "policy_had_privileged_flow_access": False,
            "full_flow_access": False,
            "training_executed": False,
            "browser_motor_training": False,
            "navier_stokes_cfd": False,
            "real_cfd_validation_run": False,
            "real_cfd_adapter_status": "interface_only_not_executed",
            "synthetic_fixture": False,
        },
    }


class InspectorSessionManager:
    """Bounded, process-local deterministic episode sessions.

    The registered flow field stays inside ``LiveScenarioRecord`` and the
    environment. Only ``_world_payload`` and ``_frame_payload`` cross the API
    boundary, and neither serializes the field arrays or obstacle mask.
    """

    def __init__(
        self,
        *,
        registry: LiveScenarioRegistry = live_scenario_registry,
        max_sessions: int = INSPECTOR_MAX_SESSIONS,
        ttl_s: float = INSPECTOR_SESSION_TTL_S,
        clock: Callable[[], float] = time.monotonic,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int) or max_sessions < 1:
            raise ValueError("max_sessions must be a positive integer")
        if not math.isfinite(float(ttl_s)) or float(ttl_s) <= 0.0:
            raise ValueError("ttl_s must be positive and finite")
        self.registry = registry
        self.max_sessions = max_sessions
        self.ttl_s = float(ttl_s)
        self.clock = clock
        self.id_factory = id_factory or (lambda: f"ufi_{secrets.token_urlsafe(18)}")
        self._sessions: OrderedDict[str, InspectorSession] = OrderedDict()
        self._lock = RLock()

    def create(
        self,
        *,
        scenario_id: str,
        seed: int,
        baseline_id: str,
        max_steps: int = INSPECTOR_DEFAULT_MAX_STEPS,
    ) -> dict:
        seed = self._validate_seed(seed)
        max_steps = self._validate_max_steps(max_steps)
        if baseline_id not in BASELINE_ORDER:
            raise ValueError(
                f"unknown baseline '{baseline_id}'; expected one of {', '.join(BASELINE_ORDER)}"
            )
        record = self.registry.get_record(scenario_id)
        env, baseline, observation, info = self._new_episode(
            record=record,
            seed=seed,
            baseline_id=baseline_id,
            max_steps=max_steps,
        )
        now = self.clock()
        session_id = self.id_factory()
        session = InspectorSession(
            session_id=session_id,
            scenario_id=record.scenario_id,
            seed=seed,
            baseline_id=baseline_id,
            max_steps=max_steps,
            record=record,
            env=env,
            baseline=baseline,
            created_at=now,
            last_accessed_at=now,
        )
        action = _validated_action(baseline.action(env.actor_snapshot()))
        frame = _frame_payload(
            session,
            observation=observation,
            action=action,
            action_phase="preview_next",
            action_source="deterministic_baseline",
            reward_components=info["reward_terms"],
            step_reward=0.0,
            clearance_m=info["clearance_m"],
            collision=False,
            terminated=False,
            truncated=False,
        )
        with self._lock:
            self._purge_expired_locked(now)
            if session_id in self._sessions:
                raise RuntimeError("inspector session id collision")
            self._sessions[session_id] = session
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        return self._response(session, frame, include_world=True, session_active=True)

    def reset(self, session_id: str) -> dict:
        with self._lock:
            session = self._get_locked(session_id)
            self._require_live_scenario_locked(session)
            env, baseline, observation, info = self._new_episode(
                record=session.record,
                seed=session.seed,
                baseline_id=session.baseline_id,
                max_steps=session.max_steps,
            )
            session.env = env
            session.baseline = baseline
            session.reset_count += 1
            session.last_accessed_at = self.clock()
            self._sessions.move_to_end(session_id)
            action = _validated_action(baseline.action(env.actor_snapshot()))
            frame = _frame_payload(
                session,
                observation=observation,
                action=action,
                action_phase="preview_next",
                action_source="deterministic_baseline",
                reward_components=info["reward_terms"],
                step_reward=0.0,
                clearance_m=info["clearance_m"],
                collision=False,
                terminated=False,
                truncated=False,
            )
            return self._response(session, frame, include_world=True, session_active=True)

    def step(
        self,
        session_id: str,
        action: Iterable[float] | None = None,
        *,
        repeat: int = 1,
    ) -> dict:
        repeat = self._validate_repeat(repeat)
        override = None if action is None else _validated_action(action)
        with self._lock:
            session = self._get_locked(session_id)
            self._require_live_scenario_locked(session)
            executed_steps = 0
            batch_reward = 0.0
            observation: np.ndarray | None = None
            command: np.ndarray | None = None
            reward = 0.0
            terminated = False
            truncated = False
            info: dict | None = None
            action_source = (
                "deterministic_baseline"
                if override is None
                else "validated_actor_override"
            )
            for _ in range(repeat):
                # Stateful waypoint guidance must observe the result of every
                # preceding step, exactly as it does for serial API requests.
                command = (
                    _validated_action(
                        session.baseline.action(session.env.actor_snapshot())
                    )
                    if override is None
                    else override
                )
                observation, reward, terminated, truncated, info = session.env.step(
                    command
                )
                executed_steps += 1
                batch_reward += reward
                if terminated or truncated:
                    break
            if observation is None or command is None or info is None:
                raise RuntimeError("inspector batch executed no environment steps")
            session.last_accessed_at = self.clock()
            self._sessions.move_to_end(session_id)
            frame = _frame_payload(
                session,
                observation=observation,
                action=command,
                action_phase="executed",
                action_source=action_source,
                reward_components=info["reward_terms"],
                step_reward=reward,
                clearance_m=info["clearance_m"],
                collision=bool(info["reward_terms"].get("collision", 0.0) < 0.0),
                terminated=terminated,
                truncated=truncated,
            )
            terminal = bool(terminated or truncated)
            response = self._response(
                session,
                frame,
                include_world=False,
                session_active=not terminal,
                requested_steps=repeat,
                executed_steps=executed_steps,
                batch_reward=batch_reward,
            )
            if terminal:
                self._sessions.pop(session_id, None)
                response["cleanup"] = "terminal_session_deleted"
            return response

    def delete(self, session_id: str) -> dict:
        with self._lock:
            self._purge_expired_locked(self.clock())
            if self._sessions.pop(session_id, None) is None:
                raise UnknownInspectorSessionError(
                    f"inspector session '{session_id}' is unknown, expired, or already closed"
                )
        return {"status": "deleted", "session_id": session_id}

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def __len__(self) -> int:
        with self._lock:
            self._purge_expired_locked(self.clock())
            return len(self._sessions)

    @staticmethod
    def _validate_seed(seed: int) -> int:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise ValueError("seed must be an integer")
        value = int(seed)
        if value < 0 or value > 2_147_483_647:
            raise ValueError("seed must be within [0, 2147483647]")
        return value

    @staticmethod
    def _validate_max_steps(max_steps: int) -> int:
        if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)):
            raise ValueError("max_steps must be an integer")
        value = int(max_steps)
        if value < 1 or value > INSPECTOR_MAX_STEPS:
            raise ValueError(f"max_steps must be within [1, {INSPECTOR_MAX_STEPS}]")
        return value

    @staticmethod
    def _validate_repeat(repeat: int) -> int:
        if isinstance(repeat, bool) or not isinstance(repeat, (int, np.integer)):
            raise ValueError("repeat must be an integer")
        value = int(repeat)
        if value < 1 or value > INSPECTOR_MAX_BATCH_STEPS:
            raise ValueError(
                f"repeat must be within [1, {INSPECTOR_MAX_BATCH_STEPS}]"
            )
        return value

    @staticmethod
    def _new_episode(
        *,
        record: LiveScenarioRecord,
        seed: int,
        baseline_id: str,
        max_steps: int,
    ) -> tuple[UrbanFlowEnv, GuidanceBaseline, np.ndarray, dict]:
        scenario = make_live_scenario(record, seed=seed)
        config = UrbanFlowConfig(max_steps=max_steps)
        env = UrbanFlowEnv(config, fixed_scenario=scenario)
        observation, info = env.reset(seed=seed)
        baseline = make_baseline(baseline_id, scenario.public_context(), config)
        return env, baseline, observation, info

    def _get_locked(self, session_id: str) -> InspectorSession:
        now = self.clock()
        self._purge_expired_locked(now)
        try:
            session = self._sessions[session_id]
        except KeyError as exc:
            raise UnknownInspectorSessionError(
                f"inspector session '{session_id}' is unknown, expired, or already closed"
            ) from exc
        session.last_accessed_at = now
        self._sessions.move_to_end(session_id)
        return session

    def _require_live_scenario_locked(self, session: InspectorSession) -> None:
        try:
            record = self.registry.get_record(session.scenario_id)
        except UnknownLiveScenarioError as exc:
            self._sessions.pop(session.session_id, None)
            raise StaleInspectorScenarioError(
                f"live scenario '{session.scenario_id}' is stale or no longer cached; session deleted"
            ) from exc
        if record.canonical_snapshot_bytes != session.record.canonical_snapshot_bytes:
            self._sessions.pop(session.session_id, None)
            raise StaleInspectorScenarioError(
                f"live scenario '{session.scenario_id}' changed unexpectedly; session deleted"
            )

    def _purge_expired_locked(self, now: float) -> None:
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_accessed_at >= self.ttl_s
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def _response(
        self,
        session: InspectorSession,
        frame: dict,
        *,
        include_world: bool,
        session_active: bool,
        requested_steps: int = 0,
        executed_steps: int = 0,
        batch_reward: float = 0.0,
    ) -> dict:
        response = {
            "session_id": session.session_id,
            "session_active": session_active,
            "scenario_id": session.scenario_id,
            "seed": session.seed,
            "baseline_id": session.baseline_id,
            "requested_steps": int(requested_steps),
            "executed_steps": int(executed_steps),
            "batch_reward": float(batch_reward),
            "limits": {
                "max_steps": session.max_steps,
                "dt_s": float(session.env.config.dt_s),
                "simulated_max_s": float(
                    session.max_steps * session.env.config.dt_s
                ),
                "max_batch_steps": INSPECTOR_MAX_BATCH_STEPS,
                "max_sessions": self.max_sessions,
                "ttl_s": self.ttl_s,
                "reset_count": session.reset_count,
            },
            "frame": frame,
        }
        if include_world:
            response["world"] = _world_payload(session.record)
        return response


inspector_session_manager = InspectorSessionManager()
