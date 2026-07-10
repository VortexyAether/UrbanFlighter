from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .observation import (
    OBSERVATION_FIELD_NAMES,
    build_policy_observation,
    gym_spaces,
    make_box_space,
    observation_bounds,
    policy_observation_contract,
)
from .mission import default_goal, default_start, random_mission
from .world import UrbanWorld
from .wind import UrbanWindField


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict

    def __iter__(self) -> Iterator:
        yield self.obs
        yield self.reward
        yield self.terminated
        yield self.truncated
        yield self.info


class UrbanFlighterEnv:
    """Gymnasium-style 3D drone path-optimization environment.

    Policy observations are intentionally limited to mission-observable state:
    relative goal, own velocity, inlet/base wind, and OSM/building-derived range
    features. The spatial wind field is hidden simulator dynamics.
    """

    metadata = {
        "name": "UrbanFlighterEnv-v0",
        "render_modes": ["trajectory_json"],
        "policy_status": "deterministic_baseline_only_not_trained_rl",
    }

    def __init__(self, world=None, wind=None, dt=0.25, max_steps=600, start=None, goal=None):
        self.world = world or UrbanWorld.toy_city()
        self.wind = wind or UrbanWindField(self.world)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.max_acc = 5.0
        self.max_speed = 14.0
        self.goal_radius = 4.0
        self.start = np.array(start if start is not None else default_start(self.world), dtype=float)
        self.goal = np.array(goal if goal is not None else default_goal(self.world), dtype=float)
        obs_low, obs_high = observation_bounds()
        self.observation_space = make_box_space(obs_low, obs_high)
        self.action_space = make_box_space(np.full(3, -1.0), np.full(3, 1.0))
        self.neighbor_positions: list[np.ndarray] = []
        self.min_pairwise_separation_m = float("inf")
        self.separation_violations = 0
        self.reset()

    def reset(self, seed=None, options=None):
        self.seed = int(seed) if seed is not None else None
        self.np_random = np.random.default_rng(self.seed)
        options = options or {}
        mission = options.get("mission")
        if mission is not None:
            self.start = np.asarray(mission["start"], dtype=float)
            self.goal = np.asarray(mission["goal"], dtype=float)
        elif bool(options.get("randomize_mission", False)):
            self.start, self.goal = random_mission(self.world, self.np_random)
        self.t = 0.0
        self.steps = 0
        self.pos = self.start.copy()
        self.vel = np.zeros(3, dtype=float)
        self.energy = 0.0
        self.path_length = 0.0
        self.collisions = 0
        self.boundary_violations = 0
        self.reward_total = 0.0
        self.reward_terms_total = {
            "progress": 0.0,
            "path_cost": 0.0,
            "time_cost": 0.0,
            "energy_cost": 0.0,
            "collision_cost": 0.0,
            "boundary_cost": 0.0,
            "separation_cost": 0.0,
            "goal_bonus": 0.0,
        }
        self.min_building_clearance_m = self.world.nearest_building_clearance(self.pos)
        self.min_pairwise_separation_m = float("inf")
        self.separation_violations = 0
        self.trajectory = [self.pos.copy()]
        return self._obs(), self._info(collision=False, energy_step=0.0, reward_terms=self.reward_terms_total.copy())

    def _obs(self):
        return build_policy_observation(self.world, self.wind.base_wind, self.pos, self.goal, self.vel, self.max_speed)

    def set_neighbor_positions(self, positions: list[np.ndarray]) -> None:
        self.neighbor_positions = [np.asarray(position, dtype=float) for position in positions]

    def _info(self, collision: bool, energy_step: float, reward_terms: dict) -> dict:
        w_local = self.wind.at(self.pos, self.t)
        v_air = self.vel - w_local
        return {
            "distance_to_goal": float(np.linalg.norm(self.goal - self.pos)),
            "energy_step": float(energy_step),
            "energy_total": float(self.energy),
            "collision": bool(collision),
            "collisions_total": int(self.collisions),
            "building_clearance_m": float(self.world.nearest_building_clearance(self.pos)),
            "min_building_clearance_m": float(self.min_building_clearance_m),
            "reward_terms": reward_terms,
            "reward_terms_total": self.reward_terms_total.copy(),
            "seed": self.seed,
            "hidden_dynamics_wind_mps": w_local.tolist(),
            "relative_airspeed_mps": v_air.tolist(),
            "policy_observation_fields": OBSERVATION_FIELD_NAMES,
            "policy_had_privileged_flow_access": False,
            "min_pairwise_separation_m": float(self.min_pairwise_separation_m),
            "separation_violations": int(self.separation_violations),
        }

    def step(self, action):
        action = np.asarray(action, dtype=float)
        norm = np.linalg.norm(action)
        if norm > 1.0:
            action = action / norm
        acc_cmd = action * self.max_acc

        old_pos = self.pos.copy()
        self.vel = self.vel + acc_cmd * self.dt
        speed = np.linalg.norm(self.vel)
        if speed > self.max_speed:
            self.vel = self.vel / speed * self.max_speed

        candidate_pos = self.pos + self.vel * self.dt
        self.t += self.dt
        self.steps += 1

        boundary = not self.world.in_bounds(candidate_pos)
        building_collision = self.world.segment_hits_building(old_pos, candidate_pos, margin=0.75)
        blocked = boundary or building_collision
        if blocked:
            if building_collision:
                self.collisions += 1
            if boundary:
                self.boundary_violations += 1
            # Bounce at the last safe pose instead of accepting a tunneled-through frame.
            self.pos = old_pos
            self.vel *= -0.25
        else:
            self.pos = candidate_pos

        segment = np.linalg.norm(self.pos - old_pos)
        self.path_length += segment
        self.trajectory.append(self.pos.copy())

        w_local = self.wind.at(self.pos, self.t)
        v_air = self.vel - w_local
        energy_step = float(np.linalg.norm(v_air) ** 2 * self.dt)
        self.energy += energy_step

        dist = float(np.linalg.norm(self.goal - self.pos))
        old_dist = float(np.linalg.norm(self.goal - old_pos))
        progress = old_dist - dist
        separation_cost = 0.0
        for other_position in self.neighbor_positions:
            sep = float(np.linalg.norm(self.pos - other_position))
            self.min_pairwise_separation_m = min(self.min_pairwise_separation_m, sep)
            if sep < 10.0:
                self.separation_violations += 1
                separation_cost += -1.5 * (10.0 - sep) / 10.0
        terminated = dist < self.goal_radius
        truncated = self.steps >= self.max_steps
        reward_terms = {
            "progress": 1.8 * progress,
            "path_cost": -0.01 * segment,
            "time_cost": -0.02,
            "energy_cost": -0.015 * energy_step,
            "collision_cost": -2.5 * float(building_collision),
            "boundary_cost": -4.0 * float(boundary),
            "separation_cost": separation_cost,
            "goal_bonus": 100.0 * float(terminated),
        }
        reward = float(sum(reward_terms.values()))
        self.reward_total += reward
        for key, value in reward_terms.items():
            self.reward_terms_total[key] += float(value)
        self.min_building_clearance_m = min(self.min_building_clearance_m, self.world.nearest_building_clearance(self.pos))

        return StepResult(self._obs(), reward, terminated, truncated, self._info(building_collision, energy_step, reward_terms))

    def metrics(self):
        return {
            "success": bool(np.linalg.norm(self.goal - self.pos) < self.goal_radius),
            "steps": self.steps,
            "sim_time_s": self.steps * self.dt,
            "path_length_m": float(self.path_length),
            "energy_relative_airspeed_l2": float(self.energy),
            "collisions": int(self.collisions),
            "boundary_violations": int(self.boundary_violations),
            "separation_violations": int(self.separation_violations),
            "min_building_clearance_m": float(self.min_building_clearance_m),
            "min_pairwise_separation_m": float(self.min_pairwise_separation_m),
            "final_distance_m": float(np.linalg.norm(self.goal - self.pos)),
            "return": float(self.reward_total),
            "reward_terms_total": self.reward_terms_total.copy(),
            "policy_had_privileged_flow_access": False,
        }

    def spec(self) -> dict:
        def space_dict(space) -> dict:
            if hasattr(space, "to_dict"):
                return space.to_dict()
            return {
                "type": space.__class__.__name__,
                "shape": list(space.shape),
                "dtype": str(space.dtype),
                "low": np.asarray(space.low).tolist(),
                "high": np.asarray(space.high).tolist(),
                "provider": "gymnasium",
            }

        return {
            "id": self.metadata["name"],
            "metadata": self.metadata,
            "gymnasium_installed": gym_spaces is not None,
            "reset_returns": ["observation", "info"],
            "step_returns": ["observation", "reward", "terminated", "truncated", "info"],
            "observation_space": space_dict(self.observation_space),
            "action_space": space_dict(self.action_space),
            "reward_terms": [
                "progress",
                "path_cost",
                "time_cost",
                "energy_cost",
                "collision_cost",
                "boundary_cost",
                "separation_cost",
                "goal_bonus",
            ],
            "cost_metrics": [
                "collisions",
                "boundary_violations",
                "separation_violations",
                "min_building_clearance_m",
                "min_pairwise_separation_m",
                "energy_relative_airspeed_l2",
                "path_length_m",
            ],
            "policy_observation_contract": policy_observation_contract(),
            "dt_s": self.dt,
            "max_steps": self.max_steps,
            "max_speed_mps": self.max_speed,
            "max_acc_mps2": self.max_acc,
            "goal_radius_m": self.goal_radius,
            "world": self.world.to_dict(),
            "wind_model": self.wind.to_dict(),
        }
