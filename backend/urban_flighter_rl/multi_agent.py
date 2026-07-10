from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .env import UrbanFlighterEnv
from .frames import initial_policy_frame, step_policy_frame
from .planner import _grid_astar, _pd_action_to_target
from .render import write_metrics


@dataclass
class DroneRun:
    drone_id: str
    start: np.ndarray
    goal: np.ndarray
    trajectory: list[np.ndarray]
    frames: list[dict]
    metrics: dict
    waypoints: list[np.ndarray]


def _nearest_free_point(world, candidate: np.ndarray, clearance_m: float = 2.5) -> np.ndarray:
    """Nudge a start/goal point upward/inward until it is collision-free."""
    p = np.array(candidate, dtype=float)
    x0, x1, y0, y1, z0, z1 = world.bounds
    p[0] = float(np.clip(p[0], x0 + 5.0, x1 - 5.0))
    p[1] = float(np.clip(p[1], y0 + 5.0, y1 - 5.0))
    p[2] = float(np.clip(p[2], z0 + 10.0, z1 - 8.0))
    if not world.collides(p, margin=clearance_m):
        return p

    # Try climbing first, then a small deterministic spiral in XY.
    for dz in np.linspace(5.0, max(12.0, z1 - p[2] - 4.0), 8):
        q = p + np.array([0.0, 0.0, dz])
        if q[2] < z1 - 3.0 and not world.collides(q, margin=clearance_m):
            return q
    for r in np.linspace(8.0, 40.0, 7):
        for theta in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
            q = p + np.array([r * np.cos(theta), r * np.sin(theta), 18.0])
            q[0] = float(np.clip(q[0], x0 + 5.0, x1 - 5.0))
            q[1] = float(np.clip(q[1], y0 + 5.0, y1 - 5.0))
            q[2] = float(np.clip(q[2], z0 + 12.0, z1 - 5.0))
            if not world.collides(q, margin=clearance_m):
                return q
    return p


def generate_cross_city_missions(world, n_drones: int = 6, altitude_m: float = 24.0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create opposing edge-to-edge missions so paths visibly cross through the city.

    Cruise altitude is automatically lifted above the tallest OSM prism. That keeps
    the multi-drone demo focused on airspace deconfliction and wind-aware energy,
    not repeatedly scraping roofs because the early controller is still simple.
    """
    x0, x1, y0, y1, z0, z1 = world.bounds
    max_building_h = max((b.height for b in world.buildings), default=0.0)
    z = min(max(altitude_m, max_building_h + 18.0, z0 + 14.0), z1 - 8.0)
    mission_templates = [((0.10, 0.12), (0.90, 0.88)), ((0.12, 0.88), (0.88, 0.12)),
                         ((0.50, 0.08), (0.50, 0.92)), ((0.08, 0.50), (0.92, 0.50)),
                         ((0.25, 0.10), (0.78, 0.90)), ((0.90, 0.30), (0.12, 0.72)),
                         ((0.30, 0.92), (0.72, 0.10)), ((0.10, 0.35), (0.90, 0.65))]
    missions: list[tuple[np.ndarray, np.ndarray]] = []
    for i, (s_frac, g_frac) in enumerate(mission_templates[:n_drones]):
        layer_offset = (i % 3) * 8.0
        zi = min(z + layer_offset, z1 - 5.0)
        start = np.array([x0 + s_frac[0] * (x1 - x0), y0 + s_frac[1] * (y1 - y0), zi], dtype=float)
        goal = np.array([x0 + g_frac[0] * (x1 - x0), y0 + g_frac[1] * (y1 - y0), zi], dtype=float)
        missions.append((_nearest_free_point(world, start), _nearest_free_point(world, goal)))
    return missions


def generate_random_missions(
    world,
    rng: np.random.Generator,
    n_drones: int = 6,
    altitude_m: float = 24.0,
    clearance_m: float = 3.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    x0, x1, y0, y1, z0, z1 = world.bounds
    max_building_h = max((b.height for b in world.buildings), default=0.0)
    z_base = min(max(altitude_m, max_building_h + 10.0, z0 + 14.0), z1 - 8.0)
    min_trip = 0.42 * float(np.hypot(x1 - x0, y1 - y0))
    missions: list[tuple[np.ndarray, np.ndarray]] = []

    def sample_point(layer: int) -> np.ndarray:
        for _ in range(240):
            p = np.array([rng.uniform(x0 + 8.0, x1 - 8.0), rng.uniform(y0 + 8.0, y1 - 8.0),
                          min(z_base + (layer % 4) * 7.0, z1 - 5.0)], dtype=float)
            if not world.collides(p, margin=clearance_m):
                return p
        fallback = np.array([x0 + 0.15 * (x1 - x0), y0 + 0.15 * (y1 - y0), z_base], dtype=float)
        return _nearest_free_point(world, fallback, clearance_m=clearance_m)

    for i in range(n_drones):
        start = sample_point(i)
        goal = sample_point(i + n_drones)
        for _ in range(160):
            if float(np.linalg.norm(goal - start)) >= min_trip:
                break
            goal = sample_point(i + n_drones)
        missions.append((start, goal))
    return missions


def _separation_action(drone_i: UrbanFlighterEnv, drones: list[UrbanFlighterEnv], min_sep_m: float = 18.0) -> np.ndarray:
    force = np.zeros(3, dtype=float)
    for other in drones:
        if other is drone_i:
            continue
        delta = drone_i.pos - other.pos
        d = float(np.linalg.norm(delta))
        if 1e-6 < d < min_sep_m:
            # Stronger horizontal deconfliction, mild vertical split.
            force += (delta / d) * ((min_sep_m - d) / min_sep_m) ** 2
            force[2] += 0.12 * np.sign(delta[2] if abs(delta[2]) > 1e-3 else 1.0)
    n = np.linalg.norm(force)
    return force / n if n > 1.0 else force


def run_multi_drone_baseline(
    world,
    wind,
    n_drones: int = 6,
    max_steps: int = 1100,
    seed: int = 7,
    randomize_missions: bool = True,
) -> tuple[list[DroneRun], dict]:
    """Multi-agent RL-style rollout: shared world/wind, individual rewards, drone separation penalty.

    This is not a trained policy yet. It is a deterministic multi-agent baseline and data generator
    with the same observation/reward/relative-air-speed energy mechanics that a MARL policy can use.
    """
    rng = np.random.default_rng(seed)
    missions = (
        generate_random_missions(world, rng, n_drones=n_drones)
        if randomize_missions
        else generate_cross_city_missions(world, n_drones=n_drones)
    )
    drones = [UrbanFlighterEnv(world=world, wind=wind, max_steps=max_steps, start=s, goal=g) for s, g in missions]
    frame_sets: list[list[dict]] = []
    for i, drone in enumerate(drones):
        obs, info = drone.reset(seed=seed + i)
        frame_sets.append([initial_policy_frame(f"drone_{i+1}", drone, obs, info)])

    max_building_h = max((b.height for b in world.buildings), default=0.0)
    safe_z = min(max(max_building_h + 18.0, 26.0), world.bounds[5] - 8.0)
    z_levels = tuple(sorted(set(round(z, 1) for z in (
        safe_z,
        min(safe_z + 10.0, world.bounds[5] - 5.0),
        min(safe_z + 20.0, world.bounds[5] - 3.0),
    ))))
    waypoint_sets = [_grid_astar(drone, spacing=7.5, z_levels=z_levels) for drone in drones]
    waypoint_indices = [1 if len(wp) > 1 else 0 for wp in waypoint_sets]
    returns = [0.0 for _ in drones]
    near_miss_count = 0
    min_pairwise_sep = float("inf")

    for _ in range(max_steps):
        all_done = True
        positions_before = [drone.pos.copy() for drone in drones]
        for i, drone in enumerate(drones):
            if np.linalg.norm(drone.goal - drone.pos) < drone.goal_radius:
                continue
            all_done = False
            waypoints = waypoint_sets[i]
            wp_i = waypoint_indices[i]
            target = waypoints[wp_i]
            if np.linalg.norm(target - drone.pos) < 4.0 and wp_i < len(waypoints) - 1:
                wp_i += 1
                waypoint_indices[i] = wp_i
                target = waypoints[wp_i]
            action = (
                _pd_action_to_target(drone, target)
                + 0.95 * _separation_action(drone, drones, min_sep_m=24.0)
                + 0.45 * drone.world.obstacle_repulsion(drone.pos, radius=24.0)
            )
            altitude_error = float(target[2] - drone.pos[2])
            altitude_hold = np.array([0.0, 0.0, np.clip(altitude_error / 8.0 - drone.vel[2] / 10.0, -0.55, 0.55)])
            action += altitude_hold
            n = np.linalg.norm(action)
            if n > 1.0:
                action = action / n
            drone.set_neighbor_positions([other.pos for j, other in enumerate(drones) if j != i])
            result = drone.step(action)
            returns[i] += result.reward
            frame_sets[i].append(step_policy_frame(f"drone_{i+1}", drone, action, result))
        for i in range(len(positions_before)):
            for j in range(i + 1, len(positions_before)):
                sep = float(np.linalg.norm(drones[i].pos - drones[j].pos))
                min_pairwise_sep = min(min_pairwise_sep, sep)
                if sep < 10.0:
                    near_miss_count += 1
        if all_done:
            break

    runs: list[DroneRun] = []
    for i, drone in enumerate(drones):
        metrics = drone.metrics()
        metrics.update({
            "drone_id": f"drone_{i+1}",
            "return": returns[i],
            "controller": "multi_agent_wind_aware_astar_pd_with_separation",
            "waypoint_count": len(waypoint_sets[i]),
            "start": drone.start.tolist(),
            "goal": drone.goal.tolist(),
            "world": world.to_dict(),
        })
        runs.append(DroneRun(
            drone_id=f"drone_{i+1}",
            start=drone.start.copy(),
            goal=drone.goal.copy(),
            trajectory=[p.copy() for p in drone.trajectory],
            frames=frame_sets[i],
            metrics=metrics,
            waypoints=waypoint_sets[i],
        ))

    swept_building_hits = sum(
        1
        for run in runs
        for index in range(1, len(run.trajectory))
        if world.segment_hits_building(run.trajectory[index - 1], run.trajectory[index], margin=0.75)
    )
    aggregate = {
        "controller": "multi_agent_wind_aware_astar_pd_with_separation",
        "n_drones": len(runs),
        "success_count": int(sum(r.metrics["success"] for r in runs)),
        "all_success": bool(all(r.metrics["success"] for r in runs)),
        "total_collisions": int(sum(r.metrics["collisions"] for r in runs)),
        "total_boundary_violations": int(sum(r.metrics["boundary_violations"] for r in runs)),
        "swept_building_hits": int(swept_building_hits),
        "collision_validation": "exact swept segment check against building prisms, margin 0.75m",
        "total_energy_relative_airspeed_l2": float(sum(r.metrics["energy_relative_airspeed_l2"] for r in runs)),
        "total_path_length_m": float(sum(r.metrics["path_length_m"] for r in runs)),
        "max_steps_used": int(max(r.metrics["steps"] for r in runs)),
        "min_pairwise_separation_m": float(min_pairwise_sep),
        "near_miss_count_sep_lt_10m": int(near_miss_count),
        "energy_model": "sum over drones,t ||v_ground - w_local(x,y,z,t)||^2 * dt",
        "note": "RL-style multi-agent rollout generator; policy is deterministic baseline, not trained MARL yet.",
        "mission_sampling": "seeded_random_collision_free" if randomize_missions else "deterministic_cross_city_templates",
        "seed": int(seed),
        "policy_had_privileged_flow_access": False,
    }
    return runs, aggregate


def write_multi_metrics(runs: list[DroneRun], aggregate: dict, output_path: str | Path):
    payload = {
        "aggregate": aggregate,
        "drones": [r.metrics for r in runs],
    }
    write_metrics(payload, output_path)


def write_trajectories_json(runs: list[DroneRun], output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "world": runs[0].metrics.get("world", None) if runs else None,
        "drones": [
            {
                "drone_id": r.drone_id,
                "start": r.start.tolist(),
                "goal": r.goal.tolist(),
                "mission": {"start": r.start.tolist(), "goal": r.goal.tolist()},
                "trajectory": [p.tolist() for p in r.trajectory],
                "frames": r.frames,
                "waypoints": [p.tolist() for p in r.waypoints],
            }
            for r in runs
        ]
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
