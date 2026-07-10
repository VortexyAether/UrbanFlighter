#!/usr/bin/env python3
"""Validate Urban_Flighter multi-drone replay metrics and trajectories.

This checks replay-data consistency and sampled swept segment clearance against the
building prisms embedded in the replay payload.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def finite_number(value: Any, name: str) -> float:
    require(isinstance(value, (int, float)), f"{name} must be numeric")
    value = float(value)
    require(math.isfinite(value), f"{name} must be finite")
    return value


def point3(value: Any, name: str) -> tuple[float, float, float]:
    require(isinstance(value, list) and len(value) == 3, f"{name} must be [x,y,z]")
    x, y, z = value
    return (
        finite_number(x, f"{name}[0]"),
        finite_number(y, f"{name}[1]"),
        finite_number(z, f"{name}[2]"),
    )


def dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def path_length(points: list[tuple[float, float, float]]) -> float:
    return sum(dist(points[i - 1], points[i]) for i in range(1, len(points)))


def _segment_intersects_building(a: tuple[float, float, float], b: tuple[float, float, float], building: dict[str, Any], margin: float) -> bool:
    cx, cy = building["center"]
    sx, sy = building["size"]
    mn = (cx - sx / 2.0 - margin, cy - sy / 2.0 - margin, 0.0)
    mx = (cx + sx / 2.0 + margin, cy + sy / 2.0 + margin, float(building["height_m"]) + margin)
    direction = tuple(b[i] - a[i] for i in range(3))
    t_min = 0.0
    t_max = 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-9:
            if a[axis] < mn[axis] or a[axis] > mx[axis]:
                return False
            continue
        inv = 1.0 / direction[axis]
        t1 = (mn[axis] - a[axis]) * inv
        t2 = (mx[axis] - a[axis]) * inv
        near = min(t1, t2)
        far = max(t1, t2)
        t_min = max(t_min, near)
        t_max = min(t_max, far)
        if t_min > t_max:
            return False
    return True


def _segment_hits_building(a: tuple[float, float, float], b: tuple[float, float, float], buildings: list[dict[str, Any]], margin: float = 0.75) -> bool:
    for building in buildings:
        if _segment_intersects_building(a, b, building, margin):
            return True
    return False


def validate(metrics_path: Path, trajectories_path: Path, tolerance: float = 1e-6) -> dict[str, Any]:
    metrics = load_json(metrics_path)
    trajectories = load_json(trajectories_path)

    require(isinstance(metrics, dict), "metrics root must be an object")
    require(isinstance(trajectories, dict), "trajectories root must be an object")
    aggregate = metrics.get("aggregate")
    drones_metrics = metrics.get("drones")
    drones_traj = trajectories.get("drones")
    require(isinstance(aggregate, dict), "metrics.aggregate missing")
    require(isinstance(drones_metrics, list), "metrics.drones missing")
    require(isinstance(drones_traj, list), "trajectories.drones missing")
    require(len(drones_metrics) == len(drones_traj), "metrics/trajectory drone count mismatch")

    n_drones = int(finite_number(aggregate.get("n_drones"), "aggregate.n_drones"))
    require(n_drones == len(drones_traj), "aggregate.n_drones mismatch")
    require("not trained" in str(aggregate.get("note", "")).lower(), "aggregate.note must state not trained")
    require("||v_ground - w_local" in str(aggregate.get("energy_model", "")), "energy model must preserve relative-air-speed formula")

    traj_by_id = {drone.get("drone_id"): drone for drone in drones_traj}
    require(len(traj_by_id) == len(drones_traj), "trajectory drone IDs must be unique")

    success_count = 0
    total_collisions = 0
    total_path = 0.0
    total_energy = 0.0
    all_positions_by_frame: list[list[tuple[float, float, float]]] = []
    swept_building_hits = 0
    world = trajectories.get("world") or {}
    buildings = world.get("buildings") if isinstance(world, dict) else None
    buildings = buildings if isinstance(buildings, list) else []
    require(len(buildings) > 0, "trajectories.world.buildings must contain replay building prisms")

    for idx, metric in enumerate(drones_metrics):
        drone_id = metric.get("drone_id")
        require(drone_id in traj_by_id, f"missing trajectory for {drone_id}")
        traj = traj_by_id[drone_id]
        points = [point3(p, f"{drone_id}.trajectory[{i}]") for i, p in enumerate(traj.get("trajectory", []))]
        require(len(points) >= 2, f"{drone_id} trajectory too short")
        require(point3(metric.get("start"), f"{drone_id}.metrics.start") == point3(traj.get("start"), f"{drone_id}.trajectory.start"), f"{drone_id} start mismatch")
        require(point3(metric.get("goal"), f"{drone_id}.metrics.goal") == point3(traj.get("goal"), f"{drone_id}.trajectory.goal"), f"{drone_id} goal mismatch")

        expected_steps = len(points) - 1
        observed_steps = int(finite_number(metric.get("steps"), f"{drone_id}.steps"))
        require(observed_steps == expected_steps, f"{drone_id} steps mismatch: {observed_steps} != {expected_steps}")

        observed_path = finite_number(metric.get("path_length_m"), f"{drone_id}.path_length_m")
        recomputed_path = path_length(points)
        require(abs(observed_path - recomputed_path) <= max(tolerance, observed_path * 1e-9), f"{drone_id} path length mismatch")
        for i in range(1, len(points)):
            if _segment_hits_building(points[i - 1], points[i], buildings):
                swept_building_hits += 1

        goal = point3(metric.get("goal"), f"{drone_id}.goal")
        observed_final = finite_number(metric.get("final_distance_m"), f"{drone_id}.final_distance_m")
        require(abs(observed_final - dist(points[-1], goal)) <= 1e-6, f"{drone_id} final distance mismatch")

        success_count += 1 if metric.get("success") else 0
        total_collisions += int(finite_number(metric.get("collisions"), f"{drone_id}.collisions"))
        total_path += observed_path
        total_energy += finite_number(metric.get("energy_relative_airspeed_l2"), f"{drone_id}.energy_relative_airspeed_l2")

        for frame, point in enumerate(points):
            while len(all_positions_by_frame) <= frame:
                all_positions_by_frame.append([])
            all_positions_by_frame[frame].append(point)

    min_sep = math.inf
    near_miss_count = 0
    for frame_points in all_positions_by_frame:
        for i in range(len(frame_points)):
            for j in range(i + 1, len(frame_points)):
                separation = dist(frame_points[i], frame_points[j])
                min_sep = min(min_sep, separation)
                if separation < 10.0:
                    near_miss_count += 1

    require(success_count == int(finite_number(aggregate.get("success_count"), "aggregate.success_count")), "success_count mismatch")
    require(total_collisions == int(finite_number(aggregate.get("total_collisions"), "aggregate.total_collisions")), "total collisions mismatch")
    require(abs(total_path - finite_number(aggregate.get("total_path_length_m"), "aggregate.total_path_length_m")) <= 1e-6, "total path mismatch")
    require(abs(total_energy - finite_number(aggregate.get("total_energy_relative_airspeed_l2"), "aggregate.total_energy_relative_airspeed_l2")) <= 1e-6, "total energy mismatch")
    require(abs(min_sep - finite_number(aggregate.get("min_pairwise_separation_m"), "aggregate.min_pairwise_separation_m")) <= 1e-6, "min pairwise separation mismatch")
    require(near_miss_count == int(finite_number(aggregate.get("near_miss_count_sep_lt_10m"), "aggregate.near_miss_count_sep_lt_10m")), "near-miss count mismatch")
    require(swept_building_hits == 0, f"swept building collision hits found: {swept_building_hits}")

    return {
        "ok": True,
        "n_drones": n_drones,
        "success_count": success_count,
        "total_collisions": total_collisions,
        "swept_building_hits": swept_building_hits,
        "continuous_collision_check": "exact swept segment intersection against replay building AABB prisms with 0.75m margin",
        "near_miss_count_sep_lt_10m": near_miss_count,
        "min_pairwise_separation_m": min_sep,
        "total_path_length_m": total_path,
        "energy_model": aggregate.get("energy_model"),
        "honesty_note": aggregate.get("note"),
        "validated_scope": "replay JSON consistency plus exact swept segment checks against axis-aligned building prisms",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("results/multi_drone_gangnam_v3/urban_flighter_multi_drone_metrics.json"))
    parser.add_argument("--trajectories", type=Path, default=Path("results/multi_drone_gangnam_v3/urban_flighter_multi_drone_trajectories.json"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = validate(args.metrics, args.trajectories)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
