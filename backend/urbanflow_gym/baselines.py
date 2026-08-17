from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Callable, Protocol

import numpy as np

from .env import UrbanFlowConfig
from .observation import ActorSnapshot
from .scenario import PublicMissionContext


DIRECT_GOAL = "direct_goal"
SHORTEST_PATH = "shortest_path"
WIND_AWARE = "wind_aware_inlet"
BASELINE_ORDER = (DIRECT_GOAL, SHORTEST_PATH, WIND_AWARE)


class GuidanceBaseline(Protocol):
    baseline_id: str
    label: str
    uses_hidden_flow: bool
    waypoints: list[np.ndarray]

    def action(self, state: ActorSnapshot) -> np.ndarray:
        ...


@dataclass
class DirectGoalBaseline:
    context: PublicMissionContext
    config: UrbanFlowConfig
    baseline_id: str = DIRECT_GOAL
    label: str = "Direct goal guidance"
    uses_hidden_flow: bool = False

    def __post_init__(self) -> None:
        self.waypoints = [self.context.start_xy.copy(), self.context.goal_xy.copy()]

    def action(self, state: ActorSnapshot) -> np.ndarray:
        desired = _desired_world_velocity(
            state.position_xy,
            state.ground_velocity_xy,
            state.goal_xy,
            self.config.max_ground_speed_mps,
            final_target=True,
        )
        return world_velocity_to_local_action(desired, state.heading_rad, self.config.max_ground_speed_mps)


class WaypointGuidanceBaseline:
    uses_hidden_flow = False

    def __init__(
        self,
        context: PublicMissionContext,
        config: UrbanFlowConfig,
        *,
        wind_aware: bool,
    ) -> None:
        self.context = context
        self.config = config
        self.wind_aware = bool(wind_aware)
        self.baseline_id = WIND_AWARE if wind_aware else SHORTEST_PATH
        self.label = (
            "Wind-aware A* + inlet feed-forward"
            if wind_aware
            else "Geometry-safe shortest-path A*"
        )
        self.waypoints = plan_geometry_safe_path(
            context,
            config,
            wind_aware=wind_aware,
        )
        self._waypoint_index = 1 if len(self.waypoints) > 1 else 0

    def action(self, state: ActorSnapshot) -> np.ndarray:
        while self._waypoint_index < len(self.waypoints) - 1:
            waypoint = self.waypoints[self._waypoint_index]
            if float(np.linalg.norm(waypoint - state.position_xy)) > 4.2:
                break
            self._waypoint_index += 1
        target = self.waypoints[self._waypoint_index]
        final_target = self._waypoint_index == len(self.waypoints) - 1
        desired = _desired_world_velocity(
            state.position_xy,
            state.ground_velocity_xy,
            target,
            self.config.max_ground_speed_mps,
            final_target=final_target,
        )
        if self.wind_aware:
            # Steady-state compensation for the public dynamics model using the
            # mission-known inlet only. No local provider sample is available here.
            desired = desired + (
                self.config.velocity_tracking_time_s
                * self.config.wind_drag_gain_per_s
                * (desired - self.context.known_inlet_velocity_xy)
            )
        speed = float(np.linalg.norm(desired))
        if speed > self.config.max_ground_speed_mps:
            desired *= self.config.max_ground_speed_mps / speed
        return world_velocity_to_local_action(
            desired,
            state.heading_rad,
            self.config.max_ground_speed_mps,
        )


def make_baseline(
    baseline_id: str,
    context: PublicMissionContext,
    config: UrbanFlowConfig,
) -> GuidanceBaseline:
    factories: dict[str, Callable[[], GuidanceBaseline]] = {
        DIRECT_GOAL: lambda: DirectGoalBaseline(context, config),
        SHORTEST_PATH: lambda: WaypointGuidanceBaseline(
            context, config, wind_aware=False
        ),
        WIND_AWARE: lambda: WaypointGuidanceBaseline(
            context, config, wind_aware=True
        ),
    }
    try:
        return factories[baseline_id]()
    except KeyError as exc:
        raise ValueError(
            f"unknown baseline '{baseline_id}'; expected one of {', '.join(BASELINE_ORDER)}"
        ) from exc


def world_velocity_to_local_action(
    desired_world_velocity_xy: np.ndarray,
    heading_rad: float,
    max_ground_speed_mps: float,
) -> np.ndarray:
    desired = np.asarray(desired_world_velocity_xy, dtype=float)
    forward = np.array([math.cos(heading_rad), math.sin(heading_rad)], dtype=float)
    left = np.array([-forward[1], forward[0]], dtype=float)
    local = np.array(
        [float(np.dot(desired, forward)), float(np.dot(desired, left))],
        dtype=float,
    ) / max(max_ground_speed_mps, 1e-9)
    local = np.clip(local, -1.0, 1.0)
    norm = float(np.linalg.norm(local))
    return local / norm if norm > 1.0 else local


def plan_geometry_safe_path(
    context: PublicMissionContext,
    config: UrbanFlowConfig,
    *,
    wind_aware: bool,
    resolution_m: float = 4.0,
    safety_margin_m: float = 2.4,
) -> list[np.ndarray]:
    geometry = context.geometry
    start = context.start_xy
    goal = context.goal_xy
    radius = config.agent_radius_m
    if geometry.segment_is_free(start, goal, radius, safety_margin_m):
        return [start.copy(), goal.copy()]

    x0, x1, y0, y1 = geometry.bounds_xy
    # Keep live 800 m OSM worlds within the public endpoint's bounded runtime;
    # the small synthetic fixtures retain the original 4 m resolution.
    resolution_m = max(
        float(resolution_m),
        geometry.width_m / 96.0,
        geometry.height_m / 96.0,
    )
    xs = np.arange(x0, x1 + resolution_m * 0.5, resolution_m, dtype=float)
    ys = np.arange(y0, y1 + resolution_m * 0.5, resolution_m, dtype=float)
    safe: set[tuple[int, int]] = set()
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            point = np.array([x, y], dtype=float)
            if geometry.point_is_free(point, radius, safety_margin_m):
                safe.add((ix, iy))
    if not safe:
        raise RuntimeError("geometry-safe planner found no free grid cells")

    def nearest_visible(point: np.ndarray) -> tuple[int, int]:
        candidates = sorted(
            safe,
            key=lambda node: float(
                (xs[node[0]] - point[0]) ** 2 + (ys[node[1]] - point[1]) ** 2
            ),
        )
        for node in candidates:
            grid_point = np.array([xs[node[0]], ys[node[1]]], dtype=float)
            if geometry.segment_is_free(point, grid_point, radius, safety_margin_m):
                return node
        raise RuntimeError("geometry-safe planner could not connect mission endpoint to grid")

    start_node = nearest_visible(start)
    goal_node = nearest_visible(goal)
    open_heap: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start_node)]
    cost_so_far = {start_node: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    neighbor_offsets = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    tie = 0.0
    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current == goal_node:
            break
        current_point = np.array([xs[current[0]], ys[current[1]]], dtype=float)
        for dx, dy in neighbor_offsets:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor not in safe:
                continue
            neighbor_point = np.array([xs[neighbor[0]], ys[neighbor[1]]], dtype=float)
            if not geometry.segment_is_free(
                current_point, neighbor_point, radius, safety_margin_m
            ):
                continue
            edge = neighbor_point - current_point
            distance = float(np.linalg.norm(edge))
            edge_cost = distance
            if wind_aware:
                direction = edge / max(distance, 1e-9)
                cruise = config.max_ground_speed_mps * 0.84
                predicted_ground_velocity = direction * cruise
                relative_air_velocity = (
                    predicted_ground_velocity - context.known_inlet_velocity_xy
                )
                predicted_energy = (
                    float(np.dot(relative_air_velocity, relative_air_velocity))
                    * distance
                    / max(cruise, 1e-9)
                )
                edge_cost += 0.075 * predicted_energy
            new_cost = cost_so_far[current] + edge_cost
            if new_cost + 1e-12 >= cost_so_far.get(neighbor, float("inf")):
                continue
            cost_so_far[neighbor] = new_cost
            parent[neighbor] = current
            heuristic = float(
                np.linalg.norm(neighbor_point - np.array([xs[goal_node[0]], ys[goal_node[1]]]))
            )
            tie += 1.0
            heapq.heappush(open_heap, (new_cost + heuristic, tie, neighbor))

    if goal_node not in cost_so_far:
        raise RuntimeError("geometry-safe A* could not find a route")
    nodes = [goal_node]
    while nodes[-1] != start_node:
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    raw_path = [start.copy()]
    raw_path.extend(np.array([xs[ix], ys[iy]], dtype=float) for ix, iy in nodes)
    raw_path.append(goal.copy())
    compact = _remove_duplicate_points(raw_path)
    return _line_of_sight_smooth(
        compact,
        context,
        config,
        safety_margin_m=safety_margin_m,
        preserve_wind_cost=wind_aware,
    )


def _line_of_sight_smooth(
    path: list[np.ndarray],
    context: PublicMissionContext,
    config: UrbanFlowConfig,
    *,
    safety_margin_m: float,
    preserve_wind_cost: bool,
) -> list[np.ndarray]:
    if len(path) <= 2:
        return [point.copy() for point in path]
    result = [path[0].copy()]
    index = 0
    while index < len(path) - 1:
        next_index = index + 1
        for candidate in range(len(path) - 1, index, -1):
            if not context.geometry.segment_is_free(
                path[index],
                path[candidate],
                config.agent_radius_m,
                safety_margin_m,
            ):
                continue
            if preserve_wind_cost and candidate > index + 1:
                direct_cost = _public_inlet_path_cost(
                    [path[index], path[candidate]], context, config
                )
                original_cost = _public_inlet_path_cost(
                    path[index : candidate + 1], context, config
                )
                if direct_cost > original_cost * 1.01:
                    continue
            next_index = candidate
            break
        result.append(path[next_index].copy())
        index = next_index
    return _remove_duplicate_points(result)


def _public_inlet_path_cost(
    path: list[np.ndarray],
    context: PublicMissionContext,
    config: UrbanFlowConfig,
) -> float:
    cost = 0.0
    cruise = config.max_ground_speed_mps * 0.84
    for index in range(1, len(path)):
        edge = path[index] - path[index - 1]
        distance = float(np.linalg.norm(edge))
        if distance < 1e-9:
            continue
        velocity = edge / distance * cruise
        relative = velocity - context.known_inlet_velocity_xy
        energy = float(np.dot(relative, relative)) * distance / cruise
        cost += distance + 0.075 * energy
    return cost


def _desired_world_velocity(
    position_xy: np.ndarray,
    ground_velocity_xy: np.ndarray,
    target_xy: np.ndarray,
    max_ground_speed_mps: float,
    *,
    final_target: bool,
) -> np.ndarray:
    delta = np.asarray(target_xy, dtype=float) - np.asarray(position_xy, dtype=float)
    distance = float(np.linalg.norm(delta))
    if distance < 1e-9:
        return -0.6 * np.asarray(ground_velocity_xy, dtype=float)
    direction = delta / distance
    cruise = max_ground_speed_mps * 0.84
    if final_target:
        target_speed = min(cruise, max(0.5, distance * 0.95))
        damping = 0.20
    else:
        target_speed = min(cruise, max(2.5, distance * 1.25))
        damping = 0.08
    desired = direction * target_speed - damping * np.asarray(ground_velocity_xy, dtype=float)
    speed = float(np.linalg.norm(desired))
    return desired * (max_ground_speed_mps / speed) if speed > max_ground_speed_mps else desired


def _remove_duplicate_points(path: list[np.ndarray]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for point in path:
        candidate = np.asarray(point, dtype=float)
        if not result or float(np.linalg.norm(candidate - result[-1])) > 1e-6:
            result.append(candidate.copy())
    return result
