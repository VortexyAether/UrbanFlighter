from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from gymnasium import spaces as gym_spaces
except ImportError:
    gym_spaces = None


OBSERVATION_FIELD_NAMES = [
    "relative_goal_vector_xyz_normalized",
    "drone_velocity_xyz_normalized",
    "inlet_base_wind_xyz_normalized",
    "nearest_building_clearance_normalized",
    "sector_clearance_px",
    "sector_clearance_py",
    "sector_clearance_nx",
    "sector_clearance_ny",
    "sector_clearance_pxy",
    "sector_clearance_nxy",
    "sector_clearance_npxy",
    "sector_clearance_pnxy",
]

SECTOR_DIRECTIONS = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0],
    ],
    dtype=float,
)
SECTOR_DIRECTIONS = SECTOR_DIRECTIONS / np.linalg.norm(SECTOR_DIRECTIONS, axis=1, keepdims=True)


@dataclass(frozen=True)
class BoxSpace:
    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...]
    dtype: np.dtype

    def sample(self) -> np.ndarray:
        return np.zeros(self.shape, dtype=self.dtype)

    def contains(self, value: np.ndarray) -> bool:
        candidate = np.asarray(value, dtype=self.dtype)
        return bool(candidate.shape == self.shape and np.all(candidate >= self.low) and np.all(candidate <= self.high))

    def to_dict(self) -> dict:
        return {
            "type": "Box",
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "provider": "fallback",
        }


def observation_bounds() -> tuple[np.ndarray, np.ndarray]:
    low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -3.0, -3.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    return low, high


def make_box_space(low: np.ndarray, high: np.ndarray):
    dtype = np.dtype(np.float32)
    shape = tuple(int(v) for v in low.shape)
    low32 = low.astype(dtype)
    high32 = high.astype(dtype)
    if gym_spaces is not None:
        return gym_spaces.Box(low=low32, high=high32, shape=shape, dtype=dtype)
    return BoxSpace(low=low32, high=high32, shape=shape, dtype=dtype)


def build_policy_observation(world, base_wind: np.ndarray, pos: np.ndarray, goal: np.ndarray, vel: np.ndarray, max_speed: float) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = world.bounds
    scale = np.array([max(x1 - x0, 1.0), max(y1 - y0, 1.0), max(z1 - z0, 1.0)])
    geometry = _geometry_features(world, pos)
    return np.concatenate([
        np.clip((goal - pos) / scale, -1.0, 1.0),
        vel / max_speed,
        base_wind / 15.0,
        geometry,
    ]).astype(np.float32)


def policy_observation_contract() -> dict:
    return {
        "fields": OBSERVATION_FIELD_NAMES,
        "source": "mission_relative_state_plus_inlet_wind_plus_osm_building_range_features",
        "privileged_flow_access": False,
        "hidden_dynamics": "UrbanWindField.at(pos,t) / CFD-lite flow remains simulator dynamics only",
        "forbidden": [
            "absolute_position",
            "full_flow_grid",
            "hidden_flow_field_grid",
            "local_cfd_velocity_as_policy_input",
            "future_trajectory",
        ],
    }


def _geometry_features(world, pos: np.ndarray) -> np.ndarray:
    x0, x1, y0, y1, _, _ = world.bounds
    max_range = max(float(x1 - x0), float(y1 - y0), 1.0)
    nearest = min(world.nearest_building_clearance(pos), max_range) / max_range
    sector_ranges = [_sector_clearance(world, pos, direction, max_range) / max_range for direction in SECTOR_DIRECTIONS]
    return np.array([nearest, *sector_ranges], dtype=float)


def _sector_clearance(world, pos: np.ndarray, direction_xy: np.ndarray, max_range: float) -> float:
    xy = pos[:2]
    distances = _boundary_intersections(world, xy, direction_xy)
    for building in world.buildings:
        if pos[2] > building.height + 4.0:
            continue
        distance = _box_intersection_distance(building.min_xy, building.max_xy, xy, direction_xy)
        if distance is not None:
            distances.append(distance)
    positive = [d for d in distances if d >= 0.0]
    return min(positive, default=max_range)


def _boundary_intersections(world, xy: np.ndarray, direction_xy: np.ndarray) -> list[float]:
    x0, x1, y0, y1, _, _ = world.bounds
    distances: list[float] = []
    if direction_xy[0] > 1e-6:
        distances.append(float((x1 - xy[0]) / direction_xy[0]))
    elif direction_xy[0] < -1e-6:
        distances.append(float((x0 - xy[0]) / direction_xy[0]))
    if direction_xy[1] > 1e-6:
        distances.append(float((y1 - xy[1]) / direction_xy[1]))
    elif direction_xy[1] < -1e-6:
        distances.append(float((y0 - xy[1]) / direction_xy[1]))
    return distances


def _box_intersection_distance(bmin: np.ndarray, bmax: np.ndarray, xy: np.ndarray, direction_xy: np.ndarray) -> float | None:
    inv = np.divide(1.0, direction_xy, out=np.full(2, np.inf), where=np.abs(direction_xy) > 1e-6)
    t1 = (bmin - xy) * inv
    t2 = (bmax - xy) * inv
    near = np.maximum.reduce(np.minimum(t1, t2))
    far = np.minimum.reduce(np.maximum(t1, t2))
    if far >= max(near, 0.0):
        return float(max(near, 0.0))
    return None
