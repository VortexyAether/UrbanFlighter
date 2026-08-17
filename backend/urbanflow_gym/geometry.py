from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, TypeAlias

import numpy as np


@dataclass(frozen=True)
class AxisAlignedPrism:
    """Known static prism; the 2D slice flies at a fixed altitude below its top."""

    obstacle_id: str
    min_xy: np.ndarray
    max_xy: np.ndarray
    height_m: float

    def __post_init__(self) -> None:
        minimum = np.asarray(self.min_xy, dtype=float)
        maximum = np.asarray(self.max_xy, dtype=float)
        if minimum.shape != (2,) or maximum.shape != (2,):
            raise ValueError("prism bounds must be two-dimensional")
        if np.any(maximum <= minimum):
            raise ValueError("prism max_xy must be greater than min_xy")
        if not math.isfinite(self.height_m) or self.height_m <= 0.0:
            raise ValueError("prism height_m must be positive and finite")
        object.__setattr__(self, "min_xy", minimum)
        object.__setattr__(self, "max_xy", maximum)
        object.__setattr__(self, "height_m", float(self.height_m))

    @property
    def center_xy(self) -> np.ndarray:
        return (self.min_xy + self.max_xy) * 0.5

    @property
    def size_xy(self) -> np.ndarray:
        return self.max_xy - self.min_xy

    def expanded(self, margin_m: float) -> tuple[np.ndarray, np.ndarray]:
        margin = max(0.0, float(margin_m))
        return self.min_xy - margin, self.max_xy + margin

    def signed_clearance_xy(self, point_xy: np.ndarray) -> float:
        point = np.asarray(point_xy, dtype=float)
        outside = np.maximum(np.maximum(self.min_xy - point, point - self.max_xy), 0.0)
        outside_distance = float(np.linalg.norm(outside))
        if outside_distance > 0.0:
            return outside_distance
        inside_depth = min(
            float(point[0] - self.min_xy[0]),
            float(self.max_xy[0] - point[0]),
            float(point[1] - self.min_xy[1]),
            float(self.max_xy[1] - point[1]),
        )
        return -inside_depth

    def segment_collides_xy(
        self,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        radius_m: float = 0.0,
    ) -> bool:
        minimum, maximum = self.expanded(radius_m)
        return segment_intersects_aabb(start_xy, end_xy, minimum, maximum)

    def ray_distance_xy(
        self,
        origin_xy: np.ndarray,
        direction_xy: np.ndarray,
    ) -> float | None:
        return ray_aabb_distance(origin_xy, direction_xy, self.min_xy, self.max_xy)

    def to_dict(self) -> dict:
        minimum = self.min_xy.tolist()
        maximum = self.max_xy.tolist()
        return {
            "obstacle_id": self.obstacle_id,
            "min_xy": minimum,
            "max_xy": maximum,
            "height_m": self.height_m,
            "footprint": [
                [minimum[0], minimum[1]],
                [maximum[0], minimum[1]],
                [maximum[0], maximum[1]],
                [minimum[0], maximum[1]],
            ],
        }


@dataclass(frozen=True)
class PolygonPrism:
    """Extruded real-world polygon used by the live OSM 2D world.

    The horizontal footprint remains in the backend/UI local east/north frame.
    Collision and LiDAR operate on the exterior polygon at the configured 2D
    flight slice; height is preserved for provenance and 3D consumers.
    """

    obstacle_id: str
    footprint_xy: np.ndarray
    height_m: float

    def __post_init__(self) -> None:
        footprint = np.asarray(self.footprint_xy, dtype=float)
        if footprint.ndim != 2 or footprint.shape[1:] != (2,):
            raise ValueError("polygon footprint must have shape (n, 2)")
        if footprint.shape[0] >= 2 and np.allclose(footprint[0], footprint[-1], atol=1e-9):
            footprint = footprint[:-1]
        if footprint.shape[0] < 3:
            raise ValueError("polygon footprint must contain at least three vertices")
        if not np.all(np.isfinite(footprint)):
            raise ValueError("polygon footprint vertices must be finite")
        if abs(polygon_signed_area(footprint)) <= 1e-9:
            raise ValueError("polygon footprint must have non-zero area")
        if not math.isfinite(self.height_m) or self.height_m <= 0.0:
            raise ValueError("polygon height_m must be positive and finite")
        footprint = np.array(footprint, dtype=float, copy=True)
        footprint.setflags(write=False)
        object.__setattr__(self, "footprint_xy", footprint)
        object.__setattr__(self, "height_m", float(self.height_m))

    @property
    def min_xy(self) -> np.ndarray:
        return np.min(self.footprint_xy, axis=0)

    @property
    def max_xy(self) -> np.ndarray:
        return np.max(self.footprint_xy, axis=0)

    @property
    def center_xy(self) -> np.ndarray:
        return np.mean(self.footprint_xy, axis=0)

    @property
    def size_xy(self) -> np.ndarray:
        return self.max_xy - self.min_xy

    def signed_clearance_xy(self, point_xy: np.ndarray) -> float:
        point = np.asarray(point_xy, dtype=float)
        distance = min(
            point_segment_distance(point, start, end)
            for start, end in polygon_edges(self.footprint_xy)
        )
        return -float(distance) if point_in_polygon(point, self.footprint_xy) else float(distance)

    def segment_collides_xy(
        self,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        radius_m: float = 0.0,
    ) -> bool:
        start = np.asarray(start_xy, dtype=float)
        end = np.asarray(end_xy, dtype=float)
        radius = max(0.0, float(radius_m))
        minimum = self.min_xy - radius
        maximum = self.max_xy + radius
        swept_min = np.minimum(start, end)
        swept_max = np.maximum(start, end)
        if np.any(swept_max < minimum) or np.any(swept_min > maximum):
            return False
        if point_in_polygon(start, self.footprint_xy) or point_in_polygon(end, self.footprint_xy):
            return True
        radius_squared = radius * radius
        return any(
            segment_distance_squared(start, end, edge_start, edge_end)
            <= radius_squared + 1e-12
            for edge_start, edge_end in polygon_edges(self.footprint_xy)
        )

    def ray_distance_xy(
        self,
        origin_xy: np.ndarray,
        direction_xy: np.ndarray,
    ) -> float | None:
        origin = np.asarray(origin_xy, dtype=float)
        if point_in_polygon(origin, self.footprint_xy):
            return 0.0
        hits = [
            distance
            for start, end in polygon_edges(self.footprint_xy)
            if (distance := ray_segment_distance(origin, direction_xy, start, end)) is not None
        ]
        return min(hits, default=None)

    def to_dict(self) -> dict:
        return {
            "obstacle_id": self.obstacle_id,
            "min_xy": self.min_xy.tolist(),
            "max_xy": self.max_xy.tolist(),
            "height_m": self.height_m,
            "footprint": self.footprint_xy.tolist(),
            "footprint_model": "polygon_exterior",
        }


UrbanPrism: TypeAlias = AxisAlignedPrism | PolygonPrism


class UrbanGeometry:
    def __init__(
        self,
        bounds_xy: Iterable[float],
        prisms: Iterable[UrbanPrism],
        flight_altitude_m: float = 18.0,
    ) -> None:
        bounds = np.asarray(tuple(bounds_xy), dtype=float)
        if bounds.shape != (4,):
            raise ValueError("bounds_xy must be (min_x, max_x, min_y, max_y)")
        if bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
            raise ValueError("geometry bounds must have positive area")
        if not np.all(np.isfinite(bounds)):
            raise ValueError("geometry bounds must be finite")
        if not math.isfinite(float(flight_altitude_m)) or float(flight_altitude_m) < 0.0:
            raise ValueError("flight_altitude_m must be non-negative and finite")
        self.bounds_xy = bounds
        self.prisms = tuple(prisms)
        self.flight_altitude_m = float(flight_altitude_m)

    @property
    def width_m(self) -> float:
        return float(self.bounds_xy[1] - self.bounds_xy[0])

    @property
    def height_m(self) -> float:
        return float(self.bounds_xy[3] - self.bounds_xy[2])

    @property
    def diagonal_m(self) -> float:
        return float(math.hypot(self.width_m, self.height_m))

    def relevant_prisms(self) -> tuple[UrbanPrism, ...]:
        return tuple(prism for prism in self.prisms if prism.height_m >= self.flight_altitude_m)

    def normalize_position(self, point_xy: np.ndarray) -> np.ndarray:
        point = np.asarray(point_xy, dtype=float)
        x0, x1, y0, y1 = self.bounds_xy
        return np.array(
            [
                2.0 * (point[0] - x0) / max(x1 - x0, 1e-9) - 1.0,
                2.0 * (point[1] - y0) / max(y1 - y0, 1e-9) - 1.0,
            ],
            dtype=float,
        )

    def boundary_clearance(self, point_xy: np.ndarray) -> float:
        point = np.asarray(point_xy, dtype=float)
        x0, x1, y0, y1 = self.bounds_xy
        return float(min(point[0] - x0, x1 - point[0], point[1] - y0, y1 - point[1]))

    def clearance(self, point_xy: np.ndarray, agent_radius_m: float = 0.0) -> float:
        point = np.asarray(point_xy, dtype=float)
        boundary = self.boundary_clearance(point) - float(agent_radius_m)
        obstacle = min(
            (prism.signed_clearance_xy(point) - float(agent_radius_m) for prism in self.relevant_prisms()),
            default=float("inf"),
        )
        return float(min(boundary, obstacle))

    def point_is_free(self, point_xy: np.ndarray, agent_radius_m: float = 0.0, margin_m: float = 0.0) -> bool:
        point = np.asarray(point_xy, dtype=float)
        required = max(0.0, float(agent_radius_m) + float(margin_m))
        if self.boundary_clearance(point) <= required + 1e-9:
            return False
        for prism in self.relevant_prisms():
            if np.any(point < prism.min_xy - required) or np.any(point > prism.max_xy + required):
                continue
            if prism.signed_clearance_xy(point) <= required + 1e-9:
                return False
        return True

    def segment_collides(self, start_xy: np.ndarray, end_xy: np.ndarray, agent_radius_m: float = 0.0) -> bool:
        start = np.asarray(start_xy, dtype=float)
        end = np.asarray(end_xy, dtype=float)
        if not self.point_is_free(start, agent_radius_m) or not self.point_is_free(end, agent_radius_m):
            return True
        for prism in self.relevant_prisms():
            if prism.segment_collides_xy(start, end, agent_radius_m):
                return True
        return False

    def segment_is_free(
        self,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        agent_radius_m: float = 0.0,
        margin_m: float = 0.0,
    ) -> bool:
        return not self.segment_collides(start_xy, end_xy, agent_radius_m + margin_m)

    def segment_min_clearance(
        self,
        start_xy: np.ndarray,
        end_xy: np.ndarray,
        agent_radius_m: float = 0.0,
        sample_spacing_m: float = 0.5,
    ) -> float:
        start = np.asarray(start_xy, dtype=float)
        end = np.asarray(end_xy, dtype=float)
        length = float(np.linalg.norm(end - start))
        count = max(2, int(math.ceil(length / max(sample_spacing_m, 0.1))) + 1)
        return float(
            min(
                self.clearance(start + alpha * (end - start), agent_radius_m)
                for alpha in np.linspace(0.0, 1.0, count)
            )
        )

    def lidar_ranges(
        self,
        origin_xy: np.ndarray,
        world_angles_rad: np.ndarray,
        max_range_m: float,
    ) -> np.ndarray:
        origin = np.asarray(origin_xy, dtype=float)
        maximum = float(max_range_m)
        ranges: list[float] = []
        for angle in np.asarray(world_angles_rad, dtype=float):
            direction = np.array([math.cos(float(angle)), math.sin(float(angle))], dtype=float)
            distance = min(maximum, ray_domain_exit_distance(origin, direction, self.bounds_xy))
            for prism in self.relevant_prisms():
                hit = prism.ray_distance_xy(origin, direction)
                if hit is not None:
                    distance = min(distance, hit)
            ranges.append(max(0.0, min(maximum, distance)))
        return np.asarray(ranges, dtype=float)

    def to_dict(self) -> dict:
        return {
            "bounds_xy": self.bounds_xy.tolist(),
            "flight_altitude_m": self.flight_altitude_m,
            "prism_count": len(self.prisms),
            "prisms": [prism.to_dict() for prism in self.prisms],
        }


def polygon_signed_area(footprint_xy: np.ndarray) -> float:
    footprint = np.asarray(footprint_xy, dtype=float)
    x = footprint[:, 0]
    y = footprint[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def polygon_edges(footprint_xy: np.ndarray):
    footprint = np.asarray(footprint_xy, dtype=float)
    for index in range(len(footprint)):
        yield footprint[index], footprint[(index + 1) % len(footprint)]


def point_segment_distance(
    point_xy: np.ndarray,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
) -> float:
    point = np.asarray(point_xy, dtype=float)
    start = np.asarray(start_xy, dtype=float)
    end = np.asarray(end_xy, dtype=float)
    edge = end - start
    length_squared = float(np.dot(edge, edge))
    if length_squared <= 1e-18:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, edge) / length_squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + edge * fraction)))


def point_in_polygon(point_xy: np.ndarray, footprint_xy: np.ndarray) -> bool:
    """Match the browser's even/odd exterior-footprint rule, including edges."""

    point = np.asarray(point_xy, dtype=float)
    footprint = np.asarray(footprint_xy, dtype=float)
    if any(point_segment_distance(point, start, end) <= 1e-9 for start, end in polygon_edges(footprint)):
        return True
    inside = False
    previous = len(footprint) - 1
    for index in range(len(footprint)):
        x, y = footprint[index]
        previous_x, previous_y = footprint[previous]
        crosses_y = (y > point[1]) != (previous_y > point[1])
        denominator = previous_y - y
        edge_x = ((previous_x - x) * (point[1] - y)) / (denominator or np.finfo(float).eps) + x
        if crosses_y and point[0] < edge_x:
            inside = not inside
        previous = index
    return inside


def _cross(origin_xy: np.ndarray, a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    origin = np.asarray(origin_xy, dtype=float)
    a = np.asarray(a_xy, dtype=float)
    b = np.asarray(b_xy, dtype=float)
    return float((a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]))


def _point_on_segment(point_xy: np.ndarray, start_xy: np.ndarray, end_xy: np.ndarray) -> bool:
    point = np.asarray(point_xy, dtype=float)
    start = np.asarray(start_xy, dtype=float)
    end = np.asarray(end_xy, dtype=float)
    return bool(np.all(point >= np.minimum(start, end) - 1e-9) and np.all(point <= np.maximum(start, end) + 1e-9))


def segments_intersect(
    first_start_xy: np.ndarray,
    first_end_xy: np.ndarray,
    second_start_xy: np.ndarray,
    second_end_xy: np.ndarray,
) -> bool:
    c1 = _cross(first_start_xy, first_end_xy, second_start_xy)
    c2 = _cross(first_start_xy, first_end_xy, second_end_xy)
    c3 = _cross(second_start_xy, second_end_xy, first_start_xy)
    c4 = _cross(second_start_xy, second_end_xy, first_end_xy)
    if (
        ((c1 > 1e-9 and c2 < -1e-9) or (c1 < -1e-9 and c2 > 1e-9))
        and ((c3 > 1e-9 and c4 < -1e-9) or (c3 < -1e-9 and c4 > 1e-9))
    ):
        return True
    return bool(
        (abs(c1) <= 1e-9 and _point_on_segment(second_start_xy, first_start_xy, first_end_xy))
        or (abs(c2) <= 1e-9 and _point_on_segment(second_end_xy, first_start_xy, first_end_xy))
        or (abs(c3) <= 1e-9 and _point_on_segment(first_start_xy, second_start_xy, second_end_xy))
        or (abs(c4) <= 1e-9 and _point_on_segment(first_end_xy, second_start_xy, second_end_xy))
    )


def segment_distance_squared(
    first_start_xy: np.ndarray,
    first_end_xy: np.ndarray,
    second_start_xy: np.ndarray,
    second_end_xy: np.ndarray,
) -> float:
    if segments_intersect(first_start_xy, first_end_xy, second_start_xy, second_end_xy):
        return 0.0
    distances = (
        point_segment_distance(first_start_xy, second_start_xy, second_end_xy),
        point_segment_distance(first_end_xy, second_start_xy, second_end_xy),
        point_segment_distance(second_start_xy, first_start_xy, first_end_xy),
        point_segment_distance(second_end_xy, first_start_xy, first_end_xy),
    )
    return float(min(distances) ** 2)


def ray_segment_distance(
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
    start_xy: np.ndarray,
    end_xy: np.ndarray,
) -> float | None:
    origin = np.asarray(origin_xy, dtype=float)
    direction = np.asarray(direction_xy, dtype=float)
    start = np.asarray(start_xy, dtype=float)
    end = np.asarray(end_xy, dtype=float)
    segment = end - start
    denominator = float(direction[0] * segment[1] - direction[1] * segment[0])
    if abs(denominator) < 1e-9:
        return None
    offset = start - origin
    distance = float((offset[0] * segment[1] - offset[1] * segment[0]) / denominator)
    edge_fraction = float((offset[0] * direction[1] - offset[1] * direction[0]) / denominator)
    if distance >= 0.0 and 0.0 <= edge_fraction <= 1.0:
        return distance
    return None


def segment_intersects_aabb(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    minimum_xy: np.ndarray,
    maximum_xy: np.ndarray,
) -> bool:
    """Exact slab intersection for a closed 2D segment and axis-aligned box."""

    start = np.asarray(start_xy, dtype=float)
    direction = np.asarray(end_xy, dtype=float) - start
    minimum = np.asarray(minimum_xy, dtype=float)
    maximum = np.asarray(maximum_xy, dtype=float)
    t_near = 0.0
    t_far = 1.0
    for axis in range(2):
        if abs(direction[axis]) < 1e-12:
            if start[axis] < minimum[axis] or start[axis] > maximum[axis]:
                return False
            continue
        inv = 1.0 / direction[axis]
        t1 = (minimum[axis] - start[axis]) * inv
        t2 = (maximum[axis] - start[axis]) * inv
        t_near = max(t_near, min(t1, t2))
        t_far = min(t_far, max(t1, t2))
        if t_near > t_far:
            return False
    return True


def ray_aabb_distance(
    origin_xy: np.ndarray,
    direction_xy: np.ndarray,
    minimum_xy: np.ndarray,
    maximum_xy: np.ndarray,
) -> float | None:
    origin = np.asarray(origin_xy, dtype=float)
    direction = np.asarray(direction_xy, dtype=float)
    minimum = np.asarray(minimum_xy, dtype=float)
    maximum = np.asarray(maximum_xy, dtype=float)
    t_near = -float("inf")
    t_far = float("inf")
    for axis in range(2):
        if abs(direction[axis]) < 1e-12:
            if origin[axis] < minimum[axis] or origin[axis] > maximum[axis]:
                return None
            continue
        t1 = (minimum[axis] - origin[axis]) / direction[axis]
        t2 = (maximum[axis] - origin[axis]) / direction[axis]
        t_near = max(t_near, min(t1, t2))
        t_far = min(t_far, max(t1, t2))
        if t_near > t_far:
            return None
    if t_far < 0.0:
        return None
    return float(max(0.0, t_near))


def ray_domain_exit_distance(origin_xy: np.ndarray, direction_xy: np.ndarray, bounds_xy: np.ndarray) -> float:
    origin = np.asarray(origin_xy, dtype=float)
    direction = np.asarray(direction_xy, dtype=float)
    x0, x1, y0, y1 = np.asarray(bounds_xy, dtype=float)
    candidates: list[float] = []
    if direction[0] > 1e-12:
        candidates.append(float((x1 - origin[0]) / direction[0]))
    elif direction[0] < -1e-12:
        candidates.append(float((x0 - origin[0]) / direction[0]))
    if direction[1] > 1e-12:
        candidates.append(float((y1 - origin[1]) / direction[1]))
    elif direction[1] < -1e-12:
        candidates.append(float((y0 - origin[1]) / direction[1]))
    positive = [value for value in candidates if value >= 0.0]
    return min(positive, default=0.0)
