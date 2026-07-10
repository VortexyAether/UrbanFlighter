from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Building:
    center: np.ndarray  # x, y
    size: np.ndarray    # width_x, width_y
    height: float

    @property
    def min_xy(self) -> np.ndarray:
        return self.center - self.size / 2.0

    @property
    def max_xy(self) -> np.ndarray:
        return self.center + self.size / 2.0

    def contains(self, p: np.ndarray, margin: float = 0.0) -> bool:
        x, y, z = p[:3]
        mn = self.min_xy - margin
        mx = self.max_xy + margin
        return bool(mn[0] <= x <= mx[0] and mn[1] <= y <= mx[1] and 0.0 <= z <= self.height + margin)

    def segment_intersects(self, a: np.ndarray, b: np.ndarray, margin: float = 0.0) -> bool:
        mn = np.array([self.min_xy[0] - margin, self.min_xy[1] - margin, 0.0], dtype=float)
        mx = np.array([self.max_xy[0] + margin, self.max_xy[1] + margin, self.height + margin], dtype=float)
        start = np.asarray(a[:3], dtype=float)
        direction = np.asarray(b[:3], dtype=float) - start
        t_min = 0.0
        t_max = 1.0

        for axis in range(3):
            if abs(direction[axis]) < 1e-9:
                if start[axis] < mn[axis] or start[axis] > mx[axis]:
                    return False
                continue
            inv = 1.0 / direction[axis]
            t1 = (mn[axis] - start[axis]) * inv
            t2 = (mx[axis] - start[axis]) * inv
            near = min(t1, t2)
            far = max(t1, t2)
            t_min = max(t_min, near)
            t_max = min(t_max, far)
            if t_min > t_max:
                return False
        return True

    def distance_xy(self, xy: np.ndarray) -> float:
        d = np.maximum(np.maximum(self.min_xy - xy, xy - self.max_xy), 0.0)
        return float(np.linalg.norm(d))

    def to_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "size": self.size.tolist(),
            "height_m": float(self.height),
        }


class UrbanWorld:
    def __init__(self, bounds=(0.0, 100.0, 0.0, 100.0, 0.0, 60.0), buildings=None):
        self.bounds = np.array(bounds, dtype=float)
        self.buildings = list(buildings or [])

    @classmethod
    def toy_city(cls) -> "UrbanWorld":
        buildings = [
            Building(np.array([25.0, 35.0]), np.array([16.0, 28.0]), 34.0),
            Building(np.array([48.0, 52.0]), np.array([18.0, 18.0]), 45.0),
            Building(np.array([72.0, 32.0]), np.array([14.0, 24.0]), 28.0),
            Building(np.array([65.0, 72.0]), np.array([22.0, 14.0]), 38.0),
            Building(np.array([34.0, 76.0]), np.array([14.0, 18.0]), 24.0),
        ]
        return cls(buildings=buildings)

    def in_bounds(self, p: np.ndarray) -> bool:
        x0, x1, y0, y1, z0, z1 = self.bounds
        x, y, z = p[:3]
        return bool(x0 <= x <= x1 and y0 <= y <= y1 and z0 <= z <= z1)

    def collides(self, p: np.ndarray, margin: float = 0.0) -> bool:
        if not self.in_bounds(p):
            return True
        return any(b.contains(p, margin=margin) for b in self.buildings)

    def segment_hits_building(self, a: np.ndarray, b: np.ndarray, margin: float = 0.0) -> bool:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return any(building.segment_intersects(a, b, margin=margin) for building in self.buildings)

    def segment_collides(self, a: np.ndarray, b: np.ndarray, margin: float = 0.0, step_m: float = 1.0) -> bool:
        del step_m
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        return bool((not self.in_bounds(a)) or (not self.in_bounds(b)) or self.segment_hits_building(a, b, margin=margin))

    def segment_min_building_clearance(self, a: np.ndarray, b: np.ndarray, step_m: float = 1.0) -> float:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        length = float(np.linalg.norm(b[:3] - a[:3]))
        samples = max(2, int(np.ceil(length / max(step_m, 0.25))) + 1)
        return float(min(self.nearest_building_clearance(a + (b - a) * alpha) for alpha in np.linspace(0.0, 1.0, samples)))

    def nearest_building_clearance(self, p: np.ndarray) -> float:
        if not self.in_bounds(p):
            return 0.0
        if not self.buildings:
            return float("inf")
        xy = p[:2]
        clearances = [
            max(0.0, b.distance_xy(xy)) if p[2] <= b.height + 2.0 else float(np.hypot(b.distance_xy(xy), p[2] - b.height))
            for b in self.buildings
        ]
        return float(min(clearances))

    def obstacle_repulsion(self, p: np.ndarray, radius: float = 14.0) -> np.ndarray:
        xy = p[:2]
        force = np.zeros(3)
        for b in self.buildings:
            d = b.distance_xy(xy)
            if d < radius and p[2] < b.height + 10.0:
                v2 = xy - b.center
                n = np.linalg.norm(v2) + 1e-6
                force[:2] += (v2 / n) * ((radius - d) / radius) ** 2
                if p[2] < b.height + 4.0:
                    force[2] += 0.4 * (b.height + 4.0 - p[2]) / max(b.height, 1.0)
        return force

    def to_dict(self) -> dict:
        return {
            "bounds": self.bounds.tolist(),
            "building_count": len(self.buildings),
            "buildings": [building.to_dict() for building in self.buildings],
        }
