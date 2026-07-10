from __future__ import annotations

import numpy as np


class UrbanWindField:
    """Heuristic 3D urban wind.

    Base wind + building wake/deflection. Not CFD. It is the MVP wind model
    used to expose RL/path planners to spatially varying local wind.
    """

    def __init__(self, world, base_wind=(5.0, 1.2, 0.0)):
        self.world = world
        self.base_wind = np.array(base_wind, dtype=float)
        norm = np.linalg.norm(self.base_wind[:2]) + 1e-9
        self.wind_dir_xy = self.base_wind[:2] / norm
        self.cross_xy = np.array([-self.wind_dir_xy[1], self.wind_dir_xy[0]])

    def at(self, p: np.ndarray, t: float = 0.0) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        w = self.base_wind.copy()
        w[:2] *= 1.0 + 0.08 * np.sin(0.15 * t + 0.05 * p[0])
        w[2] += 0.15 * np.sin(0.2 * t + 0.07 * p[1])

        for b in self.world.buildings:
            rel_xy = p[:2] - b.center
            downwind = float(np.dot(rel_xy, self.wind_dir_xy))
            cross = float(np.dot(rel_xy, self.cross_xy))
            z_factor = np.exp(-max(p[2] - b.height, 0.0) / 18.0) if p[2] >= 0 else 0.0

            # Wake: reduced speed behind building, elongated downwind.
            if 0.0 < downwind < 38.0 and abs(cross) < (max(b.size) * 0.9 + 10.0) and p[2] < b.height + 18.0:
                wake_strength = np.exp(-downwind / 24.0) * np.exp(-(cross / (max(b.size) * 0.7 + 8.0)) ** 2) * z_factor
                w[:2] -= self.wind_dir_xy * np.linalg.norm(self.base_wind[:2]) * 0.65 * wake_strength
                w[2] += 0.8 * wake_strength  # rooftop/updraft turbulence proxy

            # Side deflection near building front/sides.
            near = np.linalg.norm(rel_xy / (b.size + 1e-6))
            if near < 1.8 and p[2] < b.height + 12.0:
                side = np.sign(cross) if abs(cross) > 1e-6 else 1.0
                deflect = np.exp(-near) * z_factor
                w[:2] += self.cross_xy * side * 2.0 * deflect
                w[2] += 0.4 * np.exp(-abs(p[2] - b.height) / 10.0) * deflect
        return w

    def to_dict(self) -> dict:
        return {
            "kind": "heuristic_urban_wake_deflection",
            "base_wind_mps": self.base_wind.tolist(),
            "building_coupled": True,
            "cfd_claim": "not full CFD; RL-ready deterministic wind proxy",
        }
