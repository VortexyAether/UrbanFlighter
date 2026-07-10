from __future__ import annotations

import numpy as np


def default_start(world) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = world.bounds
    z = min(max(18.0, z0 + 12.0), z1 - 10.0)
    return np.array([x0 + 0.12 * (x1 - x0), y0 + 0.15 * (y1 - y0), z], dtype=float)


def default_goal(world) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = world.bounds
    z = min(max(22.0, z0 + 14.0), z1 - 8.0)
    return np.array([x0 + 0.88 * (x1 - x0), y0 + 0.85 * (y1 - y0), z], dtype=float)


def random_mission(world, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    min_span = 0.35 * float(np.linalg.norm(world.bounds[[1, 3, 5]] - world.bounds[[0, 2, 4]]))
    start = sample_free_point(world, rng)
    goal = sample_free_point(world, rng)
    for _ in range(200):
        if float(np.linalg.norm(goal - start)) >= min_span:
            break
        goal = sample_free_point(world, rng)
    return start, goal


def sample_free_point(world, rng: np.random.Generator) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = world.bounds
    max_building_h = max((b.height for b in world.buildings), default=0.0)
    min_z = min(max(z0 + 12.0, max_building_h + 8.0), z1 - 8.0)
    for _ in range(200):
        p = np.array(
            [
                rng.uniform(x0 + 6.0, x1 - 6.0),
                rng.uniform(y0 + 6.0, y1 - 6.0),
                rng.uniform(min_z, z1 - 6.0),
            ],
            dtype=float,
        )
        if not world.collides(p, margin=3.0):
            return p
    return default_start(world)
