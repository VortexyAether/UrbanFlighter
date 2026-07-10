from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np

from .world import UrbanWorld


@dataclass(frozen=True)
class PanelFlow2DResult:
    x: np.ndarray
    y: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    speed: np.ndarray
    mask: np.ndarray
    panels: dict[str, np.ndarray]
    sigma: np.ndarray
    streamlines: list[list[list[float]]]
    meta: dict


def _rect_panels_for_building(center: np.ndarray, size: np.ndarray, max_panel_len: float = 16.0):
    cx, cy = float(center[0]), float(center[1])
    sx, sy = float(size[0]), float(size[1])
    x0, x1 = cx - sx / 2.0, cx + sx / 2.0
    y0, y1 = cy - sy / 2.0, cy + sy / 2.0
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]  # ccw
    starts = []
    ends = []
    normals = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        ax, ay = a
        bx, by = b
        length = float(np.hypot(bx - ax, by - ay))
        nseg = max(1, int(np.ceil(length / max_panel_len)))
        tangent = np.array([bx - ax, by - ay], dtype=float) / max(length, 1e-9)
        # ccw polygon outward normal is right-hand normal.
        normal = np.array([tangent[1], -tangent[0]], dtype=float)
        for i in range(nseg):
            t0 = i / nseg
            t1 = (i + 1) / nseg
            starts.append([ax + (bx - ax) * t0, ay + (by - ay) * t0])
            ends.append([ax + (bx - ax) * t1, ay + (by - ay) * t1])
            normals.append(normal)
    return starts, ends, normals


def build_rectangle_panels(world: UrbanWorld, altitude_m: float = 35.0, max_panel_len: float = 16.0):
    starts = []
    ends = []
    normals = []
    for b in world.buildings:
        if altitude_m > b.height:
            continue
        s, e, n = _rect_panels_for_building(b.center, b.size, max_panel_len=max_panel_len)
        starts.extend(s)
        ends.extend(e)
        normals.extend(n)
    starts_a = np.asarray(starts, dtype=float)
    ends_a = np.asarray(ends, dtype=float)
    normals_a = np.asarray(normals, dtype=float)
    centers = 0.5 * (starts_a + ends_a)
    lengths = np.linalg.norm(ends_a - starts_a, axis=1)
    return {"starts": starts_a, "ends": ends_a, "centers": centers, "normals": normals_a, "lengths": lengths}


def _point_source_velocity(points: np.ndarray, centers: np.ndarray, strengths: np.ndarray, eps: float = 4.0):
    # strengths already include panel length as source flux.
    diff = points[:, None, :] - centers[None, :, :]
    r2 = np.sum(diff * diff, axis=2) + eps * eps
    coef = strengths[None, :] / (2.0 * np.pi * r2)
    vel = np.sum(coef[:, :, None] * diff, axis=1)
    return vel


def solve_panel_strengths(panels: dict[str, np.ndarray], inlet_velocity: np.ndarray, eps: float = 4.0):
    centers = panels["centers"]
    normals = panels["normals"]
    lengths = panels["lengths"]
    n = len(centers)
    if n == 0:
        return np.zeros(0, dtype=float)
    a = np.zeros((n, n), dtype=float)
    for j in range(n):
        unit_flux = np.zeros(n, dtype=float)
        unit_flux[j] = lengths[j]
        vel = _point_source_velocity(centers, centers, unit_flux, eps=eps)
        a[:, j] = np.sum(vel * normals, axis=1)
    # Self influence for source panel has a half-jump in normal velocity.
    a[np.diag_indices(n)] -= 0.5
    rhs = -normals @ np.asarray(inlet_velocity, dtype=float)
    reg = 1e-5 * np.eye(n)
    sigma = np.linalg.solve(a + reg, rhs)
    return sigma, a @ sigma - rhs


def _rasterize_mask(world: UrbanWorld, x: np.ndarray, y: np.ndarray, altitude_m: float):
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = np.zeros(xx.shape, dtype=bool)
    for b in world.buildings:
        if altitude_m > b.height:
            continue
        mn = b.min_xy
        mx = b.max_xy
        mask |= (xx >= mn[0]) & (xx <= mx[0]) & (yy >= mn[1]) & (yy <= mx[1])
    return mask


def compute_panel_flow_2d(
    world: UrbanWorld,
    inlet_velocity: tuple[float, float] | np.ndarray = (5.0, 0.0),
    altitude_m: float = 35.0,
    cell_size_m: float = 5.0,
    max_panel_len: float = 18.0,
    streamline_count: int = 28,
) -> PanelFlow2DResult:
    started = time.perf_counter()
    inlet = np.asarray(inlet_velocity, dtype=float)
    x0, x1, y0, y1, *_ = world.bounds
    x = np.arange(float(x0), float(x1) + cell_size_m * 0.5, float(cell_size_m))
    y = np.arange(float(y0), float(y1) + cell_size_m * 0.5, float(cell_size_m))
    panels = build_rectangle_panels(world, altitude_m=altitude_m, max_panel_len=max_panel_len)
    solve_started = time.perf_counter()
    sigma, algebraic_residual = solve_panel_strengths(panels, inlet, eps=max(2.5, cell_size_m * 0.8))
    solve_elapsed = time.perf_counter() - solve_started

    xx, yy = np.meshgrid(x, y, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    strengths = sigma * panels["lengths"] if len(sigma) else sigma
    vel = np.zeros((len(pts), 2), dtype=float)
    batch = 4096
    eval_started = time.perf_counter()
    for i in range(0, len(pts), batch):
        vel[i:i+batch] = inlet + _point_source_velocity(pts[i:i+batch], panels["centers"], strengths, eps=max(2.5, cell_size_m * 0.8))
    eval_elapsed = time.perf_counter() - eval_started
    ux = vel[:, 0].reshape(len(x), len(y))
    uy = vel[:, 1].reshape(len(x), len(y))
    mask = _rasterize_mask(world, x, y, altitude_m)
    ux[mask] = 0.0
    uy[mask] = 0.0
    speed = np.sqrt(ux * ux + uy * uy)
    cap = max(float(np.linalg.norm(inlet)), 1e-6) * 3.0
    scale = np.minimum(1.0, cap / np.maximum(speed, 1e-6))
    ux *= scale
    uy *= scale
    speed = np.sqrt(ux * ux + uy * uy)
    streamlines = trace_streamlines_2d(x, y, ux, uy, mask, inlet, count=streamline_count)
    total_elapsed = time.perf_counter() - started
    normal_residual = None
    if len(sigma):
        boundary_vel = inlet + _point_source_velocity(panels["centers"], panels["centers"], strengths, eps=max(2.5, cell_size_m * 0.8))
        normal_residual = np.abs(np.sum(boundary_vel * panels["normals"], axis=1))
    meta = {
        "model": "2d-source-panel-potential-guidance-field",
        "not_full_cfd": True,
        "altitude_m": float(altitude_m),
        "cell_size_m": float(cell_size_m),
        "panel_count": int(len(sigma)),
        "nx": int(len(x)),
        "ny": int(len(y)),
        "solve_elapsed_s": float(solve_elapsed),
        "eval_elapsed_s": float(eval_elapsed),
        "total_elapsed_s": float(total_elapsed),
        "streamline_count": int(len(streamlines)),
        "mean_speed_mps": float(speed.mean()),
        "max_speed_mps": float(speed.max()),
        "blocked_fraction": float(mask.mean()),
        "mean_boundary_normal_residual_mps": float(normal_residual.mean()) if normal_residual is not None else 0.0,
        "max_boundary_normal_residual_mps": float(normal_residual.max()) if normal_residual is not None else 0.0,
        "mean_panel_equation_residual_mps": float(np.abs(algebraic_residual).mean()) if len(sigma) else 0.0,
        "max_panel_equation_residual_mps": float(np.abs(algebraic_residual).max()) if len(sigma) else 0.0,
    }
    return PanelFlow2DResult(x, y, ux, uy, speed, mask, panels, sigma, streamlines, meta)


def _sample_grid(x, y, field, px, py):
    if px < x[0] or px > x[-1] or py < y[0] or py > y[-1]:
        return 0.0
    fx = (px - x[0]) / (x[1] - x[0])
    fy = (py - y[0]) / (y[1] - y[0])
    i = int(np.clip(np.floor(fx), 0, len(x) - 2)); tx = fx - i
    j = int(np.clip(np.floor(fy), 0, len(y) - 2)); ty = fy - j
    return float((1-tx)*(1-ty)*field[i,j] + tx*(1-ty)*field[i+1,j] + (1-tx)*ty*field[i,j+1] + tx*ty*field[i+1,j+1])


def _inside_mask(x, y, mask, px, py):
    if px < x[0] or px > x[-1] or py < y[0] or py > y[-1]:
        return True
    i = int(np.clip(round((px-x[0])/(x[1]-x[0])), 0, len(x)-1))
    j = int(np.clip(round((py-y[0])/(y[1]-y[0])), 0, len(y)-1))
    return bool(mask[i, j])


def trace_streamlines_2d(x, y, ux, uy, mask, inlet, count=28, max_steps=550):
    inlet = np.asarray(inlet, dtype=float)
    direction = inlet / max(float(np.linalg.norm(inlet)), 1e-6)
    sx = x[0] if direction[0] > 0 else x[-1]
    seeds = [(float(sx), float(v)) for v in np.linspace(y[2], y[-3], count)]
    ds = min(float(x[1]-x[0]), float(y[1]-y[0])) * 0.65
    lines = []
    for px, py in seeds:
        line = []
        for _ in range(max_steps):
            if _inside_mask(x, y, mask, px, py):
                break
            vx = _sample_grid(x, y, ux, px, py)
            vy = _sample_grid(x, y, uy, px, py)
            mag = float(np.hypot(vx, vy))
            if mag < 1e-4:
                break
            line.append([round(float(px), 3), round(float(py), 3)])
            px += vx / mag * ds
            py += vy / mag * ds
        if len(line) >= 8:
            lines.append(line)
    return lines


def write_panel_flow_outputs(result: PanelFlow2DResult, output_dir: str | Path):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    field_path = out / "panel_flow_field.npz"
    stream_path = out / "panel_flow_streamlines.json"
    meta_path = out / "panel_flow_meta.json"
    np.savez_compressed(field_path, x=result.x, y=result.y, ux=result.ux, uy=result.uy, speed=result.speed, mask=result.mask, panel_centers=result.panels["centers"], panel_normals=result.panels["normals"], sigma=result.sigma)
    stream_path.write_text(json.dumps({"streamlines": result.streamlines}, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(result.meta, indent=2), encoding="utf-8")
    return {"field_npz": str(field_path), "streamlines_json": str(stream_path), "meta_json": str(meta_path)}
