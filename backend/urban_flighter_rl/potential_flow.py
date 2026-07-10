from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

from .world import Building, UrbanWorld


@dataclass(frozen=True)
class PotentialFlowResult:
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    mask: np.ndarray
    streamlines: list[list[list[float]]]
    meta: dict


def wind_from_speed_direction(speed_mps: float, deg_from_north: float) -> np.ndarray:
    theta = np.deg2rad(float(deg_from_north))
    return np.array([-np.sin(theta), -np.cos(theta)], dtype=float) * float(speed_mps)


def _rasterize_world(world: UrbanWorld, x: np.ndarray, y: np.ndarray, altitude_m: float) -> np.ndarray:
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = np.zeros(xx.shape, dtype=bool)
    for b in world.buildings:
        if altitude_m > b.height:
            continue
        mn = b.min_xy
        mx = b.max_xy
        mask |= (xx >= mn[0]) & (xx <= mx[0]) & (yy >= mn[1]) & (yy <= mx[1])
    return mask


def _streamfunction_freestream(xx: np.ndarray, yy: np.ndarray, inlet_velocity: np.ndarray) -> np.ndarray:
    u, v = float(inlet_velocity[0]), float(inlet_velocity[1])
    return u * yy - v * xx


def solve_potential_flow_slice(
    world: UrbanWorld,
    inlet_velocity: tuple[float, float] | np.ndarray = (5.0, 0.0),
    altitude_m: float = 35.0,
    cell_size_m: float = 6.0,
    max_iter: int = 3500,
    tolerance: float = 1e-4,
    relaxation: float = 0.85,
    streamline_count: int = 24,
    streamline_steps: int = 420,
) -> PotentialFlowResult:
    """Solve a cheap 2D potential-flow slice around rectangular building masks.

    This is CFD-lite, not high-fidelity CFD. It solves a harmonic streamfunction
    with fixed far-field values and constant streamfunction inside each obstacle.
    That makes streamlines avoid buildings and gives a fast velocity field for
    visualization and route-cost experiments. It does not model turbulence,
    viscous separation, or true recirculation.
    """
    inlet = np.asarray(inlet_velocity, dtype=float)
    x0, x1, y0, y1, *_ = world.bounds
    x = np.arange(float(x0), float(x1) + cell_size_m * 0.5, float(cell_size_m))
    y = np.arange(float(y0), float(y1) + cell_size_m * 0.5, float(cell_size_m))
    nx, ny = len(x), len(y)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    psi = _streamfunction_freestream(xx, yy, inlet).astype(float)
    fixed = np.zeros((nx, ny), dtype=bool)
    fixed[0, :] = True
    fixed[-1, :] = True
    fixed[:, 0] = True
    fixed[:, -1] = True

    mask = _rasterize_world(world, x, y, altitude_m)
    fixed |= mask

    for b in world.buildings:
        if altitude_m > b.height:
            continue
        inside = mask & (xx >= b.min_xy[0]) & (xx <= b.max_xy[0]) & (yy >= b.min_xy[1]) & (yy <= b.max_xy[1])
        if inside.any():
            center_psi = float(_streamfunction_freestream(np.array([[b.center[0]]]), np.array([[b.center[1]]]), inlet)[0, 0])
            psi[inside] = center_psi

    fluid_update = ~fixed
    converged_iter = max_iter
    residual = float("inf")
    for it in range(1, max_iter + 1):
        old = psi.copy()
        avg = 0.25 * (psi[:-2, 1:-1] + psi[2:, 1:-1] + psi[1:-1, :-2] + psi[1:-1, 2:])
        upd = fluid_update[1:-1, 1:-1]
        psi[1:-1, 1:-1][upd] = (1.0 - relaxation) * psi[1:-1, 1:-1][upd] + relaxation * avg[upd]
        residual = float(np.max(np.abs(psi - old)))
        if residual < tolerance:
            converged_iter = it
            break

    dpsi_dx = np.gradient(psi, float(cell_size_m), axis=0)
    dpsi_dy = np.gradient(psi, float(cell_size_m), axis=1)
    ux = dpsi_dy
    uy = -dpsi_dx
    ux[mask] = 0.0
    uy[mask] = 0.0

    # Clamp extreme corner speeds. Potential flow has singular-ish corner acceleration;
    # this keeps the demo usable without pretending it is resolved CFD.
    speed = np.sqrt(ux * ux + uy * uy)
    inlet_speed = max(float(np.linalg.norm(inlet)), 1e-6)
    cap = inlet_speed * 2.8
    scale = np.minimum(1.0, cap / np.maximum(speed, 1e-6))
    ux *= scale
    uy *= scale

    streamlines = trace_streamlines(x, y, ux, uy, mask, inlet, count=streamline_count, max_steps=streamline_steps)
    meta = {
        "model": "potential-flow-cfd-lite-streamfunction",
        "not_full_cfd": True,
        "limitations": ["no_turbulence", "no_viscous_separation", "no_true_recirculation"],
        "altitude_m": float(altitude_m),
        "cell_size_m": float(cell_size_m),
        "nx": int(nx),
        "ny": int(ny),
        "inlet_ux_mps": float(inlet[0]),
        "inlet_uy_mps": float(inlet[1]),
        "inlet_speed_mps": float(np.linalg.norm(inlet)),
        "iterations": int(converged_iter),
        "residual": float(residual),
        "blocked_fraction": float(mask.mean()),
        "mean_speed_mps": float(np.sqrt(ux * ux + uy * uy).mean()),
        "max_speed_mps": float(np.sqrt(ux * ux + uy * uy).max()),
        "streamline_count": int(len(streamlines)),
    }
    return PotentialFlowResult(x=x, y=y, psi=psi, ux=ux, uy=uy, mask=mask, streamlines=streamlines, meta=meta)


def _sample_grid(x: np.ndarray, y: np.ndarray, field: np.ndarray, px: float, py: float) -> float:
    if px < x[0] or px > x[-1] or py < y[0] or py > y[-1]:
        return 0.0
    fx = (px - x[0]) / (x[1] - x[0])
    fy = (py - y[0]) / (y[1] - y[0])
    i = int(np.clip(np.floor(fx), 0, len(x) - 2))
    j = int(np.clip(np.floor(fy), 0, len(y) - 2))
    tx = fx - i
    ty = fy - j
    return float(
        (1 - tx) * (1 - ty) * field[i, j]
        + tx * (1 - ty) * field[i + 1, j]
        + (1 - tx) * ty * field[i, j + 1]
        + tx * ty * field[i + 1, j + 1]
    )


def _inside_mask(x: np.ndarray, y: np.ndarray, mask: np.ndarray, px: float, py: float) -> bool:
    if px < x[0] or px > x[-1] or py < y[0] or py > y[-1]:
        return True
    i = int(np.clip(round((px - x[0]) / (x[1] - x[0])), 0, len(x) - 1))
    j = int(np.clip(round((py - y[0]) / (y[1] - y[0])), 0, len(y) - 1))
    return bool(mask[i, j])


def trace_streamlines(
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
    inlet_velocity: np.ndarray,
    count: int = 24,
    max_steps: int = 420,
) -> list[list[list[float]]]:
    inlet = np.asarray(inlet_velocity, dtype=float)
    speed = max(float(np.linalg.norm(inlet)), 1e-6)
    direction = inlet / speed
    # Seed along the upwind domain edge.
    if abs(direction[0]) >= abs(direction[1]):
        sx = x[0] if direction[0] > 0 else x[-1]
        seeds = [(float(sx), float(v)) for v in np.linspace(y[2], y[-3], count)]
    else:
        sy = y[0] if direction[1] > 0 else y[-1]
        seeds = [(float(v), float(sy)) for v in np.linspace(x[2], x[-3], count)]

    ds = min(float(x[1] - x[0]), float(y[1] - y[0])) * 0.65
    lines: list[list[list[float]]] = []
    for px, py in seeds:
        line: list[list[float]] = []
        for _ in range(max_steps):
            if px < x[0] or px > x[-1] or py < y[0] or py > y[-1] or _inside_mask(x, y, mask, px, py):
                break
            vx = _sample_grid(x, y, ux, px, py)
            vy = _sample_grid(x, y, uy, px, py)
            mag = float(np.hypot(vx, vy))
            if mag < 1e-4:
                break
            line.append([round(float(px), 3), round(float(py), 3)])
            px += (vx / mag) * ds
            py += (vy / mag) * ds
        if len(line) >= 8:
            lines.append(line)
    return lines


def write_potential_flow_outputs(result: PotentialFlowResult, output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / "potential_flow_field.npz"
    stream_path = out / "potential_flow_streamlines.json"
    meta_path = out / "potential_flow_meta.json"
    np.savez_compressed(npz_path, x=result.x, y=result.y, psi=result.psi, ux=result.ux, uy=result.uy, mask=result.mask)
    stream_path.write_text(json.dumps({"streamlines": result.streamlines}, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(result.meta, indent=2), encoding="utf-8")
    return {"field_npz": str(npz_path), "streamlines_json": str(stream_path), "meta_json": str(meta_path)}
