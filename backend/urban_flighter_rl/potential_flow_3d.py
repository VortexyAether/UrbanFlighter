from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np

from .world import UrbanWorld


@dataclass(frozen=True)
class PotentialFlow3DResult:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    phi: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    uz: np.ndarray
    solid: np.ndarray
    streamlines: list[list[list[float]]]
    meta: dict


def _make_grid(world: UrbanWorld, nx: int, ny: int, nz: int):
    x0, x1, y0, y1, z0, z1 = world.bounds
    x = np.linspace(float(x0), float(x1), nx)
    y = np.linspace(float(y0), float(y1), ny)
    z = np.linspace(float(z0), float(z1), nz)
    return x, y, z


def _voxelize(world: UrbanWorld, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    solid = np.zeros(xx.shape, dtype=bool)
    solid_phi = np.zeros(xx.shape, dtype=float)
    for b in world.buildings:
        mn = b.min_xy
        mx = b.max_xy
        inside = (xx >= mn[0]) & (xx <= mx[0]) & (yy >= mn[1]) & (yy <= mx[1]) & (zz >= 0.0) & (zz <= b.height)
        solid |= inside
        # Approximate no-through obstacle by making each building an equipotential body.
        # Crude, but it creates a genuine 3D perturbation and vertical roof bypass flow.
        solid_phi[inside] = np.nan  # filled after inlet is known
    return solid, solid_phi


def solve_potential_flow_3d(
    world: UrbanWorld,
    inlet_velocity: tuple[float, float, float] | np.ndarray = (5.0, 0.0, 0.0),
    nx: int = 48,
    ny: int = 48,
    nz: int = 24,
    max_iter: int = 1200,
    tolerance: float = 1e-4,
    relaxation: float = 0.9,
    streamline_count_y: int = 7,
    streamline_count_z: int = 5,
    streamline_steps: int = 260,
) -> PotentialFlow3DResult:
    """Approximate 3D potential-flow CFD-lite field.

    This solves a harmonic velocity potential on a voxel grid with far-field
    Dirichlet boundaries and building voxels held fixed to freestream potential.
    It is useful for fast 3D streamlines and u/v/w route-cost experiments, but is
    not high-fidelity CFD and not a rigorous immersed-boundary Laplace solver.
    """
    started = time.perf_counter()
    inlet = np.asarray(inlet_velocity, dtype=float)
    x, y, z = _make_grid(world, nx, ny, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    phi_free = inlet[0] * xx + inlet[1] * yy + inlet[2] * zz
    phi = phi_free.copy()
    solid, solid_phi = _voxelize(world, x, y, z)
    for b in world.buildings:
        mn = b.min_xy
        mx = b.max_xy
        inside = (xx >= mn[0]) & (xx <= mx[0]) & (yy >= mn[1]) & (yy <= mx[1]) & (zz >= 0.0) & (zz <= b.height)
        if inside.any():
            center_phi = float(inlet[0] * b.center[0] + inlet[1] * b.center[1] + inlet[2] * min(b.height * 0.5, z[-1]))
            solid_phi[inside] = center_phi
    phi[solid] = solid_phi[solid]

    fixed = solid.copy()
    fixed[0, :, :] = True
    fixed[-1, :, :] = True
    fixed[:, 0, :] = True
    fixed[:, -1, :] = True
    fixed[:, :, 0] = True
    fixed[:, :, -1] = True
    update = ~fixed

    residual = float("inf")
    iterations = max_iter
    solve_started = time.perf_counter()
    for it in range(1, max_iter + 1):
        old = phi.copy()
        avg = (
            phi[:-2, 1:-1, 1:-1]
            + phi[2:, 1:-1, 1:-1]
            + phi[1:-1, :-2, 1:-1]
            + phi[1:-1, 2:, 1:-1]
            + phi[1:-1, 1:-1, :-2]
            + phi[1:-1, 1:-1, 2:]
        ) / 6.0
        upd = update[1:-1, 1:-1, 1:-1]
        core = phi[1:-1, 1:-1, 1:-1]
        core[upd] = (1.0 - relaxation) * core[upd] + relaxation * avg[upd]
        phi[1:-1, 1:-1, 1:-1] = core
        phi[fixed & ~solid] = phi_free[fixed & ~solid]
        phi[solid] = solid_phi[solid]
        residual = float(np.max(np.abs(phi - old)))
        if residual < tolerance:
            iterations = it
            break
    solve_elapsed = time.perf_counter() - solve_started

    ux = np.gradient(phi, x, axis=0)
    uy = np.gradient(phi, y, axis=1)
    uz = np.gradient(phi, z, axis=2)
    ux[solid] = 0.0
    uy[solid] = 0.0
    uz[solid] = 0.0

    speed = np.sqrt(ux * ux + uy * uy + uz * uz)
    inlet_speed = max(float(np.linalg.norm(inlet)), 1e-6)
    cap = inlet_speed * 3.0
    scale = np.minimum(1.0, cap / np.maximum(speed, 1e-6))
    ux *= scale
    uy *= scale
    uz *= scale

    trace_started = time.perf_counter()
    streamlines = trace_streamlines_3d(x, y, z, ux, uy, uz, solid, inlet, streamline_count_y, streamline_count_z, streamline_steps)
    trace_elapsed = time.perf_counter() - trace_started
    total_elapsed = time.perf_counter() - started
    speed = np.sqrt(ux * ux + uy * uy + uz * uz)
    meta = {
        "model": "3d-potential-flow-cfd-lite-voxel-laplace",
        "not_full_cfd": True,
        "not_rigorous_immersed_boundary": True,
        "limitations": ["no_turbulence", "no_viscous_separation", "approximate_building_boundary"],
        "nx": int(nx),
        "ny": int(ny),
        "nz": int(nz),
        "cells": int(nx * ny * nz),
        "solid_fraction": float(solid.mean()),
        "inlet_ux_mps": float(inlet[0]),
        "inlet_uy_mps": float(inlet[1]),
        "inlet_uz_mps": float(inlet[2]),
        "inlet_speed_mps": float(inlet_speed),
        "iterations": int(iterations),
        "residual": float(residual),
        "solve_elapsed_s": float(solve_elapsed),
        "trace_elapsed_s": float(trace_elapsed),
        "total_elapsed_s": float(total_elapsed),
        "mean_speed_mps": float(speed.mean()),
        "max_speed_mps": float(speed.max()),
        "mean_abs_w_mps": float(np.abs(uz).mean()),
        "max_abs_w_mps": float(np.abs(uz).max()),
        "streamline_count": int(len(streamlines)),
    }
    return PotentialFlow3DResult(x, y, z, phi, ux, uy, uz, solid, streamlines, meta)


def _sample3(x, y, z, field, px, py, pz) -> float:
    if px < x[0] or px > x[-1] or py < y[0] or py > y[-1] or pz < z[0] or pz > z[-1]:
        return 0.0
    fx = (px - x[0]) / (x[1] - x[0])
    fy = (py - y[0]) / (y[1] - y[0])
    fz = (pz - z[0]) / (z[1] - z[0])
    i = int(np.clip(np.floor(fx), 0, len(x) - 2)); tx = fx - i
    j = int(np.clip(np.floor(fy), 0, len(y) - 2)); ty = fy - j
    k = int(np.clip(np.floor(fz), 0, len(z) - 2)); tz = fz - k
    c000 = field[i, j, k]; c100 = field[i+1, j, k]; c010 = field[i, j+1, k]; c110 = field[i+1, j+1, k]
    c001 = field[i, j, k+1]; c101 = field[i+1, j, k+1]; c011 = field[i, j+1, k+1]; c111 = field[i+1, j+1, k+1]
    c00 = c000*(1-tx)+c100*tx; c10 = c010*(1-tx)+c110*tx; c01 = c001*(1-tx)+c101*tx; c11 = c011*(1-tx)+c111*tx
    c0 = c00*(1-ty)+c10*ty; c1 = c01*(1-ty)+c11*ty
    return float(c0*(1-tz)+c1*tz)


def _inside_solid(x, y, z, solid, px, py, pz) -> bool:
    if px < x[0] or px > x[-1] or py < y[0] or py > y[-1] or pz < z[0] or pz > z[-1]:
        return True
    i = int(np.clip(round((px - x[0]) / (x[1] - x[0])), 0, len(x) - 1))
    j = int(np.clip(round((py - y[0]) / (y[1] - y[0])), 0, len(y) - 1))
    k = int(np.clip(round((pz - z[0]) / (z[1] - z[0])), 0, len(z) - 1))
    return bool(solid[i, j, k])


def trace_streamlines_3d(x, y, z, ux, uy, uz, solid, inlet, count_y=7, count_z=5, max_steps=260):
    inlet = np.asarray(inlet, dtype=float)
    direction = inlet / max(float(np.linalg.norm(inlet)), 1e-6)
    # Current demo assumes dominant x inflow; good for west/east wind.
    sx = float(x[0] if direction[0] > 0 else x[-1])
    ys = np.linspace(float(y[2]), float(y[-3]), count_y)
    zs = np.linspace(float(max(z[2], 15.0)), float(z[-3]), count_z)
    ds = min(float(x[1]-x[0]), float(y[1]-y[0]), float(z[1]-z[0])) * 0.7
    lines = []
    for sy in ys:
        for sz in zs:
            px, py, pz = sx, float(sy), float(sz)
            line = []
            for _ in range(max_steps):
                if _inside_solid(x, y, z, solid, px, py, pz):
                    break
                vx = _sample3(x, y, z, ux, px, py, pz)
                vy = _sample3(x, y, z, uy, px, py, pz)
                vz = _sample3(x, y, z, uz, px, py, pz)
                mag = float(np.sqrt(vx*vx + vy*vy + vz*vz))
                if mag < 1e-4:
                    break
                line.append([round(float(px), 3), round(float(py), 3), round(float(pz), 3)])
                px += vx / mag * ds
                py += vy / mag * ds
                pz += vz / mag * ds
            if len(line) >= 8:
                lines.append(line)
    return lines


def write_potential_flow_3d_outputs(result: PotentialFlow3DResult, output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / "potential_flow_3d_field.npz"
    stream_path = out / "potential_flow_3d_streamlines.json"
    meta_path = out / "potential_flow_3d_meta.json"
    np.savez_compressed(npz_path, x=result.x, y=result.y, z=result.z, phi=result.phi, ux=result.ux, uy=result.uy, uz=result.uz, solid=result.solid)
    stream_path.write_text(json.dumps({"streamlines": result.streamlines}, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(result.meta, indent=2), encoding="utf-8")
    return {"field_npz": str(npz_path), "streamlines_json": str(stream_path), "meta_json": str(meta_path)}
