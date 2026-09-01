from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg

from services.cfd import point_in_polygon
from urban_flighter_rl.potential_flow import solve_potential_flow_slice
from urban_flighter_rl.wind_corrections import apply_wall_damping_and_wake
from urban_flighter_rl.world import Building, UrbanWorld

try:
    from matplotlib.path import Path as MplPath
except Exception:  # pragma: no cover - optional fast rasterizer
    MplPath = None


def wind_dir_to_inlet_vector(speed_mps: float, deg_from_north: float) -> np.ndarray:
    """
    Convert meteorological direction to a world-space inlet velocity.
    The UI/domain stays aligned to cardinal axes. The solver changes only the inlet.
    - 0 deg: wind coming from north -> flow toward south
    - 90 deg: wind coming from east -> flow toward west
    """
    theta = np.deg2rad(float(deg_from_north))
    vx = -np.sin(theta)
    vy = -np.cos(theta)
    return np.array([vx, vy], dtype=np.float32) * float(speed_mps)


def rasterize_building_mask(
    buildings: list[dict[str, Any]],
    nx: int,
    ny: int,
    cell_size_m: float,
) -> np.ndarray:
    mask = np.zeros((nx, ny), dtype=bool)
    cx = nx / 2.0
    cy = ny / 2.0

    for building in buildings:
        footprint = np.asarray(building.get("footprint", []), dtype=np.float32)
        if footprint.ndim != 2 or footprint.shape[0] < 3 or footprint.shape[1] != 2:
            continue

        min_x, min_y = footprint.min(axis=0)
        max_x, max_y = footprint.max(axis=0)
        ix0 = max(0, int(np.floor(min_x / cell_size_m + cx)) - 1)
        ix1 = min(nx - 1, int(np.ceil(max_x / cell_size_m + cx)) + 1)
        iy0 = max(0, int(np.floor(min_y / cell_size_m + cy)) - 1)
        iy1 = min(ny - 1, int(np.ceil(max_y / cell_size_m + cy)) + 1)

        for ix in range(ix0, ix1 + 1):
            wx = (ix + 0.5 - cx) * cell_size_m
            for iy in range(iy0, iy1 + 1):
                wy = (iy + 0.5 - cy) * cell_size_m
                if point_in_polygon(wx, wy, footprint):
                    mask[ix, iy] = True

    return mask


def rasterize_building_mask_on_grid(
    buildings: list[dict[str, Any]],
    x: np.ndarray,
    y: np.ndarray,
    altitude_m: float,
) -> np.ndarray:
    """Rasterize real OSM polygon footprints on an existing solver grid.

    The B solver uses rectangular prisms internally for speed, but the field mask
    returned to the frontend should align with the displayed OSM geometry, not
    those internal bounding boxes.
    """
    mask = np.zeros((len(x), len(y)), dtype=bool)
    for building in buildings:
        if float(building.get("height", 0.0) or 0.0) < altitude_m:
            continue
        footprint = np.asarray(building.get("footprint", []), dtype=np.float32)
        if footprint.ndim != 2 or footprint.shape[0] < 3 or footprint.shape[1] != 2:
            continue
        min_x, min_y = footprint.min(axis=0)
        max_x, max_y = footprint.max(axis=0)
        ix0 = max(0, int(np.searchsorted(x, min_x, side="left")) - 1)
        ix1 = min(len(x) - 1, int(np.searchsorted(x, max_x, side="right")) + 1)
        iy0 = max(0, int(np.searchsorted(y, min_y, side="left")) - 1)
        iy1 = min(len(y) - 1, int(np.searchsorted(y, max_y, side="right")) + 1)
        _fill_polygon_mask(mask, x, y, ix0, ix1, iy0, iy1, footprint)
    return mask


def _fill_polygon_mask(
    mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    ix0: int,
    ix1: int,
    iy0: int,
    iy1: int,
    footprint: np.ndarray,
) -> None:
    xs = x[ix0:ix1 + 1]
    ys = y[iy0:iy1 + 1]
    if xs.size == 0 or ys.size == 0:
        return
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    if MplPath is not None:
        inside = MplPath(np.asarray(footprint, dtype=np.float64), closed=True).contains_points(
            np.column_stack([xx.ravel(), yy.ravel()])
        ).reshape(xx.shape)
        mask[ix0:ix1 + 1, iy0:iy1 + 1] |= inside
        return
    for i, wx in enumerate(xs):
        for j, wy in enumerate(ys):
            if point_in_polygon(float(wx), float(wy), footprint):
                mask[ix0 + i, iy0 + j] = True


def build_wake_and_deflection_sources(
    buildings: list[dict[str, Any]],
    xx: np.ndarray,
    yy: np.ndarray,
    wind_hat: np.ndarray,
    cross_hat: np.ndarray,
    cell_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    shelter = np.zeros_like(xx, dtype=np.float32)
    deflection = np.zeros_like(xx, dtype=np.float32)

    for building in buildings:
        footprint = np.asarray(building.get("footprint", []), dtype=np.float32)
        if footprint.ndim != 2 or footprint.shape[0] < 3 or footprint.shape[1] != 2:
            continue

        center = footprint.mean(axis=0)
        rel_x = xx - float(center[0])
        rel_y = yy - float(center[1])
        along = rel_x * wind_hat[0] + rel_y * wind_hat[1]
        cross = rel_x * cross_hat[0] + rel_y * cross_hat[1]

        span = max(
            float(np.ptp(footprint[:, 0])),
            float(np.ptp(footprint[:, 1])),
            float(cell_size_m * 2.0),
        )

        wake_width = span * 0.9 + np.maximum(along, 0.0) * 0.22 + cell_size_m * 2.0
        wake_decay = np.exp(-np.maximum(along, 0.0) / max(span * 4.5, cell_size_m * 8.0))
        wake_profile = np.exp(-np.square(cross / np.maximum(wake_width, cell_size_m)))
        shelter += (along > 0.0).astype(np.float32) * wake_decay * wake_profile * 2.2

        side_band = np.exp(-np.abs(cross) / max(span * 0.8, cell_size_m * 2.0))
        along_falloff = np.exp(-np.abs(along) / max(span * 1.6, cell_size_m * 3.0))
        deflection += side_band * along_falloff * np.sign(cross) * 0.9

    return shelter, deflection


def solve_screened_field_cg(
    mask: np.ndarray,
    rhs: np.ndarray,
    dirichlet: np.ndarray,
    cell_size_m: float,
    reaction: np.ndarray,
) -> tuple[np.ndarray, int]:
    nx, ny = mask.shape
    total = nx * ny
    matrix = lil_matrix((total, total), dtype=np.float32)
    vector = np.zeros(total, dtype=np.float32)
    h2 = float(cell_size_m * cell_size_m)
    index = lambda ix, iy: ix * ny + iy

    for ix in range(nx):
        for iy in range(ny):
            row = index(ix, iy)
            is_boundary = ix == 0 or iy == 0 or ix == nx - 1 or iy == ny - 1
            if mask[ix, iy] or is_boundary:
                matrix[row, row] = 1.0
                vector[row] = float(dirichlet[ix, iy])
                continue

            diag = 4.0 + reaction[ix, iy] * h2
            matrix[row, row] = diag
            matrix[row, index(ix - 1, iy)] = -1.0
            matrix[row, index(ix + 1, iy)] = -1.0
            matrix[row, index(ix, iy - 1)] = -1.0
            matrix[row, index(ix, iy + 1)] = -1.0
            vector[row] = float(rhs[ix, iy] * h2)

    solution, info = cg(matrix.tocsr(), vector, rtol=1e-4, atol=1e-6, maxiter=max(800, total // 2))
    if info < 0:
        raise RuntimeError("cg solver failed")
    field = solution.reshape(nx, ny).astype(np.float32)
    field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)
    field[mask] = 0.0
    return field, int(info)


def compute_time_averaged_flow_2d(
    buildings: list[dict[str, Any]],
    inlet_velocity: np.ndarray,
    radius_m: float,
    cell_size_m: float,
) -> dict[str, Any]:
    nx = max(32, int(np.ceil((2.0 * radius_m) / cell_size_m)))
    ny = nx
    if nx % 2 != 0:
        nx += 1
        ny += 1

    mask = rasterize_building_mask(buildings, nx, ny, cell_size_m)
    ux = np.full((nx, ny), float(inlet_velocity[0]), dtype=np.float32)
    uy = np.full((nx, ny), float(inlet_velocity[1]), dtype=np.float32)

    xs = (np.arange(nx, dtype=np.float32) + 0.5 - nx / 2.0) * float(cell_size_m)
    ys = (np.arange(ny, dtype=np.float32) + 0.5 - ny / 2.0) * float(cell_size_m)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    speed = float(np.linalg.norm(inlet_velocity))
    solver_info_u = 0
    solver_info_v = 0
    if speed > 1e-4 and buildings:
        wind_hat = inlet_velocity / speed
        cross_hat = np.array([-wind_hat[1], wind_hat[0]], dtype=np.float32)
        shelter, deflection_source = build_wake_and_deflection_sources(buildings, xx, yy, wind_hat, cross_hat, cell_size_m)
        fluid = ~mask
        dist_cells = distance_transform_edt(fluid).astype(np.float32)
        dist_m = dist_cells * float(cell_size_m)
        near_wall = np.exp(-dist_m / max(18.0, cell_size_m * 2.0))
        reaction = 0.018 + 0.22 * near_wall + 0.11 * shelter

        rhs_u = np.full((nx, ny), float(inlet_velocity[0]) * 0.018, dtype=np.float32)
        rhs_v = np.full((nx, ny), float(inlet_velocity[1]) * 0.018, dtype=np.float32)
        rhs_u += cross_hat[0] * deflection_source * speed * 0.012
        rhs_v += cross_hat[1] * deflection_source * speed * 0.012

        dirichlet_u = np.full((nx, ny), float(inlet_velocity[0]), dtype=np.float32)
        dirichlet_v = np.full((nx, ny), float(inlet_velocity[1]), dtype=np.float32)
        dirichlet_u[mask] = 0.0
        dirichlet_v[mask] = 0.0

        ux, solver_info_u = solve_screened_field_cg(mask, rhs_u, dirichlet_u, cell_size_m, reaction)
        uy, solver_info_v = solve_screened_field_cg(mask, rhs_v, dirichlet_v, cell_size_m, reaction)
    elif speed > 1e-4:
        reaction = np.full((nx, ny), 0.018, dtype=np.float32)
        dirichlet_u = np.full((nx, ny), float(inlet_velocity[0]), dtype=np.float32)
        dirichlet_v = np.full((nx, ny), float(inlet_velocity[1]), dtype=np.float32)
        rhs_u = np.full((nx, ny), float(inlet_velocity[0]) * 0.018, dtype=np.float32)
        rhs_v = np.full((nx, ny), float(inlet_velocity[1]) * 0.018, dtype=np.float32)
        ux, solver_info_u = solve_screened_field_cg(mask, rhs_u, dirichlet_u, cell_size_m, reaction)
        uy, solver_info_v = solve_screened_field_cg(mask, rhs_v, dirichlet_v, cell_size_m, reaction)

    ux[mask] = 0.0
    uy[mask] = 0.0

    sigma = max(0.5, 3.0 / max(cell_size_m, 1.0))
    ux = gaussian_filter(ux, sigma=sigma, mode="nearest")
    uy = gaussian_filter(uy, sigma=sigma, mode="nearest")
    ux = np.nan_to_num(ux, nan=0.0, posinf=0.0, neginf=0.0)
    uy = np.nan_to_num(uy, nan=0.0, posinf=0.0, neginf=0.0)
    ux[mask] = 0.0
    uy[mask] = 0.0

    speed_grid = np.sqrt(ux * ux + uy * uy)
    return {
        "nx": int(nx),
        "ny": int(ny),
        "cell_size_m": float(cell_size_m),
        "bounds": {
            "min_x": float(-nx * cell_size_m * 0.5),
            "max_x": float(nx * cell_size_m * 0.5),
            "min_y": float(-ny * cell_size_m * 0.5),
            "max_y": float(ny * cell_size_m * 0.5),
        },
        "ux": ux.ravel(order="C").tolist(),
        "uy": uy.ravel(order="C").tolist(),
        "mask": mask.astype(np.uint8).ravel(order="C").tolist(),
        "stats": {
            "mean_speed_mps": float(speed_grid.mean()),
            "max_speed_mps": float(speed_grid.max()),
            "blocked_fraction": float(mask.mean()),
            "cg_info_u": int(solver_info_u),
            "cg_info_v": int(solver_info_v),
        },
    }


def _streamfunction_freestream(xx: np.ndarray, yy: np.ndarray, inlet_velocity: np.ndarray) -> np.ndarray:
    u, v = float(inlet_velocity[0]), float(inlet_velocity[1])
    return u * yy - v * xx


def _apply_urban_look_corrections(
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    inlet_velocity: np.ndarray,
    cell_size_m: float,
    buildings: list[dict[str, Any]],
    altitude_m: float,
    dist_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Cheap urban-looking corrections on top of potential flow.

    Not Navier-Stokes. Adds street-canyon speed-up, a longer wake deficit,
    a near-wake reverse bubble, and a weak pair of trailing swirls so the
    field reads like an urban CFD plot instead of a smooth irrotational wrap.
    """
    speed0 = max(float(np.linalg.norm(inlet_velocity)), 1e-6)
    wind_hat = inlet_velocity / speed0
    cross_hat = np.array([-float(wind_hat[1]), float(wind_hat[0])], dtype=np.float32)
    cell = max(float(cell_size_m), 1e-3)

    canyon = np.exp(-np.clip(dist_m - cell, 0.0, None) / 14.0)
    canyon *= (dist_m > cell * 0.55).astype(np.float32)
    boost = 1.0 + 0.34 * canyon
    ux = ux * boost
    uy = uy * boost

    deficit = np.zeros_like(ux, dtype=np.float32)
    recirc = np.zeros_like(ux, dtype=np.float32)
    swirl = np.zeros_like(ux, dtype=np.float32)
    for building in buildings:
        if float(building.get("height", 0.0) or 0.0) < altitude_m:
            continue
        center, span = _building_center_and_span(building, cell)
        rel_x = xx - float(center[0])
        rel_y = yy - float(center[1])
        along = rel_x * wind_hat[0] + rel_y * wind_hat[1]
        cross = rel_x * cross_hat[0] + rel_y * cross_hat[1]
        wake_len = max(span * 4.4, cell * 8.0)
        wake_width = span * 0.62 + np.maximum(along, 0.0) * 0.16 + cell
        envelope = (
            (along > 0.0).astype(np.float32)
            * np.exp(-np.maximum(along, 0.0) / wake_len)
            * np.exp(-np.square(cross / np.maximum(wake_width, cell)))
        )
        deficit = np.maximum(deficit, envelope)
        bubble = (
            ((along > 0.12 * span) & (along < 1.45 * span)).astype(np.float32)
            * np.exp(-np.square((along - 0.55 * span) / max(0.48 * span, cell)))
            * np.exp(-np.square(cross / max(0.42 * span, cell)))
        )
        recirc = np.maximum(recirc, bubble)
        swirl += bubble * np.sign(cross + 1e-6) * 0.20 * speed0

    ux *= 1.0 - 0.70 * np.clip(deficit, 0.0, 1.0)
    uy *= 1.0 - 0.70 * np.clip(deficit, 0.0, 1.0)

    streamwise = ux * wind_hat[0] + uy * wind_hat[1]
    lateral = ux * cross_hat[0] + uy * cross_hat[1]
    streamwise = streamwise * (1.0 - 1.05 * recirc) - 0.30 * recirc * speed0
    lateral = lateral + swirl
    ux = streamwise * wind_hat[0] + lateral * cross_hat[0]
    uy = streamwise * wind_hat[1] + lateral * cross_hat[1]

    ux_s = gaussian_filter(ux, sigma=0.70, mode="nearest")
    uy_s = gaussian_filter(uy, sigma=0.70, mode="nearest")
    fluid = ~mask
    ux = np.where(fluid, ux_s, 0.0).astype(np.float32)
    uy = np.where(fluid, uy_s, 0.0).astype(np.float32)
    speed = np.sqrt(ux * ux + uy * uy)
    cap = speed0 * 3.4
    scale = np.minimum(1.0, cap / np.maximum(speed, 1e-6))
    ux *= scale
    uy *= scale
    ux, uy = _enforce_impermeable_slip(ux, uy, mask, dist_m, cell)
    return ux, uy, {
        "wake_strength": 0.70,
        "recirculation_strength": 0.30,
        "canyon_boost": 0.34,
        "wall_condition": "impermeable_slip",
    }


def _enforce_impermeable_slip(
    ux: np.ndarray,
    uy: np.ndarray,
    mask: np.ndarray,
    dist_m: np.ndarray,
    cell_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Slip wall: remove the wall-normal flux so air cannot cross a facade.

    ``dist_m`` is distance to the nearest solid, so its gradient points from the
    building into the fluid. Any velocity component toward the wall is deleted:
    ``u <- u - n min(u·n, 0)``. Inside solids, ``u = 0``.
    This is not no-slip Navier--Stokes; it is an impermeable boundary on the
    existing CFD-lite field.
    """
    cell = max(float(cell_size_m), 1e-3)
    nx = np.gradient(dist_m, cell, axis=0)
    ny = np.gradient(dist_m, cell, axis=1)
    nlen = np.sqrt(nx * nx + ny * ny)
    valid = (nlen > 1e-6) & (~mask)
    nx_hat = np.zeros_like(ux)
    ny_hat = np.zeros_like(uy)
    nx_hat[valid] = nx[valid] / nlen[valid]
    ny_hat[valid] = ny[valid] / nlen[valid]

    normal = ux * nx_hat + uy * ny_hat
    into_wall = np.minimum(normal, 0.0)
    ux = ux - into_wall * nx_hat
    uy = uy - into_wall * ny_hat

    near = valid & (dist_m < cell * 1.15)
    normal_near = ux * nx_hat + uy * ny_hat
    ux = np.where(near, ux - normal_near * nx_hat, ux)
    uy = np.where(near, uy - normal_near * ny_hat, uy)

    ux = np.where(mask, 0.0, ux).astype(np.float32)
    uy = np.where(mask, 0.0, uy).astype(np.float32)
    return ux, uy


def _building_center_and_span(building: dict[str, Any], cell_size_m: float) -> tuple[np.ndarray, float]:
    footprint = np.asarray(building.get("footprint", []), dtype=np.float32)
    if footprint.ndim != 2 or footprint.shape[0] < 3:
        return np.zeros(2, dtype=np.float32), float(cell_size_m * 2.0)
    center = footprint[:-1].mean(axis=0) if np.allclose(footprint[0], footprint[-1]) and len(footprint) > 3 else footprint.mean(axis=0)
    span = max(float(np.ptp(footprint[:, 0])), float(np.ptp(footprint[:, 1])), float(cell_size_m * 2.0))
    return center.astype(np.float32), span


def solve_polygon_potential_flow_b(
    buildings: list[dict[str, Any]],
    inlet_velocity: np.ndarray,
    radius_m: float,
    cell_size_m: float,
    altitude_m: float,
    max_iter: int = 3500,
    tolerance: float = 1e-4,
    relaxation: float = 0.85,
) -> dict[str, Any]:
    """CFD-lite B streamfunction solve on actual OSM polygon footprints.

    Earlier B used axis-aligned bounding boxes internally. That was fast, but
    diagonal buildings became x/y-aligned rectangles in the flow model. This
    version keeps the cheap potential-flow approximation but rasterizes the real
    polygon footprint on the solver grid.
    """
    x = np.arange(-float(radius_m), float(radius_m) + cell_size_m * 0.5, float(cell_size_m), dtype=np.float32)
    y = np.arange(-float(radius_m), float(radius_m) + cell_size_m * 0.5, float(cell_size_m), dtype=np.float32)
    nx, ny = len(x), len(y)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = rasterize_building_mask_on_grid(buildings, x, y, altitude_m)

    psi = _streamfunction_freestream(xx, yy, inlet_velocity).astype(np.float64)
    fixed = np.zeros((nx, ny), dtype=bool)
    fixed[0, :] = True
    fixed[-1, :] = True
    fixed[:, 0] = True
    fixed[:, -1] = True
    fixed |= mask

    # Each obstacle gets a constant streamfunction near its actual centroid.
    # This enforces approximate no-through-flow without replacing the footprint
    # with an axis-aligned rectangle.
    for building in buildings:
        if float(building.get("height", 0.0) or 0.0) < altitude_m:
            continue
        footprint = np.asarray(building.get("footprint", []), dtype=np.float32)
        if footprint.ndim != 2 or footprint.shape[0] < 3 or footprint.shape[1] != 2:
            continue
        center, _ = _building_center_and_span(building, cell_size_m)
        min_x, min_y = footprint.min(axis=0)
        max_x, max_y = footprint.max(axis=0)
        ix0 = max(0, int(np.searchsorted(x, min_x, side="left")) - 1)
        ix1 = min(nx - 1, int(np.searchsorted(x, max_x, side="right")) + 1)
        iy0 = max(0, int(np.searchsorted(y, min_y, side="left")) - 1)
        iy1 = min(ny - 1, int(np.searchsorted(y, max_y, side="right")) + 1)
        local = np.zeros((nx, ny), dtype=bool)
        local[ix0:ix1 + 1, iy0:iy1 + 1] = mask[ix0:ix1 + 1, iy0:iy1 + 1]
        if local.any():
            center_psi = float(_streamfunction_freestream(np.array([[center[0]]]), np.array([[center[1]]]), inlet_velocity)[0, 0])
            psi[local] = center_psi

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
    ux = dpsi_dy.astype(np.float32)
    uy = (-dpsi_dx).astype(np.float32)
    ux[mask] = 0.0
    uy[mask] = 0.0

    speed0 = max(float(np.linalg.norm(inlet_velocity)), 1e-6)
    speed = np.sqrt(ux * ux + uy * uy)
    cap = speed0 * 3.2
    scale = np.minimum(1.0, cap / np.maximum(speed, 1e-6))
    ux *= scale
    uy *= scale

    dist_m = distance_transform_edt(~mask).astype(np.float32) * float(cell_size_m)
    # Near-wall-only damping: far-field stays potential-flow, walls still slow.
    wall_length_m = max(6.0, float(cell_size_m))
    near_wall = np.exp(-dist_m / wall_length_m)
    damping = 1.0 - 0.45 * near_wall
    ux *= damping
    uy *= damping

    ux, uy, urban_meta = _apply_urban_look_corrections(
        ux=ux,
        uy=uy,
        mask=mask,
        xx=xx,
        yy=yy,
        inlet_velocity=inlet_velocity,
        cell_size_m=float(cell_size_m),
        buildings=buildings,
        altitude_m=float(altitude_m),
        dist_m=dist_m,
    )

    speed_grid = np.sqrt(ux * ux + uy * uy)
    return {
        "nx": int(nx),
        "ny": int(ny),
        "cell_size_m": float(cell_size_m),
        "bounds": {"min_x": float(x[0]), "max_x": float(x[-1]), "min_y": float(y[0]), "max_y": float(y[-1])},
        "ux": ux.ravel(order="C").tolist(),
        "uy": uy.ravel(order="C").tolist(),
        "mask": mask.astype(np.uint8).ravel(order="C").tolist(),
        "stats": {
            "mean_speed_mps": float(speed_grid.mean()),
            "max_speed_mps": float(speed_grid.max()),
            "blocked_fraction": float(mask.mean()),
            "model": "polygon-potential-flow-cfd-lite-wall-damping-wake-correction",
            "iterations": int(converged_iter),
            "residual": float(residual),
            "altitude_m": float(altitude_m),
            "wall_damping_length_m": 6.0,
            "wall_damping_scope": "near_wall_only",
            "wake_strength": urban_meta["wake_strength"],
            "recirculation_strength": urban_meta["recirculation_strength"],
            "canyon_boost": urban_meta["canyon_boost"],
            "wall_condition": urban_meta.get("wall_condition", "impermeable_slip"),
            "internal_obstacle_geometry": "actual_osm_polygon_mask",
            "urban_look": "wake_recirc_canyon_corner",
        },
    }


def _buildings_to_bbox_world(buildings: list[dict[str, Any]], radius_m: float) -> UrbanWorld:
    """Convert frontend OSM polygons into rectangular prisms for the fast B solver."""
    world_buildings: list[Building] = []
    max_height = 180.0
    for building in buildings:
        footprint = np.asarray(building.get("footprint", []), dtype=np.float32)
        if footprint.ndim != 2 or footprint.shape[0] < 3 or footprint.shape[1] != 2:
            continue
        min_xy = footprint.min(axis=0)
        max_xy = footprint.max(axis=0)
        size = np.maximum(max_xy - min_xy, 2.0)
        center = 0.5 * (min_xy + max_xy)
        height = float(building.get("height", 25.0) or 25.0)
        max_height = max(max_height, height + 30.0)
        world_buildings.append(Building(center=center.astype(float), size=size.astype(float), height=height))
    return UrbanWorld(bounds=(-radius_m, radius_m, -radius_m, radius_m, 0.0, max_height), buildings=world_buildings)


def compute_cfd_lite_b_flow_2d(
    buildings: list[dict[str, Any]],
    inlet_velocity: np.ndarray,
    radius_m: float,
    cell_size_m: float,
    altitude_m: float = 35.0,
) -> dict[str, Any]:
    """Model B: potential-flow grid + wall damping + empirical wake correction."""
    return solve_polygon_potential_flow_b(
        buildings=buildings,
        inlet_velocity=np.asarray(inlet_velocity, dtype=np.float32),
        radius_m=radius_m,
        cell_size_m=cell_size_m,
        altitude_m=altitude_m,
        max_iter=3500,
        tolerance=1e-4,
    )
