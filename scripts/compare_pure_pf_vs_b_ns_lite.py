#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from services.flow_2d import (  # noqa: E402
    _building_center_and_span,
    _streamfunction_freestream,
    rasterize_building_mask_on_grid,
    wind_dir_to_inlet_vector,
)
from services.geometry import fetch_buildings  # noqa: E402

OUT = ROOT / "results/velocity_field_compare_20260714"
OUT.mkdir(parents=True, exist_ok=True)

LAT, LON = 37.497942, 127.027621
RADIUS = 400.0
CELL = 12.5
ALT = 35.0
INLET = wind_dir_to_inlet_vector(5.0, 0.0)


def solve_stages(buildings, inlet, radius_m, cell, altitude_m, max_iter=3500, tol=1e-4, relax=0.85):
    x = np.arange(-float(radius_m), float(radius_m) + cell * 0.5, float(cell), dtype=np.float32)
    y = np.arange(-float(radius_m), float(radius_m) + cell * 0.5, float(cell), dtype=np.float32)
    nx, ny = len(x), len(y)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = rasterize_building_mask_on_grid(buildings, x, y, altitude_m)

    psi = _streamfunction_freestream(xx, yy, inlet).astype(np.float64)
    fixed = np.zeros((nx, ny), dtype=bool)
    fixed[0, :] = True
    fixed[-1, :] = True
    fixed[:, 0] = True
    fixed[:, -1] = True
    fixed |= mask

    for building in buildings:
        if float(building.get("height", 0.0) or 0.0) < altitude_m:
            continue
        footprint = np.asarray(building.get("footprint", []), dtype=np.float32)
        if footprint.ndim != 2 or footprint.shape[0] < 3:
            continue
        center, _ = _building_center_and_span(building, cell)
        min_x, min_y = footprint.min(0)
        max_x, max_y = footprint.max(0)
        ix0 = max(0, int(np.searchsorted(x, min_x, side="left")) - 1)
        ix1 = min(nx - 1, int(np.searchsorted(x, max_x, side="right")) + 1)
        iy0 = max(0, int(np.searchsorted(y, min_y, side="left")) - 1)
        iy1 = min(ny - 1, int(np.searchsorted(y, max_y, side="right")) + 1)
        local = np.zeros((nx, ny), dtype=bool)
        local[ix0 : ix1 + 1, iy0 : iy1 + 1] = mask[ix0 : ix1 + 1, iy0 : iy1 + 1]
        if local.any():
            center_psi = float(
                _streamfunction_freestream(np.array([[center[0]]]), np.array([[center[1]]]), inlet)[0, 0]
            )
            psi[local] = center_psi

    fluid = ~fixed
    residual = np.inf
    it_done = max_iter
    for it in range(1, max_iter + 1):
        old = psi.copy()
        avg = 0.25 * (psi[:-2, 1:-1] + psi[2:, 1:-1] + psi[1:-1, :-2] + psi[1:-1, 2:])
        upd = fluid[1:-1, 1:-1]
        psi[1:-1, 1:-1][upd] = (1.0 - relax) * psi[1:-1, 1:-1][upd] + relax * avg[upd]
        residual = float(np.max(np.abs(psi - old)))
        if residual < tol:
            it_done = it
            break

    dpsi_dx = np.gradient(psi, float(cell), axis=0)
    dpsi_dy = np.gradient(psi, float(cell), axis=1)
    ux = dpsi_dy.astype(np.float32)
    uy = (-dpsi_dx).astype(np.float32)
    ux[mask] = 0.0
    uy[mask] = 0.0

    speed0 = max(float(np.linalg.norm(inlet)), 1e-6)
    speed = np.sqrt(ux * ux + uy * uy)
    cap = speed0 * 2.8
    scale = np.minimum(1.0, cap / np.maximum(speed, 1e-6))
    ux_a = ux * scale
    uy_a = uy * scale
    ux_a[mask] = 0.0
    uy_a[mask] = 0.0

    dist_m = distance_transform_edt(~mask).astype(np.float32) * float(cell)
    damping = 1.0 - np.exp(-dist_m / max(14.0, float(cell)))
    ux_b = ux_a * damping
    uy_b = uy_a * damping
    wind_hat = inlet / speed0
    cross_hat = np.array([-wind_hat[1], wind_hat[0]], dtype=np.float32)
    deficit = np.zeros_like(ux_b)
    for building in buildings:
        if float(building.get("height", 0.0) or 0.0) < altitude_m:
            continue
        center, span = _building_center_and_span(building, cell)
        rel_x = xx - float(center[0])
        rel_y = yy - float(center[1])
        along = rel_x * wind_hat[0] + rel_y * wind_hat[1]
        cross = rel_x * cross_hat[0] + rel_y * cross_hat[1]
        wake_len = max(span * 5.5, cell * 8.0)
        wake_width = span * 0.7 + np.maximum(along, 0.0) * 0.22 + cell
        local = (
            (along > 0.0).astype(np.float32)
            * np.exp(-np.maximum(along, 0.0) / wake_len)
            * np.exp(-np.square(cross / np.maximum(wake_width, cell)))
        )
        deficit = np.maximum(deficit, local)
    wake_factor = 1.0 - 0.55 * np.clip(deficit, 0.0, 1.0)
    ux_b *= wake_factor
    uy_b *= wake_factor
    ux_b[mask] = 0.0
    uy_b[mask] = 0.0

    # Proposed NS-lite empirical (not production): thinner near-wall-only damping,
    # stronger lee wake with lateral growth, weak reverse core.
    wall_len = max(6.0, float(cell) * 0.7)
    damp_c = 1.0 - np.exp(-dist_m / wall_len)
    near = np.clip(1.0 - dist_m / (wall_len * 3.0), 0.0, 1.0)
    ux_c = ux_a * (1.0 - near + near * damp_c)
    uy_c = uy_a * (1.0 - near + near * damp_c)

    deficit_c = np.zeros_like(ux_c)
    for building in buildings:
        if float(building.get("height", 0.0) or 0.0) < altitude_m:
            continue
        center, span = _building_center_and_span(building, cell)
        rel_x = xx - float(center[0])
        rel_y = yy - float(center[1])
        along = rel_x * wind_hat[0] + rel_y * wind_hat[1]
        cross = rel_x * cross_hat[0] + rel_y * cross_hat[1]
        wake_len = max(span * 7.0, cell * 10.0)
        wake_width = span * 0.55 + np.maximum(along, 0.0) * 0.35 + cell
        near_wake = (along > 0.0) & (along < span * 1.2)
        gauss = np.exp(-np.square(cross / np.maximum(wake_width, cell)))
        base = (along > 0.0).astype(np.float32) * np.exp(-np.maximum(along, 0.0) / wake_len) * gauss
        deficit_c = np.maximum(deficit_c, base * 0.85)
        reverse = near_wake.astype(np.float32) * gauss * np.exp(-np.maximum(along, 0.0) / (span * 0.8 + 1e-6))
        ux_c -= wind_hat[0] * speed0 * 0.22 * reverse
        uy_c -= wind_hat[1] * speed0 * 0.22 * reverse

    deficit_c = gaussian_filter(deficit_c, sigma=0.7)
    factor_c = 1.0 - 0.7 * np.clip(deficit_c, 0.0, 1.0)
    ux_c *= factor_c
    uy_c *= factor_c
    spd = np.sqrt(ux_c * ux_c + uy_c * uy_c)
    sc = np.minimum(1.0, (speed0 * 3.2) / np.maximum(spd, 1e-6))
    ux_c *= sc
    uy_c *= sc
    ux_c[mask] = 0.0
    uy_c[mask] = 0.0

    def pack(ux_i, uy_i, name):
        spd_i = np.sqrt(ux_i * ux_i + uy_i * uy_i)
        fluid_spd = spd_i[~mask]
        return {
            "name": name,
            "x": x,
            "y": y,
            "ux": ux_i,
            "uy": uy_i,
            "mask": mask,
            "mean": float(fluid_spd.mean()) if fluid_spd.size else 0.0,
            "max": float(fluid_spd.max()) if fluid_spd.size else 0.0,
            "iters": it_done,
            "residual": residual,
        }

    return {
        "A_pure_pf": pack(ux_a, uy_a, "A pure potential-flow (+cap)"),
        "B_prod": pack(ux_b, uy_b, "B production: PF + wall damping + wake"),
        "C_ns_lite": pack(ux_c, uy_c, "C proposed NS-lite empirical"),
        "inlet": np.asarray(inlet).tolist(),
    }


def panel(ax, st, title, vmax):
    x, y, ux, uy, mask = st["x"], st["y"], st["ux"], st["uy"], st["mask"]
    spd = np.sqrt(ux * ux + uy * uy).astype(float)
    spd[mask] = np.nan
    im = ax.imshow(
        spd.T,
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        cmap="turbo",
        vmin=0,
        vmax=vmax,
        aspect="equal",
        interpolation="nearest",
    )
    try:
        ax.contour(x, y, mask.T.astype(float), levels=[0.5], colors="k", linewidths=0.4, alpha=0.8)
    except Exception:
        pass
    skip = max(1, len(x) // 34)
    xx, yy = np.meshgrid(x[::skip], y[::skip], indexing="ij")
    uq = np.ma.array(ux[::skip, ::skip], mask=mask[::skip, ::skip])
    vq = np.ma.array(uy[::skip, ::skip], mask=mask[::skip, ::skip])
    ax.quiver(xx, yy, uq, vq, color="k", alpha=0.5, scale=vmax * 16, width=0.0022, pivot="mid")
    us = ux.astype(float).copy()
    vs = uy.astype(float).copy()
    us[mask] = 0.0
    vs[mask] = 0.0
    try:
        ax.streamplot(x, y, us.T, vs.T, color="white", density=1.25, linewidth=0.5, arrowsize=0.45, maxlength=3.5)
    except Exception:
        pass
    ax.set_title(f"{title}\nmean={st['mean']:.2f} max={st['max']:.2f} m/s", fontsize=10)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return im


def main() -> None:
    print("fetch buildings...")
    buildings = fetch_buildings(LAT, LON, RADIUS)
    print("n_buildings", len(buildings))
    print("solve...")
    stages = solve_stages(buildings, INLET.astype(np.float32), RADIUS, CELL, ALT)
    for key in ("A_pure_pf", "B_prod", "C_ns_lite"):
        st = stages[key]
        print(key, "mean", st["mean"], "max", st["max"])

    vmax = max(stages["A_pure_pf"]["max"], stages["B_prod"]["max"], stages["C_ns_lite"]["max"])
    vmax = float(min(vmax, 12.0))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))
    im = None
    titles = [
        ("A_pure_pf", "A. Pure potential-flow"),
        ("B_prod", "B. Production B (damp+wake)"),
        ("C_ns_lite", "C. Proposed NS-lite empirical"),
    ]
    for ax, (key, title) in zip(axes, titles):
        im = panel(ax, stages[key], title, vmax)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label="|U| [m/s]")
    fig.suptitle(
        "Gangnam CFD-lite · inlet 5 m/s from N · z~35 m mask · same OSM polygons\nNOT full Navier-Stokes",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 0.98, 0.92])
    p1 = OUT / "compare_purePF_vs_B_vs_NSlite_xy.png"
    fig.savefig(p1, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p1)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0))
    for ax, (key, title) in zip(
        axes,
        [("A_pure_pf", "A. Pure PF (slip-like)"), ("B_prod", "B. PF + wall absorb + wake")],
    ):
        im = panel(ax, stages[key], title, vmax)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02, label="|U| [m/s]")
    fig.suptitle("Why walls look absorbent: B multiplies u by (1-exp(-dist/14m)) then wake deficit", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.98, 0.93])
    p2 = OUT / "compare_purePF_vs_B_xy.png"
    fig.savefig(p2, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p2)

    A = stages["A_pure_pf"]
    B = stages["B_prod"]
    C = stages["C_ns_lite"]
    spdA = np.sqrt(A["ux"] ** 2 + A["uy"] ** 2)
    spdB = np.sqrt(B["ux"] ** 2 + B["uy"] ** 2)
    spdC = np.sqrt(C["ux"] ** 2 + C["uy"] ** 2)
    mask = A["mask"]
    x, y = A["x"], A["y"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, diff, title in [
        (axes[0], spdB - spdA, "delta speed B-A (damping+wake effect)"),
        (axes[1], spdC - spdA, "delta speed C-A (proposed NS-lite)"),
    ]:
        d = diff.astype(float)
        d[mask] = np.nan
        lim = float(np.nanpercentile(np.abs(d), 98))
        im = ax.imshow(
            d.T,
            origin="lower",
            extent=[x[0], x[-1], y[0], y[-1]],
            cmap="coolwarm",
            vmin=-lim,
            vmax=lim,
            aspect="equal",
        )
        ax.contour(x, y, mask.T.astype(float), levels=[0.5], colors="k", linewidths=0.4)
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        fig.colorbar(im, ax=ax, fraction=0.046, label="m/s")
    fig.suptitle("Negative near walls/wakes = slowed vs pure PF", fontsize=11)
    fig.tight_layout()
    p3 = OUT / "compare_speed_delta_vs_purePF.png"
    fig.savefig(p3, dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p3)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    d = np.linspace(0, 60, 300)
    near = np.clip(1.0 - d / (6.0 * 3.0), 0.0, 1.0)
    blend = 1.0 - near + near * (1.0 - np.exp(-d / 6.0))
    ax.plot(d, 1.0 - np.exp(-d / 14.0), label="B: multiply fluid by damp L=14m", lw=2.2)
    ax.plot(d, blend, label="C: damp only near wall (blend)", lw=2.2)
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("distance to wall [m]")
    ax.set_ylabel("speed retention factor")
    ax.set_title("Wall treatment: B absorbs near walls; C keeps outer slip-like PF")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    p4 = OUT / "wall_damping_profiles.png"
    fig.savefig(p4, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", p4)

    memo = OUT / "NS_like_improvement_plan.md"
    memo.write_text(
        f"""# CFD-lite to more NS-like improvement plan (Urban_Flighter)

Domain: Gangnam ({LAT}, {LON}), R={RADIUS} m, dx={CELL} m, inlet 5 m/s from N, mask z~{ALT} m.
Buildings: {len(buildings)}.

## What the plots show
- A pure PF: slip-like, channel speed-up, no wall absorption, no lee deficit.
- B production: PF core, then u *= 1-exp(-dist/14m), then wake deficit 0.55. Walls look absorbent.
- C prototype only: thinner near-wall-only damping, stronger/shorter wake + weak reverse core + wake blur. Still NOT NS.

Stats (fluid cells):
- A mean/max: {stages['A_pure_pf']['mean']:.3f} / {stages['A_pure_pf']['max']:.3f} m/s
- B mean/max: {stages['B_prod']['mean']:.3f} / {stages['B_prod']['max']:.3f} m/s
- C mean/max: {stages['C_ns_lite']['mean']:.3f} / {stages['C_ns_lite']['max']:.3f} m/s

## Why B is not NS
Potential flow cannot do viscous BL, separation bubbles, vortex shedding, or turbulence.
Wall damping is a drone-risk hack, not a boundary-layer model.

## Improvement ladder

### L1 keep PF, fix empirics (days)
1. Near-wall-only damping (C-style blend); restore channel jets.
2. Wake upgrade: lateral growth, near-wake reverse proxy, width/height scaling.
3. Better multi-building shelter merge (log-sum-exp / multiplicative), not only max.
4. Height slabs: scale wake/damping by z/H; log/power inlet shear.
5. UI toggle A/B/C with honest labels.

### L2 steady RANS-ish 2D/2.5D (1-2 weeks)
1. SIMPLE/PISO or projection RANS on OSM mask.
2. SA or k-epsilon + wall functions.
3. Coarse offline cache, serve as rans-lite endpoint.
4. Validate wake length / street speed ratio vs PF and B.

### L3 unsteady / 3D research
1. Keep AeroJAX / offline NS snapshots as high-fid path.
2. Live flight stays on horizontal B or RANS-lite; True3D is overlay or cached field.
3. LES too heavy for live browser loop.

### L4 ML residual later
1. PF features -> NS snapshot delta.
2. Trust map + safe fallback.

## Product recommendation
1. Research toggle A/B/C (default B for safety-biased flight until regression).
2. Implement L1 as Model B2.
3. Offline RANS-lite for Gangnam/Inha demos as Model R.
4. Never relabel PF/B/C as full NS.

## Do not
- Call wall damping a boundary layer.
- Call reverse-wake proxy vortex shedding.
- Replace B with C in flight without collision/energy tests.

## Plots
- {p1}
- {p2}
- {p3}
- {p4}
""",
        encoding="utf-8",
    )
    print("wrote", memo)

    summary = {
        "inlet_mps": INLET.tolist(),
        "buildings": len(buildings),
        "A": {"mean": stages["A_pure_pf"]["mean"], "max": stages["A_pure_pf"]["max"]},
        "B": {"mean": stages["B_prod"]["mean"], "max": stages["B_prod"]["max"]},
        "C": {"mean": stages["C_ns_lite"]["mean"], "max": stages["C_ns_lite"]["max"]},
        "plots": [str(p1), str(p2), str(p3), str(p4)],
        "memo": str(memo),
    }
    (OUT / "purePF_vs_B_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
