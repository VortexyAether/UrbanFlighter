from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from urban_flighter_rl.data_sources import fetch_osm_world
from urban_flighter_rl.potential_flow import solve_potential_flow_slice, wind_from_speed_direction, write_potential_flow_outputs
from urban_flighter_rl.wind_corrections import apply_wall_damping_and_wake


def _draw_buildings_3d(ax, world):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    for b in world.buildings:
        x0, y0 = b.min_xy
        x1, y1 = b.max_xy
        z0, z1 = 0.0, b.height
        verts = [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        ]
        ax.add_collection3d(Poly3DCollection(verts, alpha=0.16, facecolor="#334155", edgecolor="#0f172a", linewidth=0.25))


def render_3d_overview(world, slices, out_path: Path, elev: float, azim: float, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    _draw_buildings_3d(ax, world)
    colors = ["#38bdf8", "#fbbf24", "#a78bfa", "#34d399", "#fb7185"]
    for idx, (alt, result) in enumerate(slices):
        color = colors[idx % len(colors)]
        for line in result.streamlines:
            arr = np.asarray(line, dtype=float)
            if len(arr) >= 2:
                ax.plot(arr[:, 0], arr[:, 1], np.full(len(arr), alt), color=color, linewidth=0.9, alpha=0.75)
    x0, x1, y0, y1, z0, z1 = world.bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def render_xy(result, out_path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = result.x, result.y
    speed = np.sqrt(result.ux * result.ux + result.uy * result.uy)
    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(speed.T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="turbo")
    fig.colorbar(img, ax=ax, label="speed [m/s]")
    ax.contour(x, y, result.mask.T.astype(float), levels=[0.5], colors="white", linewidths=0.5)
    for line in result.streamlines:
        arr = np.asarray(line, dtype=float)
        if len(arr) >= 2:
            ax.plot(arr[:, 0], arr[:, 1], color="white", linewidth=0.75, alpha=0.82)
    skip = max(1, len(x) // 36)
    xx, yy = np.meshgrid(x[::skip], y[::skip], indexing="ij")
    ax.quiver(xx, yy, result.ux[::skip, ::skip], result.uy[::skip, ::skip], color="black", alpha=0.45, scale=100, width=0.002)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render multi-angle views for B: potential + wall damping + wake")
    parser.add_argument("--lat", type=float, default=37.497942)
    parser.add_argument("--lon", type=float, default=127.027621)
    parser.add_argument("--radius", type=float, default=220.0)
    parser.add_argument("--max-buildings", type=int, default=70)
    parser.add_argument("--cell-size", type=float, default=2.5)
    parser.add_argument("--wind-speed", type=float, default=5.0)
    parser.add_argument("--wind-deg", type=float, default=270.0)
    parser.add_argument("--altitudes", default="20,35,60,90")
    parser.add_argument("--out", default="results/damped_wake_views_gangnam")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world, osm_meta = fetch_osm_world(args.lat, args.lon, radius_m=args.radius, max_buildings=args.max_buildings)
    inlet = wind_from_speed_direction(args.wind_speed, args.wind_deg)
    altitudes = [float(v.strip()) for v in args.altitudes.split(",") if v.strip()]
    slices = []
    metas = []
    for alt in altitudes:
        base = solve_potential_flow_slice(world, inlet, altitude_m=alt, cell_size_m=args.cell_size, streamline_count=30)
        corrected = apply_wall_damping_and_wake(world, base, inlet)
        slices.append((alt, corrected))
        metas.append(corrected.meta)
        write_potential_flow_outputs(corrected, out / f"slice_{int(round(alt))}m")
        render_xy(corrected, out / f"xy_z{int(round(alt))}.png", f"B damped+wake XY slice z={alt:.0f}m")

    render_3d_overview(world, slices, out / "overview_iso.png", 34, -58, "B damped+wake 3D overview")
    render_3d_overview(world, slices, out / "overview_top_oblique.png", 62, -70, "B damped+wake 3D top-oblique")
    render_3d_overview(world, slices, out / "overview_side.png", 18, -90, "B damped+wake side view")
    summary = {
        "model": "potential-flow-wall-damping-wake-multi-view",
        "lat": args.lat,
        "lon": args.lon,
        "radius_m": args.radius,
        "buildings": len(world.buildings),
        "osm_meta": osm_meta,
        "cell_size_m": args.cell_size,
        "altitudes_m": altitudes,
        "slice_meta": metas,
        "images": {
            "overview_iso": str(out / "overview_iso.png"),
            "overview_top_oblique": str(out / "overview_top_oblique.png"),
            "overview_side": str(out / "overview_side.png"),
            "xy_slices": [str(out / f"xy_z{int(round(a))}.png") for a in altitudes],
        },
    }
    (out / "damped_wake_views_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("damped+wake views complete")
    print(f"buildings: {len(world.buildings)}")
    print(f"out: {out.resolve()}")
    for p in summary["images"].values():
        print(p)


if __name__ == "__main__":
    main()
