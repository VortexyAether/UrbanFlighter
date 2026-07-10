from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from urban_flighter_rl.data_sources import fetch_osm_world
from urban_flighter_rl.panel_flow_2d import compute_panel_flow_2d, write_panel_flow_outputs
from urban_flighter_rl.potential_flow import solve_potential_flow_slice, wind_from_speed_direction, write_potential_flow_outputs
from urban_flighter_rl.wind_corrections import apply_wall_damping_and_wake


def _plot_field(ax, title, x, y, ux, uy, mask, streamlines):
    speed = np.sqrt(ux * ux + uy * uy)
    img = ax.imshow(speed.T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="turbo", vmin=0, vmax=max(8.0, float(speed.max())))
    ax.contour(x, y, mask.T.astype(float), levels=[0.5], colors="white", linewidths=0.45)
    for line in streamlines:
        arr = np.asarray(line, dtype=float)
        if len(arr) >= 2:
            ax.plot(arr[:, 0], arr[:, 1], color="white", linewidth=0.65, alpha=0.78)
    skip = max(1, len(x) // 32)
    xx, yy = np.meshgrid(x[::skip], y[::skip], indexing="ij")
    ax.quiver(xx, yy, ux[::skip, ::skip], uy[::skip, ::skip], color="black", alpha=0.45, scale=120, width=0.002)
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    return img


def render_comparison(base, corrected, panel, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)
    img = _plot_field(axes[0], "A. grid potential-flow", base.x, base.y, base.ux, base.uy, base.mask, base.streamlines)
    _plot_field(axes[1], "B. potential + wall damping + wake", corrected.x, corrected.y, corrected.ux, corrected.uy, corrected.mask, corrected.streamlines)
    _plot_field(axes[2], "C. source-panel guidance field", panel.x, panel.y, panel.ux, panel.uy, panel.mask, panel.streamlines)
    fig.colorbar(img, ax=axes, shrink=0.82, label="speed [m/s]")
    fig.suptitle("Urban_Flighter CFD-lite comparison over real OSM buildings", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=175)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare potential-flow corrections vs 2D panel method")
    parser.add_argument("--lat", type=float, default=37.497942)
    parser.add_argument("--lon", type=float, default=127.027621)
    parser.add_argument("--radius", type=float, default=220.0)
    parser.add_argument("--max-buildings", type=int, default=70)
    parser.add_argument("--cell-size", type=float, default=5.0)
    parser.add_argument("--altitude", type=float, default=35.0)
    parser.add_argument("--wind-speed", type=float, default=5.0)
    parser.add_argument("--wind-deg", type=float, default=270.0)
    parser.add_argument("--out", default="results/potential_panel_compare_gangnam")
    args = parser.parse_args()

    started = time.perf_counter()
    world, osm_meta = fetch_osm_world(args.lat, args.lon, radius_m=args.radius, max_buildings=args.max_buildings)
    fetch_elapsed = time.perf_counter() - started
    inlet = wind_from_speed_direction(args.wind_speed, args.wind_deg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base_started = time.perf_counter()
    base = solve_potential_flow_slice(world, inlet, altitude_m=args.altitude, cell_size_m=args.cell_size, streamline_count=28)
    base_elapsed = time.perf_counter() - base_started
    corrected_started = time.perf_counter()
    corrected = apply_wall_damping_and_wake(world, base, inlet)
    corrected_elapsed = time.perf_counter() - corrected_started
    panel_started = time.perf_counter()
    panel = compute_panel_flow_2d(world, inlet, altitude_m=args.altitude, cell_size_m=args.cell_size, max_panel_len=18.0, streamline_count=28)
    panel_elapsed = time.perf_counter() - panel_started

    paths = {
        "base": write_potential_flow_outputs(base, out / "base_grid_potential"),
        "corrected": write_potential_flow_outputs(corrected, out / "potential_damped_wake"),
        "panel": write_panel_flow_outputs(panel, out / "panel_method"),
    }
    image = out / "comparison_potential_damping_panel.png"
    render_comparison(base, corrected, panel, image)

    summary = {
        "lat": args.lat,
        "lon": args.lon,
        "radius_m": args.radius,
        "altitude_m": args.altitude,
        "buildings": len(world.buildings),
        "osm_meta": osm_meta,
        "fetch_elapsed_s": fetch_elapsed,
        "base_elapsed_s": base_elapsed,
        "corrected_elapsed_s": corrected_elapsed,
        "panel_elapsed_s": panel_elapsed,
        "models": {
            "base_grid_potential": base.meta,
            "potential_damped_wake": corrected.meta,
            "panel_method": panel.meta,
        },
        "paths": {**paths, "comparison_image": str(image)},
    }
    summary_path = out / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Urban_Flighter potential/panel comparison complete")
    print(f"buildings: {len(world.buildings)}")
    print(f"fetch_elapsed_s: {fetch_elapsed:.3f}")
    print(f"base_elapsed_s: {base_elapsed:.3f}")
    print(f"corrected_elapsed_s: {corrected_elapsed:.3f}")
    print(f"panel_elapsed_s: {panel_elapsed:.3f}")
    print(f"panel_count: {panel.meta['panel_count']}")
    print(f"panel_boundary_normal_residual_mean: {panel.meta['mean_boundary_normal_residual_mps']:.4f}")
    print(f"base_streamlines: {base.meta['streamline_count']}")
    print(f"corrected_streamlines: {corrected.meta['streamline_count']}")
    print(f"panel_streamlines: {panel.meta['streamline_count']}")
    print(f"summary: {summary_path.resolve()}")
    print(f"image: {image.resolve()}")


if __name__ == "__main__":
    main()
