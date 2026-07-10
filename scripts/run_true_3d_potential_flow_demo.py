from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from urban_flighter_rl.data_sources import fetch_osm_world
from urban_flighter_rl.potential_flow import wind_from_speed_direction
from urban_flighter_rl.potential_flow_3d import solve_potential_flow_3d, write_potential_flow_3d_outputs
from urban_flighter_rl.render_true_potential_flow_3d import plot_true_potential_flow_3d
from urban_flighter_rl.world import UrbanWorld


def main() -> None:
    parser = argparse.ArgumentParser(description="Run true 3D voxel Laplace potential-flow CFD-lite over OSM buildings")
    parser.add_argument("--mode", choices=["toy", "real"], default="real")
    parser.add_argument("--lat", type=float, default=37.497942)
    parser.add_argument("--lon", type=float, default=127.027621)
    parser.add_argument("--radius", type=float, default=220.0)
    parser.add_argument("--max-buildings", type=int, default=70)
    parser.add_argument("--wind-speed", type=float, default=5.0)
    parser.add_argument("--wind-deg", type=float, default=270.0)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=48)
    parser.add_argument("--nz", type=int, default=24)
    parser.add_argument("--max-iter", type=int, default=1200)
    parser.add_argument("--out", default="results/potential_flow_gangnam_true3d")
    args = parser.parse_args()

    started = time.perf_counter()
    if args.mode == "real":
        world, osm_meta = fetch_osm_world(args.lat, args.lon, radius_m=args.radius, max_buildings=args.max_buildings)
    else:
        world = UrbanWorld.toy_city()
        osm_meta = {"source": "toy_city", "usable_buildings": len(world.buildings)}
    fetch_elapsed = time.perf_counter() - started

    inlet2 = wind_from_speed_direction(args.wind_speed, args.wind_deg)
    result = solve_potential_flow_3d(
        world=world,
        inlet_velocity=(float(inlet2[0]), float(inlet2[1]), 0.0),
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        max_iter=args.max_iter,
    )

    out = Path(args.out)
    paths = write_potential_flow_3d_outputs(result, out)
    render_started = time.perf_counter()
    image_path = out / "potential_flow_true3d_streamlines.png"
    plot_true_potential_flow_3d(world, result, image_path)
    render_elapsed = time.perf_counter() - render_started

    summary = {
        **result.meta,
        "lat": args.lat if args.mode == "real" else None,
        "lon": args.lon if args.mode == "real" else None,
        "radius_m": args.radius,
        "buildings": len(world.buildings),
        "osm_meta": osm_meta,
        "fetch_elapsed_s": fetch_elapsed,
        "render_elapsed_s": render_elapsed,
        "end_to_end_elapsed_s": time.perf_counter() - started,
        "paths": {**paths, "image": str(image_path)},
    }
    summary_path = out / "potential_flow_true3d_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Urban_Flighter true 3D potential-flow CFD-lite complete")
    print(f"mode: {args.mode}")
    print(f"buildings: {len(world.buildings)}")
    print(f"grid: {args.nx} x {args.ny} x {args.nz} = {args.nx * args.ny * args.nz} cells")
    print(f"iterations: {result.meta['iterations']}")
    print(f"residual: {result.meta['residual']:.6g}")
    print(f"solve_elapsed_s: {result.meta['solve_elapsed_s']:.3f}")
    print(f"trace_elapsed_s: {result.meta['trace_elapsed_s']:.3f}")
    print(f"render_elapsed_s: {render_elapsed:.3f}")
    print(f"end_to_end_elapsed_s: {summary['end_to_end_elapsed_s']:.3f}")
    print(f"streamlines: {result.meta['streamline_count']}")
    print(f"mean_speed_mps: {result.meta['mean_speed_mps']:.3f}")
    print(f"max_speed_mps: {result.meta['max_speed_mps']:.3f}")
    print(f"mean_abs_w_mps: {result.meta['mean_abs_w_mps']:.5f}")
    print(f"max_abs_w_mps: {result.meta['max_abs_w_mps']:.5f}")
    print(f"field_npz: {Path(paths['field_npz']).resolve()}")
    print(f"streamlines_json: {Path(paths['streamlines_json']).resolve()}")
    print(f"summary_json: {summary_path.resolve()}")
    print(f"image: {image_path.resolve()}")


if __name__ == "__main__":
    main()
