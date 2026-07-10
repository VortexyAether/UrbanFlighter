from __future__ import annotations

import argparse
from pathlib import Path

from urban_flighter_rl.data_sources import fetch_osm_world
from urban_flighter_rl.potential_flow import (
    solve_potential_flow_slice,
    wind_from_speed_direction,
    write_potential_flow_outputs,
)
from urban_flighter_rl.render_potential_flow import plot_potential_flow
from urban_flighter_rl.world import UrbanWorld


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2D potential-flow CFD-lite slice for Urban_Flighter")
    parser.add_argument("--mode", choices=["toy", "real"], default="real")
    parser.add_argument("--lat", type=float, default=37.497942)
    parser.add_argument("--lon", type=float, default=127.027621)
    parser.add_argument("--radius", type=float, default=260.0)
    parser.add_argument("--max-buildings", type=int, default=80)
    parser.add_argument("--wind-speed", type=float, default=5.0)
    parser.add_argument("--wind-deg", type=float, default=270.0, help="meteorological direction: wind from degrees north")
    parser.add_argument("--altitude", type=float, default=35.0)
    parser.add_argument("--cell-size", type=float, default=8.0)
    parser.add_argument("--out", default="results/potential_flow_gangnam")
    args = parser.parse_args()

    if args.mode == "real":
        world, osm_meta = fetch_osm_world(args.lat, args.lon, radius_m=args.radius, max_buildings=args.max_buildings)
    else:
        world = UrbanWorld.toy_city()
        osm_meta = {"source": "toy_city", "usable_buildings": len(world.buildings), "radius_m": None}

    inlet = wind_from_speed_direction(args.wind_speed, args.wind_deg)
    result = solve_potential_flow_slice(
        world=world,
        inlet_velocity=inlet,
        altitude_m=args.altitude,
        cell_size_m=args.cell_size,
    )
    out = Path(args.out)
    paths = write_potential_flow_outputs(result, out)
    image_path = out / "potential_flow_streamlines.png"
    plot_potential_flow(world, result, image_path)

    print("Urban_Flighter potential-flow CFD-lite complete")
    print(f"mode: {args.mode}")
    print(f"buildings: {len(world.buildings)}")
    print(f"wind_speed_mps: {args.wind_speed}")
    print(f"wind_deg_from_north: {args.wind_deg}")
    print(f"grid: {result.meta['nx']} x {result.meta['ny']}")
    print(f"iterations: {result.meta['iterations']}")
    print(f"residual: {result.meta['residual']:.6g}")
    print(f"streamlines: {result.meta['streamline_count']}")
    print(f"mean_speed_mps: {result.meta['mean_speed_mps']:.3f}")
    print(f"max_speed_mps: {result.meta['max_speed_mps']:.3f}")
    print(f"field_npz: {Path(paths['field_npz']).resolve()}")
    print(f"streamlines_json: {Path(paths['streamlines_json']).resolve()}")
    print(f"meta_json: {Path(paths['meta_json']).resolve()}")
    print(f"image: {image_path.resolve()}")
    if osm_meta:
        print(f"source: {osm_meta.get('source')}")
        print(f"usable_buildings: {osm_meta.get('usable_buildings')}")
        print(f"osm_elements: {osm_meta.get('osm_elements')}")


if __name__ == "__main__":
    main()
