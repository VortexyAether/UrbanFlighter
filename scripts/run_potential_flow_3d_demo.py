from __future__ import annotations

import argparse
import json
from pathlib import Path

from urban_flighter_rl.data_sources import fetch_osm_world
from urban_flighter_rl.potential_flow import solve_potential_flow_slice, wind_from_speed_direction, write_potential_flow_outputs
from urban_flighter_rl.render_potential_flow_3d import plot_potential_flow_3d
from urban_flighter_rl.world import UrbanWorld


def main() -> None:
    parser = argparse.ArgumentParser(description="Render stacked 3D potential-flow CFD-lite slices over OSM buildings")
    parser.add_argument("--mode", choices=["toy", "real"], default="real")
    parser.add_argument("--lat", type=float, default=37.497942)
    parser.add_argument("--lon", type=float, default=127.027621)
    parser.add_argument("--radius", type=float, default=220.0)
    parser.add_argument("--max-buildings", type=int, default=70)
    parser.add_argument("--wind-speed", type=float, default=5.0)
    parser.add_argument("--wind-deg", type=float, default=270.0)
    parser.add_argument("--cell-size", type=float, default=7.0)
    parser.add_argument("--altitudes", default="20,35,60,90")
    parser.add_argument("--out", default="results/potential_flow_gangnam_3d")
    args = parser.parse_args()

    if args.mode == "real":
        world, osm_meta = fetch_osm_world(args.lat, args.lon, radius_m=args.radius, max_buildings=args.max_buildings)
    else:
        world = UrbanWorld.toy_city()
        osm_meta = {"source": "toy_city", "usable_buildings": len(world.buildings)}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inlet = wind_from_speed_direction(args.wind_speed, args.wind_deg)
    altitudes = [float(x.strip()) for x in args.altitudes.split(",") if x.strip()]

    slices = []
    metas = []
    for alt in altitudes:
        result = solve_potential_flow_slice(
            world=world,
            inlet_velocity=inlet,
            altitude_m=alt,
            cell_size_m=args.cell_size,
            streamline_count=18,
            streamline_steps=360,
        )
        slice_dir = out / f"slice_{int(round(alt))}m"
        paths = write_potential_flow_outputs(result, slice_dir)
        slices.append((alt, result))
        metas.append({"altitude_m": alt, **result.meta, "paths": paths})

    image_path = out / "potential_flow_3d_streamlines.png"
    plot_potential_flow_3d(world, slices, image_path)

    summary = {
        "model": "stacked-2d-potential-flow-cfd-lite",
        "not_full_3d_cfd": True,
        "lat": args.lat if args.mode == "real" else None,
        "lon": args.lon if args.mode == "real" else None,
        "radius_m": args.radius,
        "buildings": len(world.buildings),
        "osm_meta": osm_meta,
        "wind_speed_mps": args.wind_speed,
        "wind_deg_from_north": args.wind_deg,
        "cell_size_m": args.cell_size,
        "altitudes_m": altitudes,
        "slices": metas,
        "image": str(image_path),
    }
    summary_path = out / "potential_flow_3d_meta.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Urban_Flighter 3D potential-flow CFD-lite complete")
    print(f"mode: {args.mode}")
    print(f"buildings: {len(world.buildings)}")
    print(f"grid_cell_size_m: {args.cell_size}")
    print(f"altitudes_m: {altitudes}")
    for meta in metas:
        print(f"slice {meta['altitude_m']}m: grid={meta['nx']}x{meta['ny']} streamlines={meta['streamline_count']} max_speed={meta['max_speed_mps']:.3f}")
    print(f"meta_json: {summary_path.resolve()}")
    print(f"image: {image_path.resolve()}")


if __name__ == "__main__":
    main()
