from __future__ import annotations

import argparse
from pathlib import Path

from urban_flighter_rl.data_sources import download_satellite_tile, fetch_osm_world
from urban_flighter_rl.multi_agent import run_multi_drone_baseline, write_multi_metrics, write_trajectories_json
from urban_flighter_rl.render import write_metrics
from urban_flighter_rl.render_multi import plot_multi_drone_trajectories
from urban_flighter_rl.wind import UrbanWindField


def main():
    parser = argparse.ArgumentParser(description="Run multi-drone RL-style rollout over real OSM city buildings")
    parser.add_argument("--lat", type=float, default=37.497942)
    parser.add_argument("--lon", type=float, default=127.027621)
    parser.add_argument("--radius", type=float, default=260.0)
    parser.add_argument("--max-buildings", type=int, default=140)
    parser.add_argument("--drones", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=1100)
    parser.add_argument("--satellite-zoom", type=int, default=17)
    parser.add_argument("--out", default="results/multi_drone_gangnam")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    world, osm_meta = fetch_osm_world(args.lat, args.lon, radius_m=args.radius, max_buildings=args.max_buildings)
    satellite_path = out / "satellite_center_tile.jpg"
    satellite_meta = download_satellite_tile(args.lat, args.lon, satellite_path, zoom=args.satellite_zoom)

    wind = UrbanWindField(world, base_wind=(5.0, 1.2, 0.0))
    runs, aggregate = run_multi_drone_baseline(
        world=world,
        wind=wind,
        n_drones=args.drones,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    image_path = out / "urban_flighter_multi_drone_real_city.png"
    metrics_path = out / "urban_flighter_multi_drone_metrics.json"
    trajectories_path = out / "urban_flighter_multi_drone_trajectories.json"
    osm_meta_path = out / "osm_buildings_meta.json"
    satellite_meta_path = out / "satellite_tile_meta.json"

    plot_multi_drone_trajectories(world, wind, runs, image_path, satellite_tile_path=satellite_path)
    write_multi_metrics(runs, {
        **aggregate,
        "lat": args.lat,
        "lon": args.lon,
        "radius_m": args.radius,
        "osm_buildings": osm_meta["usable_buildings"],
        "osm_elements": osm_meta["osm_elements"],
        "satellite_tile": str(satellite_path),
        "satellite_tile_url": satellite_meta["url"],
        "result_image": str(image_path),
        "metrics_json": str(metrics_path),
        "trajectories_json": str(trajectories_path),
    }, metrics_path)
    write_trajectories_json(runs, trajectories_path)
    write_metrics(osm_meta, osm_meta_path)
    write_metrics(satellite_meta, satellite_meta_path)

    print("Urban Flighter multi-drone real-city rollout complete")
    print(f"n_drones: {aggregate['n_drones']}")
    print(f"success_count: {aggregate['success_count']}")
    print(f"all_success: {aggregate['all_success']}")
    print(f"total_collisions: {aggregate['total_collisions']}")
    print(f"near_miss_count_sep_lt_10m: {aggregate['near_miss_count_sep_lt_10m']}")
    print(f"min_pairwise_separation_m: {aggregate['min_pairwise_separation_m']}")
    print(f"total_energy_relative_airspeed_l2: {aggregate['total_energy_relative_airspeed_l2']}")
    print(f"total_path_length_m: {aggregate['total_path_length_m']}")
    print(f"osm_buildings: {osm_meta['usable_buildings']}")
    print(f"osm_elements: {osm_meta['osm_elements']}")
    print(f"result_image: {image_path}")
    print(f"metrics_json: {metrics_path}")
    print(f"trajectories_json: {trajectories_path}")
    print(f"satellite_tile: {satellite_path}")


if __name__ == "__main__":
    main()
