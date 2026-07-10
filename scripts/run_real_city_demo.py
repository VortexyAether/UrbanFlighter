from __future__ import annotations

import argparse
from pathlib import Path

from urban_flighter_rl import UrbanFlighterEnv
from urban_flighter_rl.data_sources import download_satellite_tile, fetch_osm_world
from urban_flighter_rl.planner import run_baseline
from urban_flighter_rl.render import plot_trajectory, write_metrics
from urban_flighter_rl.wind import UrbanWindField


def main():
    parser = argparse.ArgumentParser(description="Run Urban Flighter against a real OSM city patch + satellite tile")
    parser.add_argument("--lat", type=float, default=37.451448, help="Center latitude")
    parser.add_argument("--lon", type=float, default=126.6515423, help="Center longitude")
    parser.add_argument("--radius", type=float, default=260.0, help="OSM query radius in meters")
    parser.add_argument("--max-buildings", type=int, default=120)
    parser.add_argument("--satellite-zoom", type=int, default=17)
    parser.add_argument("--out", default="results/real_city", help="Output directory")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    world, osm_meta = fetch_osm_world(args.lat, args.lon, radius_m=args.radius, max_buildings=args.max_buildings)
    satellite_path = out / "satellite_center_tile.jpg"
    satellite_meta = download_satellite_tile(args.lat, args.lon, satellite_path, zoom=args.satellite_zoom)

    # West/southwest-ish default inlet wind; weather API can replace this later.
    wind = UrbanWindField(world, base_wind=(5.0, 1.2, 0.0))
    env = UrbanFlighterEnv(world=world, wind=wind, max_steps=1000)
    env.reset(seed=args.seed)
    metrics = run_baseline(env)

    image_path = out / "urban_flighter_real_city_3d_demo.png"
    metrics_path = out / "urban_flighter_real_city_metrics.json"
    osm_meta_path = out / "osm_buildings_meta.json"
    satellite_meta_path = out / "satellite_tile_meta.json"

    plot_trajectory(env, image_path)
    write_metrics(osm_meta, osm_meta_path)
    write_metrics(satellite_meta, satellite_meta_path)

    metrics.update({
        "result_image": str(image_path),
        "metrics_json": str(metrics_path),
        "osm_meta_json": str(osm_meta_path),
        "satellite_tile": str(satellite_path),
        "satellite_meta_json": str(satellite_meta_path),
        "energy_model": "sum_t ||v_ground - w_local(x,y,z,t)||^2 * dt",
        "world": "real_osm_building_footprints_axis_aligned_prisms",
        "lat": args.lat,
        "lon": args.lon,
        "radius_m": args.radius,
        "osm_buildings": osm_meta["usable_buildings"],
        "osm_elements": osm_meta["osm_elements"],
        "satellite_source": satellite_meta["source"],
        "satellite_tile_url": satellite_meta["url"],
    })
    write_metrics(metrics, metrics_path)

    print("Urban Flighter real-city OSM/satellite demo complete")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
