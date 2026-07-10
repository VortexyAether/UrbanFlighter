from __future__ import annotations

import argparse
from pathlib import Path

from urban_flighter_rl import UrbanFlighterEnv
from urban_flighter_rl.planner import run_baseline
from urban_flighter_rl.render import plot_trajectory, write_metrics


def main():
    parser = argparse.ArgumentParser(description="Run Urban Flighter 3D RL-ready prototype demo")
    parser.add_argument("--out", default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out = Path(args.out)
    env = UrbanFlighterEnv(max_steps=800)
    env.reset(seed=args.seed)
    metrics = run_baseline(env)

    image_path = out / "urban_flighter_3d_demo.png"
    metrics_path = out / "urban_flighter_metrics.json"
    plot_trajectory(env, image_path)
    metrics.update({
        "result_image": str(image_path),
        "metrics_json": str(metrics_path),
        "energy_model": "sum_t ||v_ground - w_local(x,y,z,t)||^2 * dt",
        "world": "toy_city_5_building_prisms",
    })
    write_metrics(metrics, metrics_path)

    print("Urban Flighter 3D prototype run complete")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
