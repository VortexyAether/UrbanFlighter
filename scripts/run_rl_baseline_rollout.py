from __future__ import annotations

import argparse
import json
from pathlib import Path

from urban_flighter_rl.rollout import run_deterministic_baseline_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Urban Flighter RL baseline rollout")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--drones", type=int, default=4)
    parser.add_argument("--out", default="results/rl_baseline_rollout.json")
    parser.add_argument("--metrics-out", default="results/rl_baseline_metrics.json")
    parser.add_argument("--fixed-missions", action="store_true")
    args = parser.parse_args()

    payload = run_deterministic_baseline_rollout(
        seed=args.seed,
        max_steps=args.max_steps,
        n_drones=args.drones,
        randomize_missions=not args.fixed_missions,
    )
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({
        "metrics": payload["metrics"],
        "cost_metrics": payload["cost_metrics"],
        "missions": payload["missions"],
        "policy_had_privileged_flow_access": payload["policy_had_privileged_flow_access"],
    }, indent=2), encoding="utf-8")

    metrics = payload["metrics"]
    print("Urban Flighter RL deterministic baseline rollout complete")
    print(f"policy_label: {payload['policy_label']}")
    print(f"seed: {payload['seed']}")
    print(f"n_drones: {payload['n_drones']}")
    print(f"steps: {metrics['steps']}")
    print(f"success: {metrics['success']}")
    print(f"success_count: {metrics.get('success_count', int(metrics['success']))}")
    print(f"return: {metrics['return']:.3f}")
    print(f"collisions: {metrics['collisions']}")
    print(f"boundary_violations: {metrics.get('boundary_violations', 0)}")
    print(f"swept_building_hits: {metrics.get('swept_building_hits', 0)}")
    print(f"separation_violations: {metrics.get('separation_violations', 0)}")
    print(f"min_building_clearance_m: {metrics['min_building_clearance_m']:.3f}")
    print(f"energy_relative_airspeed_l2: {metrics['energy_relative_airspeed_l2']:.3f}")
    print(f"trajectory_json: {output_path}")
    print(f"metrics_json: {metrics_path}")
    print(f"policy_had_privileged_flow_access: {payload['policy_had_privileged_flow_access']}")


if __name__ == "__main__":
    main()
