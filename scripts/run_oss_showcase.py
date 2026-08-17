#!/usr/bin/env python3
"""Reproducible Urban Flighter OSS showcase.

No network. Writes committed artifacts under docs/showcase/.

1) Toy-city 2D potential-flow CFD-lite field + streamlines.
2) UrbanFlow Gym synthetic-fixture baselines (direct / shortest / inlet-aware).
3) One-seed trajectory overlay for the README figure.

Honesty labels are written into every artifact:
  CFD-lite != Navier-Stokes
  policy NOT TRAINED
  full flow access = NO
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from urban_flighter_rl.potential_flow import (  # noqa: E402
    solve_potential_flow_slice,
    wind_from_speed_direction,
)
from urban_flighter_rl.render_potential_flow import plot_potential_flow  # noqa: E402
from urban_flighter_rl.world import UrbanWorld  # noqa: E402
from urbanflow_gym.baselines import BASELINE_ORDER, make_baseline  # noqa: E402
from urbanflow_gym.env import UrbanFlowConfig, UrbanFlowEnv  # noqa: E402
from urbanflow_gym.evaluation import evaluation_summary, run_baseline_evaluation  # noqa: E402
from urbanflow_gym.scenario import make_seeded_scenario  # noqa: E402

OUT = ROOT / "docs" / "showcase"
SEED = 10007
MAX_STEPS = 240


def run_potential_flow() -> dict:
    world = UrbanWorld.toy_city()
    inlet = wind_from_speed_direction(5.0, 270.0)
    result = solve_potential_flow_slice(
        world=world,
        inlet_velocity=inlet,
        altitude_m=28.0,
        cell_size_m=4.0,
    )
    flow_dir = OUT / "potential_flow_toy"
    flow_dir.mkdir(parents=True, exist_ok=True)
    image = flow_dir / "potential_flow_streamlines.png"
    plot_potential_flow(world, result, image)
    meta = {
        "label": "POTENTIAL-FLOW CFD-LITE",
        "navier_stokes": False,
        "mode": "toy_city",
        "buildings": len(world.buildings),
        "grid": [int(result.meta["nx"]), int(result.meta["ny"])],
        "iterations": int(result.meta["iterations"]),
        "residual": float(result.meta["residual"]),
        "mean_speed_mps": float(result.meta["mean_speed_mps"]),
        "max_speed_mps": float(result.meta["max_speed_mps"]),
        "inlet_from_deg": 270.0,
        "inlet_speed_mps": 5.0,
        "image": str(image.relative_to(ROOT)),
    }
    (flow_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def rollout_one(baseline_id: str, seed: int, max_steps: int) -> dict:
    scenario = make_seeded_scenario(seed, randomize_domain=True)
    config = UrbanFlowConfig(max_steps=max_steps)
    env = UrbanFlowEnv(config, fixed_scenario=scenario)
    baseline = make_baseline(baseline_id, scenario.public_context(), config)
    obs, info = env.reset(seed=seed, options={"scenario": scenario, "randomize_domain": False})
    path = [env.actor_snapshot().position_xy.copy()]
    for _ in range(max_steps):
        action = baseline.action(env.actor_snapshot())
        obs, reward, terminated, truncated, info = env.step(action)
        path.append(env.actor_snapshot().position_xy.copy())
        if terminated or truncated:
            break
    metrics = env.metrics()
    buildings = []
    for prism in scenario.geometry.prisms:
        mn, mx = prism.min_xy, prism.max_xy
        buildings.append(
            [
                [float(mn[0]), float(mn[1])],
                [float(mx[0]), float(mn[1])],
                [float(mx[0]), float(mx[1])],
                [float(mn[0]), float(mx[1])],
            ]
        )
    return {
        "baseline_id": baseline_id,
        "label": baseline.label,
        "uses_hidden_flow": bool(getattr(baseline, "uses_hidden_flow", False)),
        "path": np.asarray(path, dtype=float),
        "metrics": {
            k: (
                float(v)
                if isinstance(v, (int, float, np.floating, np.integer))
                else v
            )
            for k, v in metrics.items()
            if k != "reward_terms_total"
        },
        "buildings": buildings,
        "start": scenario.start_xy.tolist(),
        "goal": scenario.goal_xy.tolist(),
        "inlet": scenario.known_inlet_velocity_xy.tolist(),
        "obs_dim": int(np.asarray(obs).size),
    }


def plot_trajectories(rollouts: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=160)
    buildings = rollouts[0]["buildings"]
    for poly in buildings:
        arr = np.asarray(poly)
        if arr.ndim != 2 or len(arr) < 3:
            continue
        closed = np.vstack([arr, arr[0]])
        ax.fill(closed[:, 0], closed[:, 1], color="#c5cdd6", alpha=0.95, zorder=1)
        ax.plot(closed[:, 0], closed[:, 1], color="#4a5560", lw=0.7, zorder=2)
    colors = {
        "direct_goal": "#d1495b",
        "shortest_path": "#2a9d8f",
        "wind_aware_inlet": "#264653",
    }
    for item in rollouts:
        xy = item["path"]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=colors.get(item["baseline_id"], "k"),
            lw=2.0,
            label=item["label"],
            zorder=3,
        )
    start = np.asarray(rollouts[0]["start"])
    goal = np.asarray(rollouts[0]["goal"])
    ax.scatter([start[0]], [start[1]], c="#1d4ed8", s=42, zorder=4, label="start")
    ax.scatter([goal[0]], [goal[1]], c="#b45309", s=42, marker="*", zorder=4, label="goal")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("east (m)")
    ax.set_ylabel("north (m)")
    ax.set_title("UrbanFlow Gym fixture · seed 10007 · NOT TRAINED")
    ax.legend(loc="best", fontsize=8, framealpha=0.92)
    ax.text(
        0.01,
        0.01,
        "synthetic fixture  ·  hidden CFD-lite  ·  actor never sees the flow grid",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#334155",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_metrics(summary: dict, path: Path) -> None:
    rows = []
    for bid in BASELINE_ORDER:
        agg = summary["baselines"][bid]["aggregate"]
        rows.append(
            (
                summary["baselines"][bid]["label"],
                100.0 * float(agg["success_rate"]),
                float(agg["mean_path_length_m"]),
                float(agg["mean_relative_air_speed_energy"]),
                float(agg["collision_episode_rate"]) * 100.0,
            )
        )
    labels = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), dpi=160)
    series = [
        ([r[1] for r in rows], "success rate (%)"),
        ([r[2] for r in rows], "mean path (m)"),
        ([r[3] for r in rows], "mean rel-air energy"),
    ]
    colors = ["#2a9d8f", "#264653", "#e9c46a"]
    for ax, (vals, title), color in zip(axes, series, colors):
        ax.bar(range(len(labels)), vals, color=color)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(["direct", "shortest", "inlet"], fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("UrbanFlow Gym fixture eval · 3 seeds · NOT TRAINED · no privileged flow", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    flow_meta = run_potential_flow()

    eval_path = OUT / "gym_fixture_eval.json"
    payload = run_baseline_evaluation(
        seeds=(10007, 10009, 10037),
        max_steps=MAX_STEPS,
        artifact_path=eval_path,
    )
    summary = evaluation_summary(payload)

    rollouts = [rollout_one(bid, SEED, MAX_STEPS) for bid in BASELINE_ORDER]
    traj_png = OUT / "gym_fixture_trajectories.png"
    plot_trajectories(rollouts, traj_png)
    metrics_png = OUT / "gym_fixture_metrics.png"
    plot_metrics(summary, metrics_png)

    compact = {
        "case": "urban_flighter_oss_showcase_v1",
        "policy_status": "not_trained",
        "policy_full_flow_access": False,
        "navier_stokes_cfd": False,
        "real_cfd_validation_run": False,
        "network_required": False,
        "potential_flow": flow_meta,
        "gym_eval": {
            "evaluation_id": summary["evaluation_id"],
            "seeds": summary["evaluation_config"]["seeds"],
            "max_steps": summary["evaluation_config"]["max_steps"],
            "metrics": summary["metrics"],
            "artifact": str(eval_path.relative_to(ROOT)),
        },
        "figures": {
            "flow": flow_meta["image"],
            "trajectories": str(traj_png.relative_to(ROOT)),
            "metrics": str(metrics_png.relative_to(ROOT)),
        },
        "one_seed_rollouts": {
            item["baseline_id"]: {
                "label": item["label"],
                "uses_hidden_flow": item["uses_hidden_flow"],
                "obs_dim": item["obs_dim"],
                "metrics": item["metrics"],
                "steps": int(len(item["path"]) - 1),
            }
            for item in rollouts
        },
    }
    (OUT / "showcase_summary.json").write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(compact, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
