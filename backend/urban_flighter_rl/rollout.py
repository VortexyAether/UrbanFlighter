from __future__ import annotations

from .env import UrbanFlighterEnv
from .multi_agent import run_multi_drone_baseline
from .planner import _grid_astar, _pd_action_to_target
from .wind import UrbanWindField
from .world import UrbanWorld


def env_spec_payload() -> dict:
    env = UrbanFlighterEnv()
    return {
        "status": "rl_ready_api_surface",
        "policy_status": "deterministic_baseline_only_not_trained_rl",
        "policy_observation_summary": "relative goal + drone velocity + inlet/base wind + OSM/building proximity sectors only",
        "policy_had_privileged_flow_access": False,
        "environment": env.spec(),
        "data_sources": {
            "world": "UrbanWorld.toy_city by default; OSM worlds available through rollout/demo scripts",
            "wind": env.wind.to_dict(),
            "hidden_training_dynamics": "CFD-lite/heuristic wind sampler is used by simulator dynamics and reward metrics, not by policy observations",
            "real_cfd_eval_hook": "Swap UrbanWindField for AeroJAX/CFD snapshot sampler behind the same wind.at(position,t) interface",
        },
    }


def _single_drone_rollout(seed: int, max_steps: int, randomize_missions: bool) -> dict:
    env = UrbanFlighterEnv(max_steps=max_steps)
    obs, reset_info = env.reset(seed=seed)
    if randomize_missions:
        obs, reset_info = env.reset(seed=seed, options={"randomize_mission": True})
    waypoints = _grid_astar(env)
    waypoint_index = 1 if len(waypoints) > 1 else 0
    frames = [
        {
            "step": 0,
            "t_s": 0.0,
            "position": env.pos.tolist(),
            "velocity": env.vel.tolist(),
            "policy_observation": obs.tolist(),
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "policy_had_privileged_flow_access": reset_info["policy_had_privileged_flow_access"],
        }
    ]

    for _ in range(max_steps):
        target = waypoints[waypoint_index]
        if env.world.nearest_building_clearance(env.pos) < 2.0 and waypoint_index < len(waypoints) - 1:
            waypoint_index += 1
            target = waypoints[waypoint_index]
        if waypoint_index < len(waypoints) - 1:
            distance_to_waypoint = float(((target - env.pos) ** 2).sum() ** 0.5)
            if distance_to_waypoint < 3.0:
                waypoint_index += 1
                target = waypoints[waypoint_index]

        action = _pd_action_to_target(env, target)
        result = env.step(action)
        frames.append({
            "step": env.steps,
            "t_s": env.t,
            "position": env.pos.tolist(),
            "velocity": env.vel.tolist(),
            "action": action.tolist(),
            "policy_observation": result.obs.tolist(),
            "reward": result.reward,
            "terminated": result.terminated,
            "truncated": result.truncated,
            "reward_terms": result.info["reward_terms"],
            "policy_had_privileged_flow_access": result.info["policy_had_privileged_flow_access"],
        })
        if result.terminated or result.truncated:
            break

    metrics = env.metrics()
    metrics["swept_building_hits"] = int(sum(
        1
        for index in range(1, len(env.trajectory))
        if env.world.segment_hits_building(env.trajectory[index - 1], env.trajectory[index], margin=0.75)
    ))
    metrics.update({
        "controller": "deterministic_wind_aware_grid_astar_pd",
        "policy_status": "deterministic_baseline_only_not_trained_rl",
        "seed": int(seed),
        "waypoint_count": len(waypoints),
    })
    return {
        "status": "ok",
        "policy_label": "Deterministic baseline, not trained RL",
        "policy_had_privileged_flow_access": False,
        "environment_id": env.metadata["name"],
        "seed": int(seed),
        "max_steps": int(max_steps),
        "n_drones": 1,
        "metrics": metrics,
        "reward_terms": list(metrics["reward_terms_total"].keys()),
        "cost_metrics": {
            "collisions": metrics["collisions"],
            "boundary_violations": metrics["boundary_violations"],
            "swept_building_hits": metrics["swept_building_hits"],
            "min_building_clearance_m": metrics["min_building_clearance_m"],
            "energy_relative_airspeed_l2": metrics["energy_relative_airspeed_l2"],
            "final_distance_m": metrics["final_distance_m"],
        },
        "trajectory": frames,
        "waypoints": [point.tolist() for point in waypoints],
        "missions": [{"drone_id": "drone_1", "start": env.start.tolist(), "goal": env.goal.tolist()}],
        "data_sources": {
            "world": env.world.to_dict(),
            "wind": env.wind.to_dict(),
            "hidden_training_dynamics": "UrbanWindField.at(pos,t) is hidden dynamics; policy observation only gets inlet/base wind.",
        },
    }


def run_deterministic_baseline_rollout(
    seed: int = 7,
    max_steps: int = 300,
    n_drones: int = 4,
    randomize_missions: bool = True,
) -> dict:
    bounded_drones = max(1, int(n_drones))
    if bounded_drones == 1:
        return _single_drone_rollout(seed=seed, max_steps=max_steps, randomize_missions=randomize_missions)

    world = UrbanWorld.toy_city()
    wind = UrbanWindField(world)
    runs, aggregate = run_multi_drone_baseline(
        world=world,
        wind=wind,
        n_drones=bounded_drones,
        max_steps=max_steps,
        seed=seed,
        randomize_missions=randomize_missions,
    )
    reward_terms_total: dict[str, float] = {}
    for run in runs:
        for key, value in run.metrics["reward_terms_total"].items():
            reward_terms_total[key] = reward_terms_total.get(key, 0.0) + float(value)

    metrics = {
        "success": bool(aggregate["all_success"]),
        "success_count": int(aggregate["success_count"]),
        "steps": int(aggregate["max_steps_used"]),
        "return": float(sum(run.metrics["return"] for run in runs)),
        "path_length_m": float(aggregate["total_path_length_m"]),
        "energy_relative_airspeed_l2": float(aggregate["total_energy_relative_airspeed_l2"]),
        "collisions": int(aggregate["total_collisions"]),
        "boundary_violations": int(aggregate["total_boundary_violations"]),
        "swept_building_hits": int(aggregate["swept_building_hits"]),
        "separation_violations": int(sum(run.metrics["separation_violations"] for run in runs)),
        "min_building_clearance_m": float(min(run.metrics["min_building_clearance_m"] for run in runs)),
        "min_pairwise_separation_m": float(aggregate["min_pairwise_separation_m"]),
        "final_distance_m": float(sum(run.metrics["final_distance_m"] for run in runs)),
        "reward_terms_total": reward_terms_total,
        "controller": aggregate["controller"],
        "policy_status": "deterministic_baseline_only_not_trained_rl",
        "waypoint_count": int(sum(run.metrics["waypoint_count"] for run in runs)),
        "policy_had_privileged_flow_access": False,
    }
    return {
        "status": "ok",
        "policy_label": "Deterministic/randomized seeded baseline, not trained RL",
        "policy_had_privileged_flow_access": False,
        "environment_id": UrbanFlighterEnv.metadata["name"],
        "seed": int(seed),
        "max_steps": int(max_steps),
        "n_drones": bounded_drones,
        "randomize_missions": bool(randomize_missions),
        "metrics": metrics,
        "aggregate": aggregate,
        "reward_terms": list(reward_terms_total.keys()),
        "cost_metrics": {
            "collisions": metrics["collisions"],
            "boundary_violations": metrics["boundary_violations"],
            "swept_building_hits": metrics["swept_building_hits"],
            "separation_violations": metrics["separation_violations"],
            "min_building_clearance_m": metrics["min_building_clearance_m"],
            "min_pairwise_separation_m": metrics["min_pairwise_separation_m"],
            "energy_relative_airspeed_l2": metrics["energy_relative_airspeed_l2"],
            "path_length_m": metrics["path_length_m"],
            "final_distance_m": metrics["final_distance_m"],
        },
        "missions": [
            {"drone_id": run.drone_id, "start": run.start.tolist(), "goal": run.goal.tolist()}
            for run in runs
        ],
        "drones": [
            {
                "drone_id": run.drone_id,
                "metrics": run.metrics,
                "trajectory": run.frames,
                "waypoints": [point.tolist() for point in run.waypoints],
            }
            for run in runs
        ],
        "trajectory": runs[0].frames,
        "data_sources": {
            "world": world.to_dict(),
            "wind": wind.to_dict(),
            "hidden_training_dynamics": "UrbanWindField.at(pos,t) is hidden dynamics; policy observation only gets inlet/base wind.",
        },
    }
