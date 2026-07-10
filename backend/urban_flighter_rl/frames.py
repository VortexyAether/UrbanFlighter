from __future__ import annotations


def initial_policy_frame(drone_id: str, env, obs, info) -> dict:
    return {
        "step": 0,
        "t_s": 0.0,
        "drone_id": drone_id,
        "position": env.pos.tolist(),
        "velocity": env.vel.tolist(),
        "action": [0.0, 0.0, 0.0],
        "policy_observation": obs.tolist(),
        "reward": 0.0,
        "reward_terms": {},
        "terminated": False,
        "truncated": False,
        "policy_had_privileged_flow_access": info["policy_had_privileged_flow_access"],
    }


def step_policy_frame(drone_id: str, env, action, result) -> dict:
    return {
        "step": int(env.steps),
        "t_s": float(env.t),
        "drone_id": drone_id,
        "position": env.pos.tolist(),
        "velocity": env.vel.tolist(),
        "action": action.tolist(),
        "policy_observation": result.obs.tolist(),
        "reward": float(result.reward),
        "reward_terms": result.info["reward_terms"],
        "terminated": bool(result.terminated),
        "truncated": bool(result.truncated),
        "policy_had_privileged_flow_access": result.info["policy_had_privileged_flow_access"],
    }
