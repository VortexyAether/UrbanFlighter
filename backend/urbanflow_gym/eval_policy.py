from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .baselines import BASELINE_ORDER, make_baseline
from .env import UrbanFlowConfig, UrbanFlowEnv
from .evaluation import run_baseline_evaluation
from .live_scenario import make_live_scenario, read_training_bundle
from .scenario import make_seeded_scenario


def _rollout(env: UrbanFlowEnv, action_fn, max_steps: int) -> dict:
    observation, _ = env.reset()
    for _ in range(max_steps):
        action = action_fn(env, observation)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    metrics = env.metrics()
    metrics["actor_full_flow_access"] = False
    return metrics


def evaluate_random_policy(env: UrbanFlowEnv, seed: int, max_steps: int) -> dict:
    rng = np.random.default_rng(seed)

    def action_fn(_env: UrbanFlowEnv, _observation: np.ndarray) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=2)

    return _rollout(env, action_fn, max_steps)


def evaluate_baseline(env: UrbanFlowEnv, baseline_id: str, max_steps: int) -> dict:
    baseline = make_baseline(baseline_id, env.scenario.public_context(), env.config)

    def action_fn(inner: UrbanFlowEnv, _observation: np.ndarray) -> np.ndarray:
        return baseline.action(inner.actor_snapshot())

    return _rollout(env, action_fn, max_steps)


def evaluate_checkpoint(
    env: UrbanFlowEnv,
    checkpoint: Path,
    max_steps: int,
) -> dict:
    try:
        from stable_baselines3 import PPO, SAC
    except ImportError as exc:
        raise SystemExit(
            "Evaluating a learned checkpoint requires UrbanFlow training extras."
        ) from exc

    loader = PPO.load if "ppo" in checkpoint.name.lower() else SAC.load
    model = loader(checkpoint)

    def action_fn(_env: UrbanFlowEnv, observation: np.ndarray) -> np.ndarray:
        action, _ = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=float)

    metrics = _rollout(env, action_fn, max_steps)
    metrics["checkpoint"] = str(checkpoint)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate random, baseline, or learned UrbanFlow policies. "
            "Never grants full-flow access."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=10007)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/urbanflow_gym/policy_eval.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = UrbanFlowConfig(max_steps=args.max_steps)
    if args.snapshot is not None:
        record = read_training_bundle(args.snapshot)
        scenario = make_live_scenario(record, seed=args.seed)
        env = UrbanFlowEnv(config, fixed_scenario=scenario)
        world = {"kind": "live_snapshot", "scenario_id": scenario.scenario_id}
    else:
        env = UrbanFlowEnv(config)
        env.reset(seed=args.seed)
        world = {"kind": "synthetic_fixture", "seed": args.seed}

    payload = {
        "world": world,
        "policy_status": "not_trained" if args.checkpoint is None else "checkpoint_evaluated",
        "full_flow_access": False,
        "real_cfd_validation": False,
        "random": evaluate_random_policy(
            UrbanFlowEnv(config, fixed_scenario=env.scenario),
            args.seed,
            args.max_steps,
        ),
        "baselines": {
            baseline_id: evaluate_baseline(
                UrbanFlowEnv(config, fixed_scenario=env.scenario),
                baseline_id,
                args.max_steps,
            )
            for baseline_id in BASELINE_ORDER
        },
    }
    if args.checkpoint is not None:
        payload["learned"] = evaluate_checkpoint(
            UrbanFlowEnv(config, fixed_scenario=env.scenario),
            args.checkpoint,
            args.max_steps,
        )
    else:
        fixture = run_baseline_evaluation(
            seeds=[args.seed],
            max_steps=max(50, min(args.max_steps, 500)),
            artifact_path=None,
        )
        payload["fixture_baseline_summary"] = {
            name: value["aggregate"] for name, value in fixture["baselines"].items()
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(args.out), "world": world}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
