from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from .contract import ALGORITHM_CONFIGS, TRAINING_EXTRAS_INSTALL_COMMAND
from .env import UrbanFlowConfig
from .live_scenario import (
    live_scenario_registry,
    make_live_scenario,
    read_training_bundle,
)


def _require_training_extras() -> None:
    missing = [
        name
        for name in ("gymnasium", "stable_baselines3", "torch")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise SystemExit(
            "UrbanFlow training extras are missing "
            f"({', '.join(missing)}). Install them from the repository root with:\n"
            f"  {TRAINING_EXTRAS_INSTALL_COMMAND}"
        )


def _resolve_fixed_scenario(args: argparse.Namespace):
    if args.world == "fixture":
        return None
    if args.snapshot:
        record = read_training_bundle(args.snapshot)
        return make_live_scenario(record, seed=args.seed)
    if args.live_scenario_id:
        record = live_scenario_registry.get_record(args.live_scenario_id)
        return make_live_scenario(record, seed=args.seed)
    if args.world == "live":
        record = live_scenario_registry.get_record()
        return make_live_scenario(record, seed=args.seed)
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a PPO or SAC actor on the UrbanFlow Gym actor-only observation. "
            "Default world is the offline synthetic fixture. Pass --snapshot to train "
            "on an exported live OSM + inlet bundle."
        )
    )
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default="ppo")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--world",
        choices=("fixture", "live", "snapshot"),
        default="fixture",
    )
    parser.add_argument("--snapshot", type=Path, default=None)
    parser.add_argument("--live-scenario-id", default=None)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/urbanflow_gym/training"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.total_timesteps < 1:
        raise SystemExit("--total-timesteps must be at least 1")
    if not 0 <= args.seed <= 2_147_483_647:
        raise SystemExit("--seed must be between 0 and 2147483647")
    if args.world == "snapshot" and args.snapshot is None:
        raise SystemExit("--world snapshot requires --snapshot PATH")
    _require_training_extras()

    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.env_checker import check_env

        from .gym_adapter import GymnasiumUrbanFlowEnv
    except ImportError as exc:
        raise SystemExit(
            "UrbanFlow training extras were found but could not be imported. "
            "Reinstall them from the repository root with:\n"
            f"  {TRAINING_EXTRAS_INSTALL_COMMAND}\n"
            f"Import error: {exc}"
        ) from exc

    fixed_scenario = _resolve_fixed_scenario(args)
    config = UrbanFlowConfig(max_steps=args.max_steps)
    env = GymnasiumUrbanFlowEnv(config=config, fixed_scenario=fixed_scenario)
    try:
        check_env(env, warn=True)
        algorithm_id = args.algorithm.lower()
        model_class = PPO if algorithm_id == "ppo" else SAC
        model_config = dict(ALGORITHM_CONFIGS[algorithm_id])
        policy = model_config.pop("policy")
        model = model_class(
            policy,
            env,
            seed=args.seed,
            verbose=1,
            **model_config,
        )
        model.learn(total_timesteps=args.total_timesteps)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir / f"urbanflow_{algorithm_id}_seed{args.seed}"
        model.save(destination)
        metadata = {
            "algorithm": algorithm_id,
            "seed": args.seed,
            "total_timesteps": args.total_timesteps,
            "world": args.world,
            "snapshot": None if args.snapshot is None else str(args.snapshot),
            "live_scenario_id": args.live_scenario_id,
            "observation_dim": int(env.observation_space.shape[0]),
            "full_flow_access": False,
            "trained_policy_validated_on_navier_stokes": False,
        }
        (args.output_dir / f"{destination.name}.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
    finally:
        env.close()
    print(
        f"Saved a genuinely trained {algorithm_id.upper()} model to {destination}.zip. "
        "It has not been validated on a real 3D Navier-Stokes field."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
