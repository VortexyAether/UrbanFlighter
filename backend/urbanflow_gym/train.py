from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from .contract import ALGORITHM_CONFIGS, TRAINING_EXTRAS_INSTALL_COMMAND


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a PPO or SAC actor on the UrbanFlow Gym actor-only observation."
    )
    parser.add_argument("--algorithm", choices=("ppo", "sac"), default="ppo")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=17)
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

    env = GymnasiumUrbanFlowEnv()
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
    finally:
        env.close()
    print(
        f"Saved a genuinely trained {algorithm_id.upper()} model to {destination}.zip. "
        "It has not been validated on a real 3D Navier-Stokes field."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
