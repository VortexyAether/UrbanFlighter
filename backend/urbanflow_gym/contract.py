from __future__ import annotations

import importlib.util

from .evaluation import DEFAULT_ARTIFACT_PATH
from .schemas import (
    CONTRACT_VERSION,
    ENVIRONMENT_ID,
    action_contract,
    actor_observation_contract,
    leakage_guard_report,
    metrics_contract,
    privileged_critic_contract,
    reward_contract,
)


TRAINING_EXTRAS_INSTALL_COMMAND = (
    ".venv/bin/python -m pip install -r backend/requirements-urbanflow-train.txt"
)

ALGORITHM_CONFIGS = {
    "ppo": {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "gamma": 0.99,
        "gae_lambda": 0.95,
    },
    "sac": {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "batch_size": 256,
        "gamma": 0.99,
        "learning_starts": 2_000,
    },
}


def urbanflow_contract_payload() -> dict:
    gymnasium_installed = importlib.util.find_spec("gymnasium") is not None
    sb3_installed = importlib.util.find_spec("stable_baselines3") is not None
    torch_installed = importlib.util.find_spec("torch") is not None
    return {
        "contract_version": CONTRACT_VERSION,
        "environment_id": ENVIRONMENT_ID,
        "status": {
            "implemented": True,
            "rl_ready": True,
            "primary_environment": "registered_live_osm_current_inlet_snapshot",
            "training_environment_ready": True,
            "training_executed_on_this_machine": False,
            "trained_policy_available": False,
            "real_3d_navier_stokes_validated": False,
        },
        "labels": {
            "primary": "URBANFLOW GYM / LIVE OSM WORLD · NOT TRAINED",
            "world": "LIVE OSM WORLD",
            "full_flow_access": "NO",
            "hidden_flow": "synthetic CFD-lite grid registered from the same UI flow response",
            "action": "bounded local-guidance command",
            "real_cfd_boundary": (
                "future user-supplied external 3D CFD dataset adapter; no dataset run or "
                "validation in this slice"
            ),
        },
        "actor_observation": actor_observation_contract(),
        "privileged_critic_state": privileged_critic_contract(),
        "action": action_contract(),
        "reward": reward_contract(),
        "episode_metrics": metrics_contract(),
        "leakage_guard": leakage_guard_report(),
        "dynamics": {
            "training_provider_interface": "urbanflow_gym.wind.WindProvider",
            "primary_provider": "RegisteredFlowGridWindProvider",
            "primary_provider_claim": (
                "read-only browser-resolved CFD-lite grid for the registered real OSM/current-inlet snapshot; "
                "not Navier-Stokes CFD"
            ),
            "synthetic_rectangular_provider": "explicit fixture-only compatibility path",
            "full_spatial_field_hidden_from_actor": True,
            "direct_provider_state_hidden_from_actor": True,
            "onboard_relative_air_velocity_estimate_available": True,
            "relative_air_velocity_formula": "ground_velocity - known_inlet",
            "hidden_relative_air_used_for": ["dynamics", "reward", "episode_metrics"],
            "drag_model_id": "quadratic-air-relative-v1",
            "drag_model_shared_with_cockpit": True,
            "hidden_field_allowed_uses": [
                "dynamics",
                "reward",
                "episode_metrics",
                "optional_privileged_critic",
            ],
        },
        "future_external_cfd_evaluation": {
            "dataset_protocol": "urbanflow_gym.cfd_adapter.CFDFieldDataset",
            "adapter": "urbanflow_gym.cfd_adapter.CFDWindProvider2DAdapter",
            "structured_npz_adapter": "urbanflow_gym.cfd_adapter.StructuredCFDDataset",
            "actor_schema_unchanged": True,
            "zero_shot_semantics": (
                "A frozen actor must receive the same v1 actor observation while the hidden "
                "wind provider is replaced by an unseen offline 3D field."
            ),
            "status": "interface_only_awaiting_user_supplied_dataset",
            "validation_claim": "none",
        },
        "runtime_dependencies": {
            "core": ["numpy"],
            "rendering": [],
            "gymnasium_required_for_core": False,
            "gymnasium_installed": gymnasium_installed,
            "stable_baselines3_installed": sb3_installed,
            "torch_installed": torch_installed,
        },
        "training": {
            "status": "not_run_on_mac_mini",
            "requires_exported_or_registered_live_snapshot": True,
            "entrypoint": "urbanflow_gym.train; fixture by default, optional live snapshot",
            "ppo_command": (
                "PYTHONPATH=backend .venv/bin/python -m urbanflow_gym.train "
                "--algorithm ppo --total-timesteps 100000 --world fixture"
            ),
            "sac_command": (
                "PYTHONPATH=backend .venv/bin/python -m urbanflow_gym.train "
                "--algorithm sac --total-timesteps 100000 --world fixture"
            ),
            "eval_command": (
                "PYTHONPATH=backend .venv/bin/python -m urbanflow_gym.eval_policy "
                "--checkpoint results/urbanflow_gym/training/urbanflow_ppo_seed17.zip"
            ),
            "algorithms": ALGORITHM_CONFIGS,
            "extras_install_command": TRAINING_EXTRAS_INSTALL_COMMAND,
            "bundled_weights": False,
            "asymmetric_critic_used_by_entrypoint": False,
        },
        "evaluation": {
            "primary_api": "/urbanflow-gym/live/evaluate",
            "requires_live_scenario": True,
            "synthetic_fixture_api": "/urbanflow-gym/fixtures/evaluate",
            "legacy_synthetic_alias": "/urbanflow-gym/evaluate",
            "default_artifact_path": str(
                DEFAULT_ARTIFACT_PATH.relative_to(DEFAULT_ARTIFACT_PATH.parents[2])
            ),
            "baselines": ["direct_goal", "shortest_path", "wind_aware_inlet"],
            "api_limits": {"max_seeds": 5, "min_steps": 50, "max_steps": 500},
            "api_response": "aggregate summary; trajectories remain in optional JSON artifact",
        },
        "compatibility": {
            "legacy_rl_spec_preserved": "/rl/spec and /api/rl/spec",
            "versioned_endpoints": [
                "/urbanflow-gym/spec",
                "/urbanflow-gym/live-scenarios/current",
                "/urbanflow-gym/live/evaluate",
                "/urbanflow-gym/fixtures/evaluate",
                "/urbanflow-gym/evaluations/latest",
            ],
        },
    }
