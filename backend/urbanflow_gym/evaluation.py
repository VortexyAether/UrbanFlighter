from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Callable, Iterable, Mapping

import numpy as np

from .baselines import BASELINE_ORDER, make_baseline
from .cfd_adapter import CFDFieldDataset, CFDWindProvider2DAdapter
from .env import UrbanFlowConfig, UrbanFlowEnv
from .live_scenario import (
    LiveScenarioRecord,
    live_scenario_summary,
    make_live_scenario,
)
from .scenario import DEFAULT_HELD_OUT_SEEDS, UrbanFlowScenario, make_seeded_scenario
from .schemas import (
    ACTION_SCHEMA_ID,
    ACTOR_OBSERVATION_SCHEMA_ID,
    CONTRACT_VERSION,
    ENVIRONMENT_ID,
    METRICS_SCHEMA_ID,
    leakage_guard_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_PATH = REPO_ROOT / "results" / "urbanflow_gym" / "baseline_eval_v1.json"
MAX_API_SEEDS = 5
MIN_EVALUATION_STEPS = 50
MAX_EVALUATION_STEPS = 500


def validate_evaluation_inputs(seeds: Iterable[int], max_steps: int) -> tuple[tuple[int, ...], int]:
    normalized = tuple(int(seed) for seed in seeds)
    if not 1 <= len(normalized) <= MAX_API_SEEDS:
        raise ValueError(f"seeds must contain between 1 and {MAX_API_SEEDS} values")
    if any(seed < 0 or seed > 2_147_483_647 for seed in normalized):
        raise ValueError("each seed must be between 0 and 2147483647")
    bounded_steps = int(max_steps)
    if not MIN_EVALUATION_STEPS <= bounded_steps <= MAX_EVALUATION_STEPS:
        raise ValueError(
            f"max_steps must be between {MIN_EVALUATION_STEPS} and {MAX_EVALUATION_STEPS}"
        )
    return normalized, bounded_steps


def run_baseline_evaluation(
    *,
    seeds: Iterable[int] = DEFAULT_HELD_OUT_SEEDS,
    max_steps: int = 360,
    artifact_path: str | Path | None = DEFAULT_ARTIFACT_PATH,
    cfd_dataset: CFDFieldDataset | None = None,
) -> dict:
    normalized_seeds, bounded_steps = validate_evaluation_inputs(seeds, max_steps)
    source = _evaluation_source(cfd_dataset)
    return _run_baseline_evaluation(
        normalized_seeds=normalized_seeds,
        bounded_steps=bounded_steps,
        artifact_path=artifact_path,
        source=source,
        scenario_for_seed=lambda seed: _scenario_for_source(seed, cfd_dataset),
        scenario_kind="synthetic_fixture",
        scenario_identity={
            "fixture_schema": "urbanflow.synthetic_rectangular_fixture.v1",
            "seeds": list(normalized_seeds),
        },
        live_snapshot_summary=None,
    )


def run_live_baseline_evaluation(
    record: LiveScenarioRecord,
    *,
    seeds: Iterable[int] = DEFAULT_HELD_OUT_SEEDS,
    max_steps: int = 360,
    artifact_path: str | Path | None = None,
    start_xy: Iterable[float] | None = None,
    goal_xy: Iterable[float] | None = None,
) -> dict:
    """Evaluate baselines in one immutable UI-registered OSM/current-inlet world."""

    normalized_seeds, bounded_steps = validate_evaluation_inputs(seeds, max_steps)
    snapshot = record.snapshot()
    summary = live_scenario_summary(snapshot)
    scenarios: dict[int, UrbanFlowScenario] = {}

    def scenario_for_seed(seed: int) -> UrbanFlowScenario:
        if seed not in scenarios:
            scenarios[seed] = make_live_scenario(
                record,
                seed=seed,
                start_xy=start_xy,
                goal_xy=goal_xy,
            )
        return scenarios[seed]

    first_scenario = scenario_for_seed(normalized_seeds[0])
    source = {
        **record.flow_field.source_metadata(),
        "purpose": "live_osm_current_inlet_baseline_evaluation",
        "offline_3d_dataset_run": False,
        "real_cfd_validation_status": "not_run_adapter_only",
        "claim": (
            "Synthetic CFD-lite hidden grid from the registered real OSM geometry and "
            "backend-fetched inlet; no Navier-Stokes CFD or real-CFD validation was run."
        ),
    }
    return _run_baseline_evaluation(
        normalized_seeds=normalized_seeds,
        bounded_steps=bounded_steps,
        artifact_path=artifact_path,
        source=source,
        scenario_for_seed=scenario_for_seed,
        scenario_kind="live_osm_current_inlet",
        scenario_identity={
            "scenario_id": snapshot["scenario_id"],
            "content_hash_sha256": snapshot["content_hash_sha256"],
            "start_xy_m": first_scenario.start_xy.tolist(),
            "goal_xy_m": first_scenario.goal_xy.tolist(),
        },
        live_snapshot_summary=summary,
    )


def _run_baseline_evaluation(
    *,
    normalized_seeds: tuple[int, ...],
    bounded_steps: int,
    artifact_path: str | Path | None,
    source: dict,
    scenario_for_seed: Callable[[int], UrbanFlowScenario],
    scenario_kind: str,
    scenario_identity: dict,
    live_snapshot_summary: dict | None,
) -> dict:
    leakage = leakage_guard_report()
    config = replace(UrbanFlowConfig(), max_steps=bounded_steps)
    results: dict[str, dict] = {}

    for baseline_id in BASELINE_ORDER:
        episodes: list[dict] = []
        for seed in normalized_seeds:
            scenario = scenario_for_seed(seed)
            env = UrbanFlowEnv(config=config, fixed_scenario=scenario)
            observation, reset_info = env.reset(seed=seed)
            del observation
            if reset_info["policy_had_privileged_flow_access"] is not False:
                raise RuntimeError("actor leakage invariant failed during evaluation reset")
            controller = make_baseline(baseline_id, scenario.public_context(), config)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = controller.action(env.actor_snapshot())
                _, _, terminated, truncated, step_info = env.step(action)
                if step_info["policy_had_privileged_flow_access"] is not False:
                    raise RuntimeError("actor leakage invariant failed during evaluation step")
            metrics = env.metrics()
            episodes.append(
                {
                    "scenario": scenario.to_public_dict(),
                    "baseline_id": baseline_id,
                    "uses_hidden_flow": False,
                    "waypoints_xy": [point.tolist() for point in controller.waypoints],
                    "metrics": metrics,
                    "trajectory": env.trajectory,
                }
            )
        results[baseline_id] = {
            "baseline_id": baseline_id,
            "label": make_baseline(
                baseline_id,
                scenario_for_seed(normalized_seeds[0]).public_context(),
                config,
            ).label,
            "uses_hidden_flow": False,
            "allowed_inputs": (
                ["own_state", "goal_context"]
                if baseline_id == "direct_goal"
                else ["own_state", "goal_context", "known_static_geometry"]
                if baseline_id == "shortest_path"
                else [
                    "own_state",
                    "goal_context",
                    "known_static_geometry",
                    "known_inlet_velocity",
                ]
            ),
            "aggregate": _aggregate_metrics([episode["metrics"] for episode in episodes]),
            "episodes": episodes,
        }

    identity_payload = {
        "version": CONTRACT_VERSION,
        "seeds": list(normalized_seeds),
        "max_steps": bounded_steps,
        "source": source,
        "scenario_kind": scenario_kind,
        "scenario_identity": scenario_identity,
    }
    evaluation_id = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    payload = {
        "artifact_schema_id": "urbanflow.baseline_evaluation.v1",
        "contract_version": CONTRACT_VERSION,
        "evaluation_id": evaluation_id,
        "status": "ok",
        "environment_id": ENVIRONMENT_ID,
        "scenario_kind": scenario_kind,
        "scenario_id": scenario_identity.get("scenario_id"),
        "scenario_identity": scenario_identity,
        "live_scenario": live_snapshot_summary,
        "policy_status": "not_trained",
        "real_cfd_validation_status": source["real_cfd_validation_status"],
        "real_cfd_validation_run": False,
        "synthetic_hidden_flow": cfd_dataset_is_synthetic(source),
        "actor_observation_schema_id": ACTOR_OBSERVATION_SCHEMA_ID,
        "action_schema_id": ACTION_SCHEMA_ID,
        "metrics_schema_id": METRICS_SCHEMA_ID,
        "policy_had_privileged_flow_access": False,
        "policy_full_flow_access": False,
        "leakage_guard": leakage,
        "evaluation_config": {
            "seeds": list(normalized_seeds),
            "split": (
                "same_immutable_live_world_caller_selected_seeds"
                if scenario_kind == "live_osm_current_inlet"
                else "synthetic_fixture_held_out_seed_namespace"
                if all(seed >= 10_000 for seed in normalized_seeds)
                else "synthetic_fixture_caller_selected"
            ),
            "max_steps": bounded_steps,
            "dt_s": config.dt_s,
            "baselines": list(BASELINE_ORDER),
        },
        "dynamics_source": source,
        "baselines": results,
        "artifact_path": _display_path(Path(artifact_path)) if artifact_path is not None else None,
    }
    if artifact_path is not None:
        write_evaluation_artifact(payload, artifact_path)
    return payload


def cfd_dataset_is_synthetic(source: Mapping[str, object]) -> bool:
    return bool(source.get("synthetic_hidden_flow", source.get("kind") != "offline_3d_cfd_dataset_adapter"))


def write_evaluation_artifact(payload: dict, artifact_path: str | Path) -> Path:
    destination = Path(artifact_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_latest_evaluation(artifact_path: str | Path = DEFAULT_ARTIFACT_PATH) -> dict:
    path = Path(artifact_path).expanduser().resolve()
    expected_root = (REPO_ROOT / "results" / "urbanflow_gym").resolve()
    if expected_root not in path.parents:
        raise ValueError("evaluation artifact path is outside the generated UrbanFlow result directory")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("artifact_schema_id") != "urbanflow.baseline_evaluation.v1":
        raise ValueError("latest evaluation artifact has an unsupported schema")
    return payload


def evaluation_summary(payload: dict) -> dict:
    return {
        "artifact_schema_id": payload["artifact_schema_id"],
        "contract_version": payload["contract_version"],
        "status": payload["status"],
        "environment_id": payload["environment_id"],
        "scenario_kind": payload["scenario_kind"],
        "scenario_id": payload["scenario_id"],
        "scenario_identity": payload["scenario_identity"],
        "live_scenario": payload["live_scenario"],
        "evaluation_id": payload["evaluation_id"],
        "artifact_path": payload["artifact_path"],
        "policy_status": payload["policy_status"],
        "real_cfd_validation_status": payload["real_cfd_validation_status"],
        "real_cfd_validation_run": payload["real_cfd_validation_run"],
        "synthetic_hidden_flow": payload["synthetic_hidden_flow"],
        "policy_had_privileged_flow_access": payload[
            "policy_had_privileged_flow_access"
        ],
        "policy_full_flow_access": payload["policy_full_flow_access"],
        "evaluation_config": payload["evaluation_config"],
        "dynamics_source": payload["dynamics_source"],
        "baselines": {
            baseline_id: {
                "baseline_id": baseline["baseline_id"],
                "label": baseline["label"],
                "uses_hidden_flow": baseline["uses_hidden_flow"],
                "allowed_inputs": baseline["allowed_inputs"],
                "aggregate": baseline["aggregate"],
            }
            for baseline_id, baseline in payload["baselines"].items()
        },
        "metrics": {
            baseline_id: baseline["aggregate"]
            for baseline_id, baseline in payload["baselines"].items()
        },
    }


def _aggregate_metrics(metrics: list[dict]) -> dict:
    episode_count = len(metrics)
    successes = sum(int(metric["success"]) for metric in metrics)
    collisions = sum(int(metric["collision_count"]) for metric in metrics)
    return {
        "episodes": episode_count,
        "success_count": successes,
        "success_rate": successes / episode_count,
        "collision_count": collisions,
        "collision_episode_rate": sum(
            int(metric["collision_count"] > 0) for metric in metrics
        )
        / episode_count,
        "mean_path_length_m": fmean(float(metric["path_length_m"]) for metric in metrics),
        "mean_relative_air_speed_energy": fmean(
            float(metric["relative_air_speed_energy"]) for metric in metrics
        ),
        "mean_time_s": fmean(float(metric["time_s"]) for metric in metrics),
        "min_clearance_m": min(float(metric["min_clearance_m"]) for metric in metrics),
        "mean_score": fmean(float(metric["score"]) for metric in metrics),
    }


def _scenario_for_source(seed: int, cfd_dataset: CFDFieldDataset | None) -> UrbanFlowScenario:
    scenario = make_seeded_scenario(seed, randomize_domain=True)
    if cfd_dataset is None:
        return scenario
    provider = CFDWindProvider2DAdapter(
        cfd_dataset,
        flight_altitude_m=scenario.geometry.flight_altitude_m,
    )
    return UrbanFlowScenario(
        scenario_id=f"{scenario.scenario_id}-offline-cfd",
        seed=scenario.seed,
        geometry=scenario.geometry,
        start_xy=scenario.start_xy,
        goal_xy=scenario.goal_xy,
        initial_heading_rad=scenario.initial_heading_rad,
        known_inlet_velocity_xy=provider.inlet_velocity,
        wind_provider=provider,
        randomization={
            **scenario.randomization,
            "wind_replaced_by_offline_cfd_adapter": True,
        },
    )


def _evaluation_source(cfd_dataset: CFDFieldDataset | None) -> dict:
    if cfd_dataset is None:
        return {
            "kind": "synthetic_potential_flowish_wake_proxy",
            "purpose": "explicit_synthetic_unit_test_fixture_only",
            "navier_stokes_cfd": False,
            "synthetic_hidden_flow": True,
            "offline_3d_dataset_run": False,
            "real_cfd_validation_status": "not_run_interface_only",
            "claim": "Synthetic held-out evaluation only; no real 3D Navier-Stokes field was run.",
        }
    metadata = cfd_dataset.dataset_metadata()
    declared_ns3d = metadata.get("solver_family") == "navier_stokes_3d"
    return {
        "kind": "offline_3d_cfd_dataset_adapter",
        "purpose": "future_external_dataset_evaluation_boundary",
        "navier_stokes_cfd": False,
        "dataset_declares_navier_stokes_3d": bool(declared_ns3d),
        "provenance_verified_by_urbanflow": False,
        "offline_3d_dataset_run": True,
        "real_cfd_validation_status": "external_dataset_executed_not_independently_validated",
        "dataset": metadata,
        "claim": (
            "The supplied external dataset was sampled through the hidden-field adapter. "
            "UrbanFlow does not verify its solver family, provenance, or physical validity."
        ),
    }


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)
