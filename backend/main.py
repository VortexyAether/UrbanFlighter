from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
import uvicorn

from services.geometry import fetch_buildings
from services.aerojax_demo import build_aerojax_demo_flow
from services.flow_2d import compute_cfd_lite_b_flow_2d, wind_dir_to_inlet_vector
from services.wind import get_real_weather, generate_global_wind_params
from services.simulation_manager import SimulationManager
from urban_flighter_rl.rollout import env_spec_payload, run_deterministic_baseline_rollout
from urbanflow_gym.contract import urbanflow_contract_payload
from urbanflow_gym.evaluation import (
    DEFAULT_ARTIFACT_PATH as URBANFLOW_DEFAULT_ARTIFACT_PATH,
    evaluation_summary,
    load_latest_evaluation,
    run_baseline_evaluation,
    run_live_baseline_evaluation,
)
from urbanflow_gym.live_scenario import (
    LiveScenarioValidationError,
    NoLiveScenarioError,
    UnknownLiveScenarioError,
    build_live_scenario_record,
    live_scenario_registry,
    snapshot_buildings_for_flow,
)
from urbanflow_gym.inspector import (
    INSPECTOR_DEFAULT_MAX_STEPS,
    INSPECTOR_MAX_BATCH_STEPS,
    INSPECTOR_MAX_STEPS,
    StaleInspectorScenarioError,
    UnknownInspectorSessionError,
    inspector_session_manager,
)
from urbanflow_gym.scenario import DEFAULT_HELD_OUT_SEEDS

app = FastAPI(title="Urban Drone Challenge API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sim_manager = SimulationManager(output_root=Path(__file__).parent / "sim_outputs", max_recent=5)

# Production safety limits for the dense 2D solver and its JSON response. The
# current 400 m / 2.5 m cockpit request is 321 x 321 (103,041 cells).
FLOW_FIELD_2D_MAX_GRID_AXIS_CELLS = 384
FLOW_FIELD_2D_MAX_GRID_TOTAL_CELLS = 131_072


def estimate_flow_field_2d_grid_axis_cells(radius_m: float, grid_size_m: float) -> int:
    """Mirror the inclusive ``np.arange`` grid used by the CFD-lite B solver."""
    return int(math.ceil((2.0 * float(radius_m) / float(grid_size_m)) + 0.5))


class SimulationRequest(BaseModel):
    lat: float = 37.451448
    lon: float = 126.6515423
    radius_m: float = Field(default=800.0, gt=50.0)
    nx: int = Field(default=64, ge=16, le=192)
    ny: int = Field(default=64, ge=16, le=192)
    nz: int = Field(default=32, ge=8, le=96)
    voxel_size_m: float = Field(default=10.0, gt=0.5)
    mode: str = "real"  # real | random
    use_real_weather: bool = True
    save_vtk: bool = True


class SampleWindRequest(BaseModel):
    simulation_id: str
    points: list[list[float]]


class FlowField2DRequest(BaseModel):
    lat: float = Field(default=37.451448, ge=-90.0, le=90.0, allow_inf_nan=False)
    lon: float = Field(default=126.6515423, ge=-180.0, le=180.0, allow_inf_nan=False)
    geometry_radius_m: float = Field(default=400.0, gt=50.0, le=1000.0)
    solve_radius_m: float = Field(default=400.0, ge=200.0, le=3000.0)
    grid_size_m: float = Field(default=20.0, gt=0.1, le=80.0)
    use_real_weather: bool = True

    @model_validator(mode="after")
    def reject_impractical_derived_grid(self) -> "FlowField2DRequest":
        if self.geometry_radius_m > self.solve_radius_m:
            raise ValueError("geometry_radius_m cannot exceed solve_radius_m")
        axis_cells = estimate_flow_field_2d_grid_axis_cells(self.solve_radius_m, self.grid_size_m)
        total_cells = axis_cells * axis_cells
        if (
            axis_cells > FLOW_FIELD_2D_MAX_GRID_AXIS_CELLS
            or total_cells > FLOW_FIELD_2D_MAX_GRID_TOTAL_CELLS
        ):
            raise ValueError(
                "derived 2D solver grid "
                f"{axis_cells}x{axis_cells} ({total_cells} cells) exceeds production limits "
                f"of {FLOW_FIELD_2D_MAX_GRID_AXIS_CELLS} cells per axis and "
                f"{FLOW_FIELD_2D_MAX_GRID_TOTAL_CELLS} total cells; increase grid_size_m "
                "or reduce solve_radius_m"
            )
        return self


UrbanFlowSeed = Annotated[int, Field(ge=0, le=2_147_483_647, strict=True)]


class UrbanFlowEvaluationRequest(BaseModel):
    """Strict, short-running public baseline evaluation request."""

    model_config = ConfigDict(extra="forbid", strict=True)

    seeds: list[UrbanFlowSeed] = Field(
        default_factory=lambda: list(DEFAULT_HELD_OUT_SEEDS),
        min_length=1,
        max_length=5,
    )
    max_steps: int = Field(default=360, ge=50, le=500, strict=True)
    save_artifact: bool = False


UrbanFlowLiveCoordinate = Annotated[
    float,
    Field(ge=-3_500.0, le=3_500.0, allow_inf_nan=False),
]


class UrbanFlowLiveEvaluationRequest(BaseModel):
    """Strict, bounded baseline evaluation in one registered live snapshot."""

    model_config = ConfigDict(extra="forbid", strict=True)

    scenario_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=96,
        pattern=r"^urbanflow-live-v1-[0-9a-f]{24}$",
    )
    seeds: list[UrbanFlowSeed] = Field(
        default_factory=lambda: list(DEFAULT_HELD_OUT_SEEDS),
        min_length=1,
        max_length=5,
    )
    max_steps: int = Field(default=360, ge=50, le=500, strict=True)
    start_xy: list[UrbanFlowLiveCoordinate] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
    )
    goal_xy: list[UrbanFlowLiveCoordinate] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
    )
    save_artifact: bool = False

    @model_validator(mode="after")
    def require_complete_mission_override(self) -> "UrbanFlowLiveEvaluationRequest":
        if (self.start_xy is None) != (self.goal_xy is None):
            raise ValueError("start_xy and goal_xy must be provided together")
        return self


class UrbanFlowInspectorCreateRequest(BaseModel):
    """Create one bounded deterministic replay in an exact live scenario."""

    model_config = ConfigDict(extra="forbid", strict=True)

    scenario_id: str = Field(
        min_length=1,
        max_length=96,
        pattern=r"^urbanflow-live-v1-[0-9a-f]{24}$",
    )
    seed: UrbanFlowSeed = 10_007
    baseline: Literal["direct_goal", "shortest_path", "wind_aware_inlet"] = (
        "shortest_path"
    )
    max_steps: int = Field(
        default=INSPECTOR_DEFAULT_MAX_STEPS,
        ge=1,
        le=INSPECTOR_MAX_STEPS,
        strict=True,
    )


UrbanFlowInspectorAction = Annotated[
    float,
    Field(ge=-1.0, le=1.0, allow_inf_nan=False),
]


class UrbanFlowInspectorStepRequest(BaseModel):
    """Bounded sequential steps with an optional repeated actor-space override."""

    model_config = ConfigDict(extra="forbid", strict=True)

    action: list[UrbanFlowInspectorAction] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
    )
    repeat: int = Field(
        default=1,
        ge=1,
        le=INSPECTOR_MAX_BATCH_STEPS,
        strict=True,
    )

    @model_validator(mode="after")
    def reject_boolean_actions(self) -> "UrbanFlowInspectorStepRequest":
        if self.action is not None and any(isinstance(value, bool) for value in self.action):
            raise ValueError("action components must be finite numbers, not booleans")
        return self


def _configured_flow_weather() -> dict[str, Any]:
    return {
        "wind_speed": 5.0,
        "wind_deg": 0.0,
        "description": "Configured deterministic baseline wind",
        "units": {
            "wind_speed": "m/s",
            "wind_deg": "degrees_from_north",
        },
        "source": {
            "provider": "urban-flighter",
            "kind": "configured_baseline",
            "variable_height_m": 10.0,
        },
        "fallback": {
            "used": False,
            "reason": None,
        },
    }


def _flow_weather_response(weather: dict[str, Any]) -> dict[str, Any]:
    """Keep legacy wind keys while always exposing structured provenance."""
    units = weather.get("units")
    source = weather.get("source")
    fallback = weather.get("fallback")
    return {
        "wind_speed": float(weather.get("wind_speed", 5.0)),
        "wind_deg": float(weather.get("wind_deg", 0.0)),
        "description": weather.get("description", "unknown"),
        "units": units if isinstance(units, dict) else {
            "wind_speed": "m/s",
            "wind_deg": "degrees_from_north",
        },
        "source": source if isinstance(source, dict) else {
            "provider": "unknown",
            "kind": "legacy_weather_payload",
        },
        "fallback": fallback if isinstance(fallback, dict) else {
            "used": False,
            "reason": None,
        },
    }


@app.get("/")
def read_root():
    return {"status": "ok", "service": "Urban Drone Challenge Backend"}


@app.get("/health")
def health():
    return {"status": "ok", "service": "UrbanDroneBackend"}


@app.get("/map")
def get_map_data(lat: float, lon: float, radius: float = 300):
    buildings = fetch_buildings(lat, lon, radius)
    if not buildings:
        return {"features": [], "message": "No buildings found or error occurred."}
    return {"features": buildings, "count": len(buildings)}


@app.get("/weather")
def get_weather_data(lat: float, lon: float):
    real_weather = get_real_weather(lat, lon)
    sim_params = generate_global_wind_params()
    return {
        "real": real_weather,
        "simulation": sim_params,
    }


@app.get("/rl/spec")
@app.get("/api/rl/spec")
def get_rl_environment_spec():
    return env_spec_payload()


@app.get("/rl/baseline")
@app.get("/api/rl/baseline")
def get_rl_baseline(seed: int = 7, max_steps: int = 300, n_drones: int = 4, randomize_missions: bool = True):
    bounded_steps = min(max(int(max_steps), 1), 1200)
    bounded_drones = min(max(int(n_drones), 1), 12)
    return run_deterministic_baseline_rollout(
        seed=int(seed),
        max_steps=bounded_steps,
        n_drones=bounded_drones,
        randomize_missions=bool(randomize_missions),
    )


@app.get("/urbanflow-gym/spec")
@app.get("/api/urbanflow-gym/spec")
def get_urbanflow_gym_spec():
    return urbanflow_contract_payload()


@app.post("/urbanflow-gym/evaluate")
@app.post("/api/urbanflow-gym/evaluate")
@app.post("/urbanflow-gym/fixtures/evaluate")
@app.post("/api/urbanflow-gym/fixtures/evaluate")
def evaluate_urbanflow_gym(req: UrbanFlowEvaluationRequest):
    """Backward-compatible explicit synthetic-fixture evaluation."""

    payload = run_baseline_evaluation(
        seeds=req.seeds,
        max_steps=req.max_steps,
        artifact_path=URBANFLOW_DEFAULT_ARTIFACT_PATH if req.save_artifact else None,
    )
    return evaluation_summary(payload)


def _resolve_live_record(scenario_id: str | None = None):
    if scenario_id is not None and (
        len(scenario_id) != len("urbanflow-live-v1-") + 24
        or not scenario_id.startswith("urbanflow-live-v1-")
        or any(character not in "0123456789abcdef" for character in scenario_id[-24:])
    ):
        raise HTTPException(
            status_code=404,
            detail=f"live scenario '{scenario_id}' is invalid, stale, or no longer cached",
        )
    try:
        return live_scenario_registry.get_record(scenario_id)
    except NoLiveScenarioError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except UnknownLiveScenarioError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/urbanflow-gym/live-scenarios/current")
@app.get("/api/urbanflow-gym/live-scenarios/current")
def get_current_live_urbanflow_scenario():
    _resolve_live_record()
    return live_scenario_registry.current_summary()


@app.get("/urbanflow-gym/live-scenarios/{scenario_id}/summary")
@app.get("/api/urbanflow-gym/live-scenarios/{scenario_id}/summary")
def get_live_urbanflow_scenario_summary(scenario_id: str):
    record = _resolve_live_record(scenario_id)
    summary = record.summary()
    try:
        current_id = live_scenario_registry.current_summary()["scenario_id"]
    except NoLiveScenarioError:
        current_id = None
    summary["is_current"] = scenario_id == current_id
    return summary


@app.get("/urbanflow-gym/live-scenarios/{scenario_id}/geometry")
@app.get("/api/urbanflow-gym/live-scenarios/{scenario_id}/geometry")
def get_live_urbanflow_scenario_geometry(scenario_id: str):
    _resolve_live_record(scenario_id)
    return live_scenario_registry.snapshot(scenario_id)


@app.post("/urbanflow-gym/live-scenarios/{scenario_id}/activate")
@app.post("/api/urbanflow-gym/live-scenarios/{scenario_id}/activate")
def activate_live_urbanflow_scenario(scenario_id: str):
    _resolve_live_record(scenario_id)
    try:
        return live_scenario_registry.activate(scenario_id)
    except UnknownLiveScenarioError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/urbanflow-gym/live/evaluate")
@app.post("/api/urbanflow-gym/live/evaluate")
def evaluate_live_urbanflow_gym(req: UrbanFlowLiveEvaluationRequest):
    record = _resolve_live_record(req.scenario_id)
    artifact_path = None
    if req.save_artifact:
        artifact_path = (
            URBANFLOW_DEFAULT_ARTIFACT_PATH.parent
            / f"live_baseline_eval_{record.scenario_id.removeprefix('urbanflow-live-v1-')}.json"
        )
    try:
        payload = run_live_baseline_evaluation(
            record,
            seeds=req.seeds,
            max_steps=req.max_steps,
            artifact_path=artifact_path,
            start_xy=req.start_xy,
            goal_xy=req.goal_xy,
        )
    except LiveScenarioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"live scenario baseline evaluation could not construct a safe route: {exc}",
        ) from exc
    return evaluation_summary(payload)


@app.post("/urbanflow-gym/inspector/sessions")
@app.post("/api/urbanflow-gym/inspector/sessions")
@app.post("/urbanflow-gym/live/inspector/sessions")
@app.post("/api/urbanflow-gym/live/inspector/sessions")
def create_urbanflow_inspector_session(req: UrbanFlowInspectorCreateRequest):
    """Pin a live OSM/current-inlet scenario for deterministic visual replay."""

    _resolve_live_record(req.scenario_id)
    try:
        return inspector_session_manager.create(
            scenario_id=req.scenario_id,
            seed=req.seed,
            baseline_id=req.baseline,
            max_steps=req.max_steps,
        )
    except UnknownLiveScenarioError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LiveScenarioValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"inspector could not construct the pinned live-world baseline: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/urbanflow-gym/inspector/sessions/{session_id}/reset")
@app.post("/api/urbanflow-gym/inspector/sessions/{session_id}/reset")
@app.post("/urbanflow-gym/live/inspector/sessions/{session_id}/reset")
@app.post("/api/urbanflow-gym/live/inspector/sessions/{session_id}/reset")
def reset_urbanflow_inspector_session(session_id: str):
    try:
        return inspector_session_manager.reset(session_id)
    except (UnknownInspectorSessionError, StaleInspectorScenarioError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (LiveScenarioValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"inspector reset could not reconstruct the pinned baseline: {exc}",
        ) from exc


@app.post("/urbanflow-gym/inspector/sessions/{session_id}/step")
@app.post("/api/urbanflow-gym/inspector/sessions/{session_id}/step")
@app.post("/urbanflow-gym/live/inspector/sessions/{session_id}/step")
@app.post("/api/urbanflow-gym/live/inspector/sessions/{session_id}/step")
def step_urbanflow_inspector_session(
    session_id: str,
    req: UrbanFlowInspectorStepRequest | None = None,
):
    try:
        return inspector_session_manager.step(
            session_id,
            None if req is None else req.action,
            repeat=1 if req is None else req.repeat,
        )
    except (UnknownInspectorSessionError, StaleInspectorScenarioError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.delete("/urbanflow-gym/inspector/sessions/{session_id}")
@app.delete("/api/urbanflow-gym/inspector/sessions/{session_id}")
@app.delete("/urbanflow-gym/live/inspector/sessions/{session_id}")
@app.delete("/api/urbanflow-gym/live/inspector/sessions/{session_id}")
def delete_urbanflow_inspector_session(session_id: str):
    try:
        return inspector_session_manager.delete(session_id)
    except UnknownInspectorSessionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/urbanflow-gym/evaluations/latest")
@app.get("/api/urbanflow-gym/evaluations/latest")
def get_latest_urbanflow_gym_evaluation():
    try:
        return evaluation_summary(load_latest_evaluation(URBANFLOW_DEFAULT_ARTIFACT_PATH))
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=(
                "no generated UrbanFlow evaluation artifact exists yet; "
                "POST /urbanflow-gym/evaluate first"
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/simulations")
def create_simulation(req: SimulationRequest):
    rec = sim_manager.submit(req.model_dump())
    return {
        "simulation_id": rec.simulation_id,
        "status": rec.status,
        "wind_json_path": rec.wind_json_path,
        "wind_vtk_path": rec.wind_vtk_path,
        "solid_npy_path": rec.solid_npy_path,
        "error": rec.error,
        "meta": rec.meta,
    }


@app.get("/simulations/{simulation_id}")
def get_simulation(simulation_id: str):
    rec = sim_manager.get(simulation_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="simulation not found")
    return {
        "simulation_id": rec.simulation_id,
        "status": rec.status,
        "wind_json_path": rec.wind_json_path,
        "wind_vtk_path": rec.wind_vtk_path,
        "solid_npy_path": rec.solid_npy_path,
        "error": rec.error,
        "meta": rec.meta,
    }


@app.post("/sample-wind")
def sample_wind(req: SampleWindRequest):
    try:
        velocities = sim_manager.sample(req.simulation_id, req.points)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"velocities": velocities}


@app.post("/flow-fields/2d")
def create_flow_field_2d(req: FlowField2DRequest):
    buildings = fetch_buildings(req.lat, req.lon, req.geometry_radius_m)
    weather = get_real_weather(req.lat, req.lon) if req.use_real_weather else _configured_flow_weather()
    weather_response = _flow_weather_response(weather)
    inlet = wind_dir_to_inlet_vector(float(weather.get("wind_speed", 5.0)), float(weather.get("wind_deg", 0.0)))
    field = compute_cfd_lite_b_flow_2d(buildings, inlet, req.solve_radius_m, req.grid_size_m)
    source = {
        "kind": "POTENTIAL-FLOW CFD-LITE B: streamfunction grid + wall damping/wake correction",
        "model": "potential-flow-cfd-lite-wall-damping-wake",
        "navier_stokes_cfd": False,
    }
    live_scenario = None
    response_buildings = buildings
    if buildings:
        try:
            record = build_live_scenario_record(
                lat=req.lat,
                lon=req.lon,
                geometry_radius_m=req.geometry_radius_m,
                solve_radius_m=req.solve_radius_m,
                buildings=buildings,
                weather=weather_response,
                inlet_velocity_xy=inlet,
                field=field,
                flow_source=source,
            )
            live_scenario = live_scenario_registry.register(record)
            response_buildings = snapshot_buildings_for_flow(record.snapshot())
        except LiveScenarioValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"live UrbanFlow scenario validation failed: {exc}",
            ) from exc
    return {
        "buildings": response_buildings,
        "weather": weather_response,
        "inlet": {
            "ux": float(inlet[0]),
            "uy": float(inlet[1]),
            "speed_mps": float(np.linalg.norm(inlet)),
        },
        "domain": {
            "geometry_radius_m": float(req.geometry_radius_m),
            "solve_radius_m": float(req.solve_radius_m),
        },
        "field": field,
        "source": source,
        "live_scenario": live_scenario,
    }


@app.get("/flow-fields/aerojax-demo")
def get_aerojax_demo_flow(stride: int = 8, snapshot_t: int | None = None):
    try:
        return build_aerojax_demo_flow(stride=stride, snapshot_t=snapshot_t)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
