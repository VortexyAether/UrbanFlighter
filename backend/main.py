from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, Field
import uvicorn

from services.geometry import fetch_buildings
from services.aerojax_demo import build_aerojax_demo_flow
from services.flow_2d import compute_cfd_lite_b_flow_2d, wind_dir_to_inlet_vector
from services.wind import get_real_weather, generate_global_wind_params
from services.simulation_manager import SimulationManager
from urban_flighter_rl.rollout import env_spec_payload, run_deterministic_baseline_rollout

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
    lat: float = 37.451448
    lon: float = 126.6515423
    geometry_radius_m: float = Field(default=400.0, gt=50.0, le=1000.0)
    solve_radius_m: float = Field(default=400.0, ge=200.0, le=3000.0)
    grid_size_m: float = Field(default=20.0, gt=0.1, le=80.0)
    use_real_weather: bool = True


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
    weather = get_real_weather(req.lat, req.lon) if req.use_real_weather else {"wind_speed": 5.0, "wind_deg": 0.0}
    inlet = wind_dir_to_inlet_vector(float(weather.get("wind_speed", 5.0)), float(weather.get("wind_deg", 0.0)))
    field = compute_cfd_lite_b_flow_2d(buildings, inlet, req.solve_radius_m, req.grid_size_m)
    return {
        "buildings": buildings,
        "weather": {
            "wind_speed": float(weather.get("wind_speed", 5.0)),
            "wind_deg": float(weather.get("wind_deg", 0.0)),
            "description": weather.get("description", "unknown"),
        },
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
        "source": {
            "kind": "POTENTIAL-FLOW CFD-LITE B: streamfunction grid + wall damping/wake correction",
            "model": "potential-flow-cfd-lite-wall-damping-wake",
        },
    }


@app.get("/flow-fields/aerojax-demo")
def get_aerojax_demo_flow(stride: int = 8, snapshot_t: int | None = None):
    try:
        return build_aerojax_demo_flow(stride=stride, snapshot_t=snapshot_t)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
