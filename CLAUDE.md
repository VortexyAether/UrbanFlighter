# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Urban Flighter is a full-stack interactive drone simulator for urban environments. A pilot flies a drone through a real city's building layout while a precomputed/analytical wind field affects flight dynamics and energy consumption. Three moving parts:

- **Backend** (`backend/`): Python FastAPI server. Fetches real OSM building geometry + Open-Meteo forecast-model current conditions, computes wind fields (CFD-lite 2D and voxel 3D), and serves precomputed AeroJAX CFD snapshots to the frontend.
- **Frontend** (`frontend/`): React 19 + TypeScript + Vite. A hybrid simulator with three public modes — **2D** (default), **3D Lite**, and **True 3D Wind**. Drone physics, sensor returns, rolling display maps, and the energy model run client-side.
- **AeroJAX_map_test2** (`AeroJAX_map_test2/`): Standalone JAX-based 2D incompressible Navier–Stokes solver. Produces the real-CFD wind snapshots that the backend serves. This is the active high-fidelity CFD engine.

> **Note:** `urbanFlowGen/` is referenced in older docs but currently contains no source files (only a `.venv` and `__pycache__`). The geometry/grid generator scripts described in past versions of this file no longer exist in the repo. **AeroJAX_map_test2 has replaced it** as the CFD pipeline. Do not document or invoke the old `geometryGenerator.py` / `gridGenerator.py` commands.

## Quick Start Commands

### Backend
```bash
cd backend
../.venv/bin/python main.py       # Runs on http://localhost:8000 (uvicorn, reload=True)
```
No API keys required — Open-Meteo (weather) and OSMnx (buildings) are both free.

### Frontend
```bash
cd frontend
npm install                       # First time only
npm run dev                       # Vite dev server on http://localhost:5173
npm run build                     # tsc -b && vite build
npm run lint                      # ESLint check
npm run preview                   # Preview production build
```

### AeroJAX CFD solver (offline, generates Real-mode data)
```bash
cd AeroJAX_map_test2
python run_map.py --check         # Build SDF/mask + diagnostic plots, no solve
python run_map.py --smoke         # Short coarse-grid sanity run (352x256)
python run_map.py --series        # Full transient solve to t=1000, writes snapshot_t*.npz (default)
JAX_PLATFORMS=cpu python run_map.py --series   # Force CPU (GPU/CUDA is default)
python fetch_buildings.py --point <lat> <lon> --dist <m>   # Refresh OSM building input
```
Outputs land in `AeroJAX_map_test2/outputs/` (`snapshot_t*.npz`, figures, vorticity GIF). The backend reads these snapshots directly via `backend/services/aerojax_demo.py`.

## Architecture & Data Flow

### The three simulation modes (frontend)
`App.tsx` holds `simulationMode: '2d' | '3d' | 'true3d'` and renders accordingly.

- **2D** (default, `TopDownGame.tsx`): HTML5 canvas, north-up flight over OSM footprints. It samples the backend `field.ux/uy` grid, advances motion at a fixed 120 Hz with swept-circle building clearance, and emits a deterministic 180-ray simulator scan. WASD controls.
- **3D Lite** (`Simulation3D.tsx` → `Aircraft.tsx`, `CityModel.tsx`, `CFDLiteWindLayer.tsx`): Three.js/react-three-fiber city flight. Aircraft motion advances through the pure 120 Hz `simulation/flight3dMotion.ts` core, samples the same backend 2D grid, sweeps a labelled live-world safety envelope against OSM building prisms, and uses the loaded rectangular field bounds. The sensor view adds deterministic physical-building and ground returns.
- **True 3D Wind** (`Simulation3D.tsx`, `True3DWindStreamlines.tsx`): overlays a bundled Gangnam U/V/W potential-flow streamline visualization on the same 3D flight presentation. The flyable aircraft still uses the loaded horizontal CFD-lite grid; do not describe the bundled visualization as U/V/W flight dynamics or full CFD.

The compact 3D quad is 0.58 m across. Its visual radius is not the collision radius: browser collision uses the registered live-world 1.25 m horizontal radius (with the same labelled fallback) and retains the pre-existing 2.0 m vertical roof clearance. This anisotropic research envelope is shown in the neutral view and stated in the HUD. `simulation/flight3dControls.ts` owns Arcade/Pilot mappings; 2D keyboard behavior is separate and unchanged.

`presentation/urbanDressing.ts` produces capped trees, streetlights, inferred road treatment, rooftop units, and facade panels from scenario/location/building identities. `presentation/freeFlightBeacon.ts` produces one display-only waypoint. `UrbanDressing.tsx` renders repeated props with instancing. These objects are deliberately absent from the building collision meshes, browser LiDAR input, flow sampler, rolling maps, Gym bridge, observations, reward, APIs, and scenario registration/hash. They are not claims about real OSM vegetation/roads. Keep that mechanical separation and the on-screen legend if extending presentation.

The rolling sensor-map windows use bounded display-only scan history and simulator odometry. They are not hardware-qualified LiDAR, localization, loop closure, pose-graph optimization, or production SLAM. Keep the exact frontend sensor observation at five values per ray: local direction x/y/z, normalized distance, hit flag. This is distinct from the backend RL environment's 18-scalar planner observation.

### Backend wind pipeline
Three independent solvers live in `backend/services/`, chosen by endpoint:
- `flow_2d.py` — potential-flow CFD-lite B streamfunction grid with wall damping and empirical wake correction. It is not full CFD. Backs `/flow-fields/2d`, the primary endpoint the frontend uses.
- `cfd.py` — lightweight 3D voxel wind field (distance-transform obstacle shadow + near-wall deflection). Backs the async `/simulations` job system.
- `aerojax_demo.py` — loads/downsamples precomputed AeroJAX NPZ snapshots from `AeroJAX_map_test2/outputs/`. Backs `/flow-fields/aerojax-demo`. **Reads an absolute path to the AeroJAX directory — see Gotchas.**

`geometry.py` (OSM buildings via OSMnx → UTM-projected footprints + heights) and `wind.py` (Open-Meteo forecast-model current 10 m wind, with deterministic labeled fallback) feed all three.

### UrbanFlow Gym live bridge
Every successful non-empty `/flow-fields/2d` response is canonicalized and atomically registered as `urbanflow.live_scenario.v1`. The UI and headless environment share that content-addressed snapshot: selected location/radii, local east/north metre frame, bounds, OSM polygon exteriors and heights, inlet source/timestamp/vector, solid-boundary polygon collision/LiDAR semantics, and the resolved synthetic CFD-lite grid identity. The actor never receives the full grid. `/urbanflow-gym/live/evaluate` and the bounded episode-inspector sessions never fall back to a toy world; the rectangular/random routes are explicit test fixtures only.

The docked Gym Episode Inspector replays deterministic direct-goal, shortest-path, or inlet-aware baseline actions in one pinned headless environment. Its inspector-only horizon defaults to 1,200 steps (300 simulated seconds) and is capped at 1,600 without changing the core environment default. Its create/reset/step/delete sessions are process-local, capped, TTL/LRU-cleaned, and removed after the terminal frame. Play uses a 4 Hz visual scheduler and bounded sequential batches to target 1–64 simulated steps/second without overlapping HTTP requests. The UI renders the registered OSM polygons, actor LiDAR, own trajectory, pose, and action north-up, plus `dt`, simulated time, batch pacing, goal distance, and a straight-line minimum-time estimate. Exact clearance/reward are explicitly debug readouts, not actor inputs; no flow/grid/wake arrays cross the inspector API. No PPO/SAC training ran on the Mac mini, no browser training/motor loop exists, and the real-CFD adapter remains interface-only for future user data.

### Async simulation jobs (`simulation_manager.py`)
The `/simulations` endpoints are **poll-based, not websocket**. POST queues a job (single-worker `ThreadPoolExecutor`), status transitions `queued → running → done/error`, client polls `GET /simulations/{id}`. Results persist to `backend/sim_outputs/{id}/` as `wind_field.json/.npy/.vtk` + `solid.npy`. An LRU cache keeps only the 5 most recent simulations on disk.

### Coordinate systems
CFD grids are X-forward; the game engine is Y-up. The mapping (CFD x/y/z → game X/Z/Y) is handled in `cfd.py:sample_wind_trilinear`. Meteorological wind direction is degrees-from-north (0° = wind *from* north → flow south); conversion to inlet velocity vectors lives in `flow_2d.py` and `cfd.py`.

## API Endpoints

| Endpoint | Method | Key params | Description |
|----------|--------|------------|-------------|
| `/`, `/health` | GET | — | Health checks |
| `/map` | GET | `lat`, `lon`, `radius` | OSM building footprints (GeoJSON) — legacy, not in main frontend flow |
| `/weather` | GET | `lat`, `lon` | Live weather + procedural wind params — legacy |
| `/flow-fields/2d` | POST | `lat`, `lon`, `geometry_radius_m`, `solve_radius_m`, `grid_size_m`, `use_real_weather` | **Primary**: buildings + analytical 2D velocity grid + weather |
| `/urbanflow-gym/live-scenarios/current` | GET | — | Current registered live-world summary; 409 until a location loads successfully |
| `/urbanflow-gym/live-scenarios/{id}/geometry` | GET | `scenario_id` | Bounded canonical snapshot geometry/provenance |
| `/urbanflow-gym/live/evaluate` | POST | `scenario_id`, `seeds`, `max_steps`, optional bounded start/goal | Primary aggregate baseline evaluation in the specified live world |
| `/urbanflow-gym/fixtures/evaluate` | POST | `seeds`, `max_steps` | Synthetic rectangular unit-test fixture only |
| `/urbanflow-gym/inspector/sessions` | POST | `scenario_id`, `seed`, `baseline`, `max_steps` | Create a bounded deterministic visual replay in the exact registered live world |
| `/urbanflow-gym/inspector/sessions/{id}/reset` | POST | `session_id` | Reset the pinned seed/baseline environment |
| `/urbanflow-gym/inspector/sessions/{id}/step` | POST | optional bounded actor action, `repeat` 1–64 | Execute sequential steps under the session lock, stop at terminal state, and return the final compact visual frame plus executed count |
| `/urbanflow-gym/inspector/sessions/{id}` | DELETE | `session_id` | Explicitly clean up a replay session |
| `/flow-fields/aerojax-demo` | GET | `stride` (default 8), `snapshot_t` | Real-mode AeroJAX CFD snapshot |
| `/simulations` | POST | SimulationRequest | Queue async 3D voxel CFD job |
| `/simulations/{id}` | GET | `simulation_id` | Poll job status |
| `/sample-wind` | POST | `simulation_id`, `points` | Trilinear-interpolated wind at points |

## Energy / Physics Model

Client-side, shared by all modes. Core math in `frontend/src/utils/dragEnergy.ts`; `energySystem.ts` wraps it. Total power = hover + sensors + drag + induced-rotor + climb + slow-flight penalty, where drag scales with relative-air-speed² (`F = 0.5·ρ·Cd·A·v²`). Wind alignment angle classifies flight as COUNTER (headwind, >120°), CROSS, or TAIL (<60°). Key constants (ρ=1.225, Cd=1.05, A=0.18 m², hover=68 W, optimal cruise=11 m/s) are at the top of `dragEnergy.ts`. `EnergyGraph.tsx` plots burn history; `MissionIntelligence.tsx` derives headwind alerts and endurance estimates.

## Code Style

### Python (backend, AeroJAX)
- `snake_case` for variables/functions; `os.path.join()` for paths
- Config via TOML (`import toml`) where present

### TypeScript/React (frontend)
- `camelCase` variables/functions, `PascalCase` components
- TypeScript interfaces for all backend data shapes (mirror them in `api.ts`)
- Functional components with hooks; physics in `useFrame`/canvas loops

## Configuration

- **Default location**: Incheon / Inha, Korea (lat 37.451448, lon 126.6515423) — set in `frontend/src/appModel.ts` and `AeroJAX_map_test2/config_map.py`. (Older docs say Gangnam, Seoul — that is stale.)
- AeroJAX grid/domain/flow params: `AeroJAX_map_test2/config_map.py` (2016×1280 cells @ dx=0.9 m, west wind, U_ref=1.0 m/s, T_end=1000 s).
- Wind units: m/s (speed), degrees 0–360 from north (direction).

## Testing

- **Backend RL contract**: `../.venv/bin/python test_rl_contract.py` from `backend/`.
- **Backend flow/weather contract**: `../.venv/bin/python test_flow_weather_contract.py` from `backend/`.
- **Backend UrbanFlow core/API/live bridge/inspector**: `../.venv/bin/python test_urbanflow_gym_core.py`, `test_urbanflow_gym_api.py`, `test_urbanflow_live_scenario.py`, and `test_urbanflow_episode_inspector.py` from `backend/`.
- **Frontend sensor contract**: `npm run smoke:lidar` from `frontend/`.
- **Frontend flight core**: `npm run smoke:flight-core`.
- **Frontend True 3D coordinates**: `npm run smoke:true3d`.
- **Frontend request/cache behavior**: `npm run smoke:data-flow`.
- **Frontend live-world identity/stale guard**: `npm run smoke:live-world`.
- **Frontend episode-inspector transforms/render/stale guard**: `npm run smoke:episode-inspector`.
- **Frontend render determinism/history bounds**: `npm run smoke:render-determinism`.
- **Frontend 3D presentation isolation/placement**: `npm run smoke:presentation`.
- **Frontend 3D controls/fixed-step/scale/safety**: `npm run smoke:flight3d`.
- **Frontend quality gates**: `npm run lint` must complete with zero errors; `npm run build` must complete successfully.
- **Manual/browser**: exercise all three public modes, keyboard controls, window drag/resize/hide/reset, sensor-map clear/zoom/orbit, Gym Inspector reset/step/play/delete and location-stale cleanup, and backend reload/fallback labeling.
- **3D manual/browser**: in both 3D Lite and True 3D Wind, verify Arcade/Pilot input, W motion, C Chase/Orbit, presentation/research toggle, readable compact drone, physical-only LiDAR cloud, safety-envelope label, and no console/page/local-request failures. Development StrictMode can abort superseded fetch effects during cleanup; a succeeding request must still complete.
- **AeroJAX offline pipeline**: `run_map.py --check` before a long `--series` solve when that ignored local workspace is present.

## Gotchas

- `backend/services/aerojax_demo.py` resolves an **absolute path** to the AeroJAX directory (`_demo_root()`). If `AeroJAX_map_test2/` moves or `outputs/` is empty, `/flow-fields/aerojax-demo` (Real mode) fails — run a `--series` solve first.
- If OSM fetches fail, `geometry.py` returns no buildings; missing OSM heights use a labelled deterministic 10 m default (never display-randomized). If Open-Meteo fails or returns unusable wind, `wind.py` returns a deterministic 5 m/s north-origin fallback with explicit `source` and `fallback` metadata.
- The procedural road strips, markings, vegetation, facade bands, rooftop units, and beacon are presentation aids inferred from the live identity and empty-space checks. Only OSM building footprints/heights and the loaded field/bounds are physical source-of-truth geometry.
- `backend/cache/` and `cache/` hold OSMnx query caches (hash-named JSON). `backend/services/clear_cache.py` disables/clears them.
- The async simulation manager is **single-worker** — concurrent `/simulations` jobs queue, they don't parallelize.
