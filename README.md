# Urban Flighter

Most drone simulators give you still air and a pretty city.
Most CFD packages give you a beautiful field you cannot fly.

Urban Flighter is the awkward, useful thing in between: **a live city, a live weather inlet, a cheap hidden flow, and a quadrotor that actually pays drag for it.**

Pick a point on Earth. The backend pulls real OpenStreetMap building footprints and a **live location weather API** for that lat/lon. Today the local flow is a **toy CG model** (CFD-lite): cheap, interactive, good enough for drag. The slot is built so a **CFD surrogate** can replace that field later without rewriting the cockpit, the energy core, or the Gym contract. You fly a metre-scale quad in the browser. Relative airspeed is `v − w`, so a tailwind is cheap, a headwind is expensive, and a slab wake can be both in one block.

That pairing is the product. Not photorealism. Not a Navier–Stokes paper. A research cockpit where **geometry is live, wind is live, and drag is local.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Honest by design. CFD-lite is **not** Navier–Stokes. True 3D Wind is a **visual overlay**. LiDAR and rolling maps are **simulator rays + odometry**, not hardware SLAM. UrbanFlow Gym is **NOT TRAINED**. The actor never sees the full flow grid.

**Paper:** [`paper/urban_flighter.pdf`](paper/urban_flighter.pdf) · source [`paper/urban_flighter.tex`](paper/urban_flighter.tex)

<p align="center">
  <img src="docs/showcase/components/radar_split_nyc.gif" alt="3D Lite with toy CG wind on the left, rolling sensor map on the right" width="920" />
  <br />
  <sub>Midtown Manhattan · left: 3D Lite + toy CG flow · right: rolling sensor map (SIM odometry)</sub>
</p>

## Quick start

You need **Python 3.11+**, **Node 20+**, and **two terminals**. No API key. The first city load needs network (~10 s on a laptop).

```bash
git clone https://github.com/VortexyAether/UrbanFlighter.git
cd UrbanFlighter

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r backend/requirements.txt

cd frontend
npm install
cd ..
```

**Terminal 1 — backend**

```bash
cd backend
../.venv/bin/python main.py        # http://127.0.0.1:8000
```

**Terminal 2 — frontend**

```bash
cd frontend
npm run dev                        # http://127.0.0.1:5173
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Default city is **Midtown Manhattan / Times Square**. Wait until the bar says `BACKEND OK` and the footprints appear (~180 buildings). Then fly.

| | |
| --- | --- |
| **2D** (default) | WASD |
| **3D Arcade** | W/S forward · A/D strafe · Q/E yaw · Space/Shift climb · R boost · F brake |
| **3D Pilot** | A/D yaw · Q/E strafe |
| Camera | `C` Chase / Orbit |
| Mode | Map / Mode panel → 2D, 3D Lite, or True 3D Wind |

If the city never loads: confirm terminal 1 is still running and [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) returns OK. Keep both processes up. Presets: NYC, Paris, Tokyo, Inha.

## Look around

Same session, 2026-09-01: `BACKEND OK`, **179 OSM buildings / 400 m**, inlet **1.9 m/s from 215°**, `CFD-LITE B GRID`.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="docs/showcase/components/geometry_loader_map_nyc.png" alt="OSM geometry picker over Midtown Manhattan" />
      <br />
      <sub><b>1 · City</b> — live OSM footprints. Collision, LiDAR, and the flow mask share these solids.</sub>
    </td>
    <td align="center" width="50%">
      <img src="docs/showcase/components/simplecfd_2d_field_nyc.png" alt="Toy CG flow field around Midtown blocks" />
      <br />
      <sub><b>2 · Flow</b> — toy CG CFD-lite around those blocks. Surrogate-ready later.</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="docs/showcase/components/cockpit_3d_lite_nyc.png" alt="3D Lite cockpit with stone and glass facades" />
      <br />
      <sub><b>3 · 3D Lite</b> — fly the same horizontal field between limestone–glass prisms.</sub>
    </td>
    <td align="center" width="50%">
      <img src="docs/showcase/components/cockpit_2d_nyc.png" alt="2D north-up Midtown canvas" />
      <br />
      <sub><b>4 · 2D</b> — north-up canvas and a command bar. Same city, same drag.</sub>
    </td>
  </tr>
</table>

Recapture (optional, servers already up):

```bash
node scripts/capture_live_shots.mjs
node scripts/capture_radar_gif.mjs
```

## Why this is unusual

Typical stacks split the wrong way. AirSim-class tools optimize visuals and vehicle dynamics. Full CFD is the right tool for pedestrian-wind studies and the wrong tool for an interactive flight product. Urban Flighter occupies the gap those two keep leaving empty.

| What you get | Why it matters |
| --- | --- |
| **Live city geometry** | OSM footprints + heights, not a hand-built toy map. Collision, LiDAR, and the CFD mask share the same solids. |
| **Live location weather API** | The inlet is the current 10 m wind for the selected lat/lon. That is the known wind on the HUD and in actor observations. |
| **Toy CG flow, surrogate-ready** | Right now: CFD-lite B (potential flow, wall damping, canyon / wake, impermeable slip). Interactive on a laptop. Later: drop a CFD surrogate into the same `ux,uy` slot — drag, sensors, and Gym stay wired. |
| **Drag that notices the city** | Shared quadratic air-relative core. Stick-off equilibrium is local wind, not still air. Energy burns for heading into a canyon jet. |
| **A cockpit, not a plot window** | 2D, 3D Lite, and a True 3D Wind overlay. Deterministic returns. A rolling sensor map that admits it is odometry. |
| **An honest RL door** | UrbanFlow Gym sees the same world and is forbidden the velocity grid. Status: `NOT TRAINED`. No shipped weights. |

## How the four modules wire

**Geometry** · `backend/services/geometry.py` · OSMnx `features_from_point` → projected footprints + height (`osm:height`, else `building:levels × 3.5 m`, else a labelled 10 m default). Presentation trees and roads are scenery only.

**Wind** · `backend/services/wind.py` · A live location weather API returns current 10 m wind for the selected point. Building-scale local flow stays hidden.

**simplecfd** · `backend/services/flow_2d.py` · Today `/flow-fields/2d` is a **toy CG model**: streamfunction, wall damping, empirical wake / canyon, impermeable slip (`u·n = 0`). The cockpit does not care who wrote that grid — a later **CFD surrogate** can occupy the same endpoint. 3D Lite flies this `ux,uy` field. True 3D Wind is a Gangnam u/v/w overlay, not the flyable dynamics.

**Radar** · `LocalReturns2D.tsx` / `LocalReturnsRadar.tsx` · Deterministic rays vs OSM meshes (3D also hits `y=0`). Rolling map = **SIM odometry · no loop closure**.

## Modes

| Mode | Flyable wind | Extra |
| --- | --- | --- |
| 2D | Live horizontal CFD-lite B grid | 180-ray scan, rolling 2D map |
| 3D Lite | Same B grid | 600-sample spherical returns, 3D map |
| True 3D Wind | Still the B grid | Bundled Gangnam u/v/w streamlines |

## Showcase case (offline, no network)

A contract test, not a “wind-aware always wins” claim.

```bash
PYTHONPATH=backend python scripts/run_oss_showcase.py
```

<p align="center">
  <img src="docs/showcase/potential_flow_toy/potential_flow_streamlines.png" alt="Potential-flow CFD-lite on a five-building toy city" width="640" />
  <br />
  <sub>Toy city · 5 buildings · 5 m/s west inlet · residual 9.8e-5</sub>
</p>

UrbanFlow Gym path-tracking figures are omitted here. The contract stays `NOT TRAINED`.

## UrbanFlow Gym

Status: `LIVE OSM WORLD · NOT TRAINED`.

The headless env shares the registered live city. The actor sees motion, goal, 16-ray geometry LiDAR, 8-beam simulated radar (range + Doppler proxy), the known inlet, the previous action, and an inlet-only relative-air estimate. Never the velocity grid.

```bash
curl http://127.0.0.1:8000/urbanflow-gym/spec
# after a UI location load:
curl http://127.0.0.1:8000/urbanflow-gym/live-scenarios/current
```

Optional training extras are **not** on the default path. No shipped weights.

## Layout

```text
backend/                 FastAPI, OSM, weather, CFD-lite
backend/urbanflow_gym/   headless env + inspector API
frontend/                React + Three.js cockpit
scripts/run_oss_showcase.py
docs/showcase/           committed demo figures
paper/                   technical report (LaTeX + PDF)
```

Contributor notes: `CLAUDE.md`. Design tokens: `DESIGN.md`.

## Tests

```bash
cd frontend && npm run lint && npm run smoke:lidar && npm run build
cd ../backend
../.venv/bin/python test_urbanflow_gym_core.py
../.venv/bin/python test_urbanflow_rl_foundation.py
../.venv/bin/python test_urbanflow_gym_api.py
PYTHONPATH=backend ../.venv/bin/python -m urbanflow_gym.eval_policy --seed 10007 --max-steps 80
```

## Cite

```bibtex
@techreport{jang2026urbanflighter,
  title  = {Urban Flighter: An Honest Urban-Air Simulator
            with CFD-lite Hidden Flow and Policy-Visible Sensing},
  author = {Jang, Jaewon},
  year   = {2026},
  note   = {Open-source technical report},
  url    = {https://github.com/VortexyAether/UrbanFlighter}
}
```

## License

MIT. See `LICENSE`.
