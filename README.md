# Urban Flighter

Interactive urban-drone flight simulator using real OSM building footprints, live weather, CFD-lite wind fields, and LiDAR-inspired local sensor maps.

> **Status:** research/demo simulator. The CFD-lite field and LiDAR/SLAM-lite views are honest prototypes, not hardware-qualified sensing, full CFD, or production SLAM.

## What it does

- Fly through a real urban building layout with keyboard controls and wind/energy effects.
- Load OSM building geometry and current wind conditions through a FastAPI backend.
- Choose three flight views:
  - **2D** — top-down flight, CFD-lite wind field, JET LiDAR returns, rolling 2D sensor map.
  - **3D Lite** — Three.js city flight with a visible ground surface, ground/building LiDAR returns, and a rolling 3D local sensor map.
  - **True 3D Wind** — 3D U/V/W streamline demonstration mode.
- Inspect a deterministic, fixed-layout LiDAR observation vector for future RL work.
- Move, hide, resize, orbit, and zoom independent cockpit windows during flight.

## LiDAR and sensor maps

The 3D LiDAR prototype uses **600 deterministic Fibonacci-sphere samples** at a 120 m maximum range. It renders both building/ground hits and max-range misses with a JET-style range colormap.

The 2D view uses **180 rays** at 180 m. Its default spawn selection uses real nearby scan returns so the local map is readable on load.

The rolling map windows retain bounded recent scan keyframes plus a simulated-odometry trajectory:

- Current scan: vivid JET returns.
- Earlier scans: age-faded history.
- Trajectory/keyframes: display-only simulated odometry.
- **Not included:** pose graph optimization, loop closure, localization uncertainty estimation, or hardware sensor calibration.

The fixed RL observation contract remains five values per ray:

```text
local direction x, local direction y, local direction z,
normalized distance, hit flag
```

## Quick start

### Backend

```bash
cd backend
source venv/bin/activate
python main.py
```

The API serves on `http://localhost:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Controls

| View | Flight controls |
| --- | --- |
| 2D | `WASD` moves the drone in map directions. |
| 3D | `W/S` forward/reverse, `A/D` yaw, `E/Q`, arrows, or space/shift for climb/descend. |

Cockpit windows:

- Drag `⠿ MOVE` title bars to reposition panels.
- Use the SLAM window corner handle to resize it.
- Use `−`, `+`, and `RESET VIEW` to change sensor-map scale.
- In 3D, drag inside the map to orbit and scroll to zoom.
- `CLEAR MAP` clears display history only; it does not reset the simulator or change RL state.

## Validation

```bash
cd frontend
npm run smoke:lidar
npm run build
```

`npm run lint` may surface pre-existing project-wide lint findings; the LiDAR/cockpit implementation is validated with targeted ESLint and production builds.

## Architecture

- `backend/` — FastAPI, OSM geometry, weather, and CFD-lite/3D wind services.
- `frontend/` — React 19, TypeScript, Vite, Three.js/react-three-fiber.
- `frontend/src/sensors/` — deterministic LiDAR scans, 2D scans, and bounded rolling sensor-map utilities.
- `docs/` — simulator, RL, and UX design notes.

See `CLAUDE.md` for contributor notes and `DESIGN.md` for design decisions.
