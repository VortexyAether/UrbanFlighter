# Quadratic air-relative drag

Urban Flighter now uses one shared drag core in the flyable 2D/3D motion, energy HUD, Python physics, and a Unity C# port.

## Model

- Relative airspeed: `v_air = v_ground - w_local`
- Parasite acceleration: implicit `a = -k |v_air| v_air - c v_air` with `k = 0.5 ρ Cd A / m`
- Induced power: momentum-theory `P_ind = κ W v_h / sqrt(1 + (|v_air|/v_h)^2)`
- Stick-off equilibrium is **local wind**, not still air

## Honesty

`QUADRATIC AIR-RELATIVE DRAG · MOMENTUM-THEORY INDUCED · NOT BLADE-ELEMENT / NOT NS`

Gym UrbanFlow still uses its existing linear `wind_drag_gain_per_s` by default so the RL contract does not silently change.

## Files

- `frontend/src/simulation/quadraticAirDrag.ts`
- `frontend/src/simulation/flight2dMotion.ts`
- `frontend/src/simulation/flight3dMotion.ts`
- `backend/urban_flighter_physics/quadratic_air_drag.py`
- `unity/UrbanFlighterDrag/Runtime/QuadraticAirDrag.cs`
- `scripts/run_drag_essence_demo.py`

## Verify

```bash
cd frontend && npm run smoke:flight-core && npm run smoke:flight3d
cd ../backend && ../.venv/bin/python test_quadratic_air_drag.py
cd .. && PYTHONPATH=backend .venv/bin/python scripts/run_drag_essence_demo.py
```
