# Urban Flighter 3D RL Prototype

This folder contains a lightweight Python prototype for 3D path optimization before committing to Unreal integration.

## What it does

- Creates a 3D toy urban world with building prisms.
- Computes a local wind field:
  - base wind
  - building wake speed reduction
  - side deflection
  - rooftop/updraft proxy
- Simulates a point-mass drone in 3D.
- Uses only relative airspeed for energy:

```text
v_air = v_ground - w_local(x, y, z, t)
energy = Σ ||v_air||² Δt
```

- Runs a wind-aware potential-field baseline planner.
- Saves:
  - `results/urban_flighter_3d_demo.png`
  - `results/urban_flighter_metrics.json`

## Run

```bash
cd .
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements-prototype.txt
PYTHONPATH=backend python scripts/run_3d_rl_demo.py
```

## RL path forward

The `UrbanFlighterEnv` class is Gymnasium-style:

```python
obs = env.reset()
result = env.step(action)
```

Action: 3D acceleration command normalized to `[-1, 1]` direction/magnitude.

Observation:

```text
normalized velocity
normalized vector-to-goal
inlet/base wind vector
nearest building clearance
OSM/building sector range features
```

The policy must not receive the full CFD-lite/AeroJAX flow grid or hidden local
flow sample. Hidden wind is simulator dynamics and reward accounting only.

Recommended training sequence:

1. Train SAC/TD3 for continuous action local control in the toy world.
2. Add curriculum: sparse buildings → dense blocks → random wind → real OSM layout.
3. Use global A*/RRT* or Fast Marching for waypoint skeleton.
4. Use RL as local wind-aware controller between waypoints.
5. Transfer the same API to Unreal via a bridge.
