# UrbanFlow Gym contract

Primary RL environment: `UrbanFlowGym-v1` (`backend/urbanflow_gym/`).
Legacy 3D demo env: `backend/urban_flighter_rl/` (`/rl/spec`). Do not train new work there.

## Actor observation (v1.1, 49 scalars)

The actor never receives the hidden CFD-lite grid.

| Field | Dim | Source |
| --- | --- | --- |
| position, velocity, heading | 6 | own odometry |
| relative air estimate | 2 | `v_ground - known_inlet` |
| known inlet | 2 | Open-Meteo / labelled fallback |
| goal context | 5 | mission |
| LiDAR ranges | 16 | geometry rays, 360° |
| radar ranges | 8 | forward-fan geometry rays |
| radar range-rate | 8 | Doppler proxy from own ground velocity |
| previous action | 2 | last command |

Radar is a **simulated range–Doppler proxy**. It is not RF hardware.

## Hidden uses of local wind

Hidden local wind (CFD-lite or synthetic wake) is allowed only for:

- flyable / Gym dynamics (quadratic air-relative drag)
- reward and episode metrics
- optional privileged critic (`privileged_critic_state()`, never returned by `reset`/`step`)

## Train / eval

```bash
# Offline fixture (no network)
PYTHONPATH=backend .venv/bin/python -m urbanflow_gym.eval_policy --seed 10007 --max-steps 240

# Optional PPO/SAC extras
.venv/bin/python -m pip install -r backend/requirements-urbanflow-train.txt
PYTHONPATH=backend .venv/bin/python -m urbanflow_gym.train --algorithm ppo --world fixture --total-timesteps 100000
PYTHONPATH=backend .venv/bin/python -m urbanflow_gym.eval_policy --checkpoint results/urbanflow_gym/training/urbanflow_ppo_seed17.zip
```

Live OSM training needs a registered `/flow-fields/2d` snapshot or an exported training bundle (`export_training_bundle` / `--snapshot`).

No shipped weights. No Navier–Stokes validation claim.
