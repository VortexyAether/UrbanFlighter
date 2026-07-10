# Urban Flighter RL Environment Contract

This pass implements an RL-ready simulator surface, not trained RL.

## Policy Observation Boundary

`UrbanFlighterEnv.reset(seed, options)` and `UrbanFlighterEnv.step(action)` return observations limited to mission-observable data:

- relative goal vector
- drone velocity
- inlet/base wind vector
- nearest building clearance
- OSM/building sector range features

The policy does not receive absolute position, the full flow grid, an AeroJAX/CFD snapshot, or the hidden local wind sample. CFD-lite and heuristic wind remain simulator dynamics and reward inputs through the `wind.at(position, t)` interface.

## Reward And Metrics

Reward terms:

- progress
- path cost
- time cost
- relative-air-speed energy cost
- building collision cost
- boundary cost
- multi-drone separation cost
- goal bonus

Metrics include collisions, boundary/separation violations, relative-air-speed energy, path length, success, minimum building clearance, minimum pairwise separation, and `policy_had_privileged_flow_access=false`.

## Verified Commands

From the repo root:

```bash
PYTHONPYCACHEPREFIX=/tmp/urban_flighter_pycache python3 -m compileall backend/urban_flighter_rl backend/main.py backend/services/flow_2d.py
PYTHONPATH=backend .venv/bin/python scripts/run_rl_baseline_rollout.py --seed 11 --max-steps 40 --drones 4 --out results/rl_contract_rollout.json --metrics-out results/rl_contract_metrics.json
cd frontend && npm run build
```

Live API smoke:

```bash
cd .
PYTHONPATH=backend .venv/bin/uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
curl -i http://127.0.0.1:8000/rl/spec
curl -i 'http://127.0.0.1:8000/rl/baseline?seed=11&max_steps=40&n_drones=4'
```

Frontend browser smoke:

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
open http://127.0.0.1:5173
```

Select `RL Baseline`. The panel should show OSM+inlet-only observations, hidden heuristic/CFD-lite proxy dynamics, deterministic/random baseline, reward terms, and separation/collision/energy/path metrics.

## Real CFD/AeroJAX Evaluation Hook

The eval path should swap the training wind object behind the same `wind.at(position, t)` method:

```text
UrbanWindField hidden dynamics
→ AeroJAX/CFD snapshot sampler hidden dynamics
→ same UrbanFlighterEnv observation contract
→ compare deterministic baseline or learned policy metrics
```

Do not add AeroJAX grid values to policy observations. A learned policy must be evaluated with the same OSM+inlet-only observation schema, with the real CFD sampler used only by environment dynamics and reward/metric calculation.
