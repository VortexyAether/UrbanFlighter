# Multi-drone real-city RL-style rollout

This extends the real OSM/satellite prototype from a single drone to a **multi-agent RL-style scenario**: several drones fly across the same real city patch, share the same wind/building field, and collect per-agent reward/energy/safety metrics.

It is still a deterministic baseline, not a trained MARL policy. That is intentional: it gives a visible, debuggable baseline and generates trajectories/metrics that can become training/evaluation data.

## New files

```text
backend/urban_flighter_rl/multi_agent.py
backend/urban_flighter_rl/render_multi.py
scripts/run_multi_drone_real_city_demo.py
docs/multi_drone_real_city_demo.md
```

## What the demo does when network is available

```text
Gangnam lat/lon
→ OSM Overpass building footprints
→ Esri satellite center tile
→ local-meter 3D city
→ 6 drones with crossing missions
→ wind-aware A* + PD baseline per drone
→ inter-drone separation action
→ altitude-layer deconfliction
→ multi-trajectory 3D render + metrics JSON + trajectory JSON
```

The checked-in `multi_drone_gangnam_v3` replay is currently a geometry-bearing fallback artifact, not a fresh Gangnam OSM regeneration. Overpass DNS/network access was unavailable during collision-hardening verification, so the replay was regenerated from `UrbanWorld.toy_city` with embedded building prisms. That keeps the public replay honestly and exactly swept-validated instead of claiming validation against missing OSM geometry.

## Run

```bash
cd .
source .venv/bin/activate
PYTHONPATH=backend python scripts/run_multi_drone_real_city_demo.py \
  --lat 37.497942 \
  --lon 127.027621 \
  --radius 260 \
  --max-buildings 140 \
  --drones 6 \
  --max-steps 1100 \
  --out results/multi_drone_gangnam_v3
```

## Current checked-in v3 replay — 2026-06-24

```text
world_source: UrbanWorld.toy_city fallback; Gangnam OSM regeneration blocked by network
n_drones: 6
success_count: 6
all_success: True
total_collisions: 0
total_boundary_violations: 0
swept_building_hits: 0
near_miss_count_sep_lt_10m: 56
min_pairwise_separation_m: 1.078510182653687
total_energy_relative_airspeed_l2: 4895.072580892918
total_path_length_m: 551.3681732442597
usable_buildings: 5
validated_scope: replay JSON consistency plus exact swept segment checks against axis-aligned building prisms
```

Validation command:

```bash
python3 scripts/validate_swarm_replay.py \
  --metrics frontend/public/data/multi_drone_gangnam_v3/urban_flighter_multi_drone_metrics.json \
  --trajectories frontend/public/data/multi_drone_gangnam_v3/urban_flighter_multi_drone_trajectories.json \
  --out /tmp/uf_swarm_validate.json
```

## Previous real-city result — archived context, not current public replay

```text
n_drones: 6
success_count: 6
all_success: True
total_collisions: 0
near_miss_count_sep_lt_10m: 0
min_pairwise_separation_m: 10.071595673737944
total_energy_relative_airspeed_l2: 18578.877382436804
total_path_length_m: 3019.646903777134
osm_buildings: 140
osm_elements: 178
```

Artifacts:

```text
results/multi_drone_gangnam_v3/urban_flighter_multi_drone_real_city.png
results/multi_drone_gangnam_v3/urban_flighter_multi_drone_metrics.json
results/multi_drone_gangnam_v3/urban_flighter_multi_drone_trajectories.json
results/multi_drone_gangnam_v3/satellite_center_tile.jpg
```

## RL/MARL interpretation

Current policy:

```text
per-drone action = waypoint PD action
                 + inter-drone separation action
                 + obstacle repulsion action
```

Current reward source remains the same as `UrbanFlighterEnv`:

```text
reward = progress/success terms
       - relative-air-speed energy penalty
       - collision penalty
       - time/distance cost
```

Energy remains the VA-requested abstraction:

```text
v_air = v_ground - w_local(x,y,z,t)
energy = Σ_drones Σ_t ||v_air||² Δt
```

This can become a MARL environment by exposing:

- per-agent observations: relative goal, own velocity, inlet/base wind, OSM/building proximity sectors, and nearby drone relative ranges
- per-agent action: acceleration or desired velocity
- shared world: OSM buildings + hidden local wind sampler for dynamics/reward only
- reward: progress, energy, building collision, pairwise separation, success
- algorithms: MAPPO, MASAC, MADDPG-style centralized critic, or independent SAC as first baseline

## Next improvements

1. Turn `run_multi_drone_baseline` into a proper `MultiDroneUrbanEnv` with `reset/step` API.
2. Add pairwise separation penalty directly to reward, not only action shaping.
3. Export trajectory JSON into the React/Three.js frontend for animated swarm replay.
4. Preserve exact OSM polygon extrusion instead of bounding prisms.
5. Train independent SAC first, then compare against MAPPO/MASAC.
