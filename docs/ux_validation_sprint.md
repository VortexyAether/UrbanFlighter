# Urban Flighter UX / Validation Sprint Notes

## Current honesty labels

- **2D Analytical**: backend OSM geometry + analytical 2D wind field.
- **3D Analytical**: same OSM buildings in Three.js with local CG wind approximation.
- **Real CFD**: precomputed AeroJAX snapshot. Not a live CFD solve.
- **Swarm Replay**: Gangnam v3 deterministic A*+PD + separation baseline. **Not trained RL/MARL.**

## Reference comparison decisions

### Add now

- UI state/source labels: backend health, data source, deterministic baseline / not-trained-RL badges.
- Swarm replay controls: play/pause, reset, speed, frame scrubber.
- Swarm aggregate metrics: success count, collisions, near-miss, min separation, path length, relative-air-speed energy proxy.
- Replay JSON validator: reproducibility and consistency gate for metrics/trajectory pairs.

### Add later

- Gymnasium `UrbanFlightEnv` / PettingZoo-style multi-agent skeleton.
- deck.gl/MapLibre analysis layer for paths, no-fly zones, and time-based TripsLayer replay.
- OSM2World / glTF city asset pipeline.
- Stable-Baselines3 single-agent training only after observations/actions/rewards stabilize.

### Too heavy for now

- AirSim, Flightmare, Isaac Sim, Habitat, ROS/Gazebo/PX4, full MARL frameworks.
- They are useful for high-fidelity robotics or sim-to-real later, but they slow down the current browser-first demo.

## Verification commands

From repo root:

```bash
# Validate deterministic Gangnam v3 replay JSON.
python3 scripts/validate_swarm_replay.py \
  --metrics results/multi_drone_gangnam_v3/urban_flighter_multi_drone_metrics.json \
  --trajectories results/multi_drone_gangnam_v3/urban_flighter_multi_drone_trajectories.json \
  --out results/multi_drone_gangnam_v3/validation_report.json

# Frontend build.
cd frontend
npm run build
npm run preview -- --host 127.0.0.1 --port 5174

# Backend health.
cd ../backend
source ../.venv/bin/activate
python main.py
curl http://127.0.0.1:8000/health
```

## Browser acceptance checklist

- Status shows backend `BACKEND OK` when FastAPI is running.
- Mode buttons are radios and explicitly named: 2D Analytical / 3D Analytical / Real CFD / Swarm Replay.
- Swarm mode shows `DETERMINISTIC SWARM`, `A*+PD BASELINE`, and `NOT TRAINED RL`.
- Gangnam v3 source card shows controller, OSM count, radius, trajectory URL, and honesty note.
- Replay controls work: play/pause, reset, speed, frame slider.
- Aggregate metrics are visible: success, collisions, near-miss, min separation, total path, energy proxy.
- Colored drone markers and trajectory paths are visible in the Three.js canvas.

## Known limits

- Swarm collision/separation validation is discrete replay-data consistency, not continuous swept-volume proof.
- Building geometry remains simplified to OSM-derived prism approximations for this baseline.
- Backend OSM/weather requests can fail depending on network/API availability; UI now exposes backend/offline state more clearly.
