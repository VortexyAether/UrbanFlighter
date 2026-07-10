# Backend CFD Quickstart

## Run
```bash
cd <repo>/backend
source venv/bin/activate
python main.py
```

## Create simulation (real mode)
```bash
curl -X POST http://127.0.0.1:8000/simulations \
  -H "Content-Type: application/json" \
  -d '{
    "lat":37.4979,"lon":127.0276,
    "radius_m":800,
    "nx":64,"ny":64,"nz":32,
    "voxel_size_m":10.0,
    "mode":"real",
    "use_real_weather":true,
    "save_vtk":true
  }'
```

## Poll status
```bash
curl http://127.0.0.1:8000/simulations/<SIM_ID>
```

Success criteria:
- `status=done`
- `meta.fallback_used=false` for real geometry/weather

Output files:
- `backend/sim_outputs/<SIM_ID>/wind_field.json`
- `backend/sim_outputs/<SIM_ID>/wind_field.npy`
- `backend/sim_outputs/<SIM_ID>/wind_field.vtk`
- `backend/sim_outputs/<SIM_ID>/solid.npy`
