from __future__ import annotations

import json
import threading
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from services.cfd import (
    compute_wind_field,
    sample_wind_trilinear,
    save_wind_json,
    save_wind_npy,
    save_wind_vtk,
    voxelize_buildings,
    wind_to_vector,
)
from services.geometry import fetch_buildings
from services.wind import get_real_weather


@dataclass
class SimulationRecord:
    simulation_id: str
    status: str = "queued"
    request: dict[str, Any] | None = None
    wind_json_path: str | None = None
    wind_vtk_path: str | None = None
    solid_npy_path: str | None = None
    error: str | None = None
    meta: dict[str, Any] | None = None


class SimulationManager:
    def __init__(self, output_root: Path, max_recent: int = 5) -> None:
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.max_recent = max_recent
        self.records: dict[str, SimulationRecord] = {}
        self.recent: deque[str] = deque()
        self.lock = threading.Lock()
        self.pool = ThreadPoolExecutor(max_workers=1)

    def submit(self, payload: dict[str, Any]) -> SimulationRecord:
        sim_id = uuid.uuid4().hex[:12]
        rec = SimulationRecord(simulation_id=sim_id, request=payload)
        with self.lock:
            self.records[sim_id] = rec
        self.pool.submit(self._run_job, sim_id)
        return rec

    def _run_job(self, sim_id: str) -> None:
        with self.lock:
            rec = self.records[sim_id]
            rec.status = "running"
            req = rec.request or {}

        try:
            lat = float(req.get("lat", 40.7580))
            lon = float(req.get("lon", -73.9855))
            radius_m = float(req.get("radius_m", 800.0))
            nx = int(req.get("nx", 64))
            ny = int(req.get("ny", 64))
            nz = int(req.get("nz", 32))
            voxel_size_m = float(req.get("voxel_size_m", 10.0))
            mode = str(req.get("mode", "real"))
            use_real_weather = bool(req.get("use_real_weather", True))

            fallback_used = False
            if mode == "real":
                try:
                    buildings = fetch_buildings(lat, lon, radius_m)
                    weather = get_real_weather(lat, lon) if use_real_weather else {"wind_speed": 5.0, "wind_deg": 0.0}
                except Exception:
                    fallback_used = True
                    buildings = []
                    weather = {"wind_speed": 5.0, "wind_deg": 0.0}
            else:
                fallback_used = False
                buildings = []
                weather = {"wind_speed": 5.0, "wind_deg": 0.0}

            if not buildings:
                # Small randomized fallback geometry if map empty.
                fallback_used = True
                rng = np.random.default_rng(42)
                buildings = []
                for _ in range(120):
                    cx = float(rng.uniform(-radius_m * 0.8, radius_m * 0.8))
                    cy = float(rng.uniform(-radius_m * 0.8, radius_m * 0.8))
                    w = float(rng.uniform(10, 40))
                    h = float(rng.uniform(8, 60))
                    fp = [[cx - w, cy - w], [cx + w, cy - w], [cx + w, cy + w], [cx - w, cy + w], [cx - w, cy - w]]
                    buildings.append({"height": h, "footprint": fp})

            # Auto domain fit for real radius.
            fit_cells = int(np.ceil((2.0 * radius_m * 1.15) / voxel_size_m))
            fit_cells = min(max(fit_cells, 24), 192)
            nx = max(nx, fit_cells)
            ny = max(ny, fit_cells)

            solid = voxelize_buildings(buildings, nx, ny, nz, voxel_size_m)
            global_wind = wind_to_vector(float(weather.get("wind_speed", 5.0)), float(weather.get("wind_deg", 0.0)))
            wind = compute_wind_field(solid, global_wind, voxel_size_m)

            out_dir = self.output_root / sim_id
            out_dir.mkdir(parents=True, exist_ok=True)
            wind_json_path = out_dir / "wind_field.json"
            wind_npy_path = out_dir / "wind_field.npy"
            wind_vtk_path = out_dir / "wind_field.vtk"
            solid_npy_path = out_dir / "solid.npy"

            save_wind_json(wind_json_path, wind, voxel_size_m=voxel_size_m)
            save_wind_npy(wind_npy_path, wind)
            save_wind_vtk(wind_vtk_path, wind, solid, voxel_size_m=voxel_size_m)
            np.save(solid_npy_path, solid.astype(np.uint8))

            meta = {
                "wind_speed_mps": float(weather.get("wind_speed", 5.0)),
                "wind_deg": float(weather.get("wind_deg", 0.0)),
                "build_count": int(len(buildings)),
                "fallback_used": bool(fallback_used),
                "effective_nx": int(nx),
                "effective_ny": int(ny),
                "effective_nz": int(nz),
                "solid_ratio": float(solid.mean()),
                "domain_fit_radius_m": float(min(nx, ny) * voxel_size_m * 0.5),
            }

            with self.lock:
                rec.status = "done"
                rec.wind_json_path = str(wind_json_path)
                rec.wind_vtk_path = str(wind_vtk_path)
                rec.solid_npy_path = str(solid_npy_path)
                rec.meta = meta
                self._touch_recent(sim_id)
        except Exception as e:
            with self.lock:
                rec.status = "error"
                rec.error = str(e)

    def _touch_recent(self, sim_id: str) -> None:
        if sim_id in self.recent:
            self.recent.remove(sim_id)
        self.recent.appendleft(sim_id)
        while len(self.recent) > self.max_recent:
            old = self.recent.pop()
            old_dir = self.output_root / old
            if old_dir.exists():
                for p in old_dir.glob("*"):
                    p.unlink(missing_ok=True)
                old_dir.rmdir()

    def get(self, sim_id: str) -> SimulationRecord | None:
        with self.lock:
            return self.records.get(sim_id)

    def sample(self, sim_id: str, points: list[list[float]]) -> list[list[float]]:
        rec = self.get(sim_id)
        if rec is None:
            raise KeyError("simulation not found")
        if rec.status != "done" or not rec.wind_json_path:
            raise RuntimeError("simulation not ready")

        payload = json.loads(Path(rec.wind_json_path).read_text(encoding="utf-8"))
        nx, ny, nz = int(payload["nx"]), int(payload["ny"]), int(payload["nz"])
        wind = np.zeros((nx, ny, nz, 3), dtype=np.float32)
        wind[..., 0] = np.asarray(payload["ux"], dtype=np.float32).reshape(nx, ny, nz)
        wind[..., 1] = np.asarray(payload["uy"], dtype=np.float32).reshape(nx, ny, nz)
        wind[..., 2] = np.asarray(payload["uz"], dtype=np.float32).reshape(nx, ny, nz)
        voxel_size = float(payload.get("voxel_size_m", 1.0))
        return sample_wind_trilinear(wind, voxel_size, points)
