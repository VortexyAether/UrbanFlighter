from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class CFDFieldDataset(Protocol):
    """Protocol for an unseen offline 3D snapshot or temporal CFD dataset."""

    @property
    def inlet_velocity_xyz(self) -> np.ndarray:
        ...

    def sample_velocity_xyz(self, position_xyz: np.ndarray, time_s: float) -> np.ndarray:
        ...

    def dataset_metadata(self) -> dict:
        ...


class CFDWindProvider2DAdapter:
    """Maps the 2D Gym flight plane into a hidden 3D CFD dataset sampler."""

    def __init__(self, dataset: CFDFieldDataset, flight_altitude_m: float) -> None:
        if not isinstance(dataset, CFDFieldDataset):
            raise TypeError("dataset must implement CFDFieldDataset")
        inlet = np.asarray(dataset.inlet_velocity_xyz, dtype=float)
        if inlet.shape != (3,) or not np.all(np.isfinite(inlet)):
            raise ValueError("CFD dataset inlet_velocity_xyz must be a finite 3D vector")
        if not np.isfinite(float(flight_altitude_m)):
            raise ValueError("flight_altitude_m must be finite")
        self.dataset = dataset
        self.flight_altitude_m = float(flight_altitude_m)
        self._inlet_velocity = inlet[:2].copy()

    @property
    def inlet_velocity(self) -> np.ndarray:
        return self._inlet_velocity.copy()

    def velocity_at(self, position_xy: np.ndarray, time_s: float) -> np.ndarray:
        point = np.asarray(position_xy, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("CFD adapter position must be a finite 2D vector")
        if not np.isfinite(float(time_s)):
            raise ValueError("CFD adapter time must be finite")
        sample = np.asarray(
            self.dataset.sample_velocity_xyz(
                np.array([point[0], point[1], self.flight_altitude_m], dtype=float),
                float(time_s),
            ),
            dtype=float,
        )
        if sample.shape != (3,) or not np.all(np.isfinite(sample)):
            raise ValueError("CFD dataset returned a non-finite or non-3D velocity sample")
        return sample[:2]

    def source_metadata(self) -> dict:
        return {
            "kind": "offline_3d_cfd_dataset_adapter",
            "flight_altitude_m": self.flight_altitude_m,
            "hidden_from_actor_except_known_inlet": True,
            "dataset": self.dataset.dataset_metadata(),
        }


@dataclass(frozen=True)
class StructuredCFDDataset:
    """NumPy structured-grid adapter for snapshot or temporal offline datasets.

    Velocity layout is ``[time, z, y, x, xyz]``. A snapshot may omit the time
    dimension when loaded through :meth:`from_npz`.
    """

    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    time_s: np.ndarray
    velocity_mps: np.ndarray
    _inlet_velocity_xyz: np.ndarray
    metadata: dict

    def __post_init__(self) -> None:
        x = _validated_axis(self.x_m, "x_m")
        y = _validated_axis(self.y_m, "y_m")
        z = _validated_axis(self.z_m, "z_m")
        times = np.asarray(self.time_s, dtype=float)
        if times.ndim != 1 or times.size < 1 or not np.all(np.isfinite(times)):
            raise ValueError("time_s must be a finite non-empty 1D axis")
        if times.size > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("time_s must be strictly increasing")
        velocity = np.asarray(self.velocity_mps, dtype=float)
        expected = (times.size, z.size, y.size, x.size, 3)
        if velocity.shape != expected or not np.all(np.isfinite(velocity)):
            raise ValueError(f"velocity_mps must have finite shape {expected}")
        inlet = np.asarray(self._inlet_velocity_xyz, dtype=float)
        if inlet.shape != (3,) or not np.all(np.isfinite(inlet)):
            raise ValueError("inlet velocity must be a finite 3D vector")
        object.__setattr__(self, "x_m", x)
        object.__setattr__(self, "y_m", y)
        object.__setattr__(self, "z_m", z)
        object.__setattr__(self, "time_s", times)
        object.__setattr__(self, "velocity_mps", velocity)
        object.__setattr__(self, "_inlet_velocity_xyz", inlet)
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a JSON-compatible object")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def inlet_velocity_xyz(self) -> np.ndarray:
        return self._inlet_velocity_xyz.copy()

    @classmethod
    def from_npz(cls, path: str | Path) -> "StructuredCFDDataset":
        dataset_path = Path(path).expanduser().resolve()
        with np.load(dataset_path, allow_pickle=False) as archive:
            required = {"x_m", "y_m", "z_m", "velocity_mps", "inlet_velocity_mps"}
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"CFD NPZ is missing required arrays: {', '.join(missing)}")
            velocity = np.asarray(archive["velocity_mps"], dtype=float)
            if velocity.ndim == 4:
                velocity = velocity[np.newaxis, ...]
            times = np.asarray(archive["time_s"], dtype=float) if "time_s" in archive else np.array([0.0])
            raw_metadata = str(archive["metadata_json"].item()) if "metadata_json" in archive else "{}"
            try:
                metadata = json.loads(raw_metadata)
            except json.JSONDecodeError as exc:
                raise ValueError("CFD NPZ metadata_json is not valid JSON") from exc
            metadata = {
                **metadata,
                "path_name": dataset_path.name,
                "adapter_format": "urbanflow_structured_cfd_npz_v1",
            }
            return cls(
                x_m=archive["x_m"],
                y_m=archive["y_m"],
                z_m=archive["z_m"],
                time_s=times,
                velocity_mps=velocity,
                _inlet_velocity_xyz=archive["inlet_velocity_mps"],
                metadata=metadata,
            )

    def sample_velocity_xyz(self, position_xyz: np.ndarray, time_s: float) -> np.ndarray:
        position = np.asarray(position_xyz, dtype=float)
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("CFD sample position must be a finite 3D vector")
        if not np.isfinite(float(time_s)):
            raise ValueError("CFD sample time must be finite")
        ix0, ix1, tx = _bracket(self.x_m, position[0])
        iy0, iy1, ty = _bracket(self.y_m, position[1])
        iz0, iz1, tz = _bracket(self.z_m, position[2])
        it0, it1, tt = _bracket(self.time_s, float(time_s))

        def spatial(time_index: int) -> np.ndarray:
            grid = self.velocity_mps[time_index]
            c000 = grid[iz0, iy0, ix0]
            c001 = grid[iz0, iy0, ix1]
            c010 = grid[iz0, iy1, ix0]
            c011 = grid[iz0, iy1, ix1]
            c100 = grid[iz1, iy0, ix0]
            c101 = grid[iz1, iy0, ix1]
            c110 = grid[iz1, iy1, ix0]
            c111 = grid[iz1, iy1, ix1]
            c00 = c000 * (1.0 - tx) + c001 * tx
            c01 = c010 * (1.0 - tx) + c011 * tx
            c10 = c100 * (1.0 - tx) + c101 * tx
            c11 = c110 * (1.0 - tx) + c111 * tx
            c0 = c00 * (1.0 - ty) + c01 * ty
            c1 = c10 * (1.0 - ty) + c11 * ty
            return c0 * (1.0 - tz) + c1 * tz

        before = spatial(it0)
        after = spatial(it1)
        return before * (1.0 - tt) + after * tt

    def dataset_metadata(self) -> dict:
        return {
            **self.metadata,
            "grid_shape_tzyx": [
                int(self.time_s.size),
                int(self.z_m.size),
                int(self.y_m.size),
                int(self.x_m.size),
            ],
            "temporal": bool(self.time_s.size > 1),
            "inlet_velocity_mps": self._inlet_velocity_xyz.tolist(),
        }


def _validated_axis(values: np.ndarray, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if axis.ndim != 1 or axis.size < 1 or not np.all(np.isfinite(axis)):
        raise ValueError(f"{name} must be a finite non-empty 1D axis")
    if axis.size > 1 and np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return axis


def _bracket(axis: np.ndarray, value: float) -> tuple[int, int, float]:
    if axis.size == 1:
        return 0, 0, 0.0
    clipped = float(np.clip(value, axis[0], axis[-1]))
    upper = int(np.searchsorted(axis, clipped, side="right"))
    upper = min(max(upper, 1), axis.size - 1)
    lower = upper - 1
    fraction = (clipped - axis[lower]) / max(axis[upper] - axis[lower], 1e-12)
    return lower, upper, float(fraction)
