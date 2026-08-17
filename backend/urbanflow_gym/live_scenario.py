from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import hashlib
import json
import math
from threading import RLock
from typing import Any, Iterable, Mapping

import numpy as np

from .geometry import PolygonPrism, UrbanGeometry, polygon_signed_area
from .scenario import UrbanFlowScenario


LIVE_SCENARIO_SCHEMA_ID = "urbanflow.live_scenario.v1"
LIVE_SCENARIO_SCHEMA_VERSION = 1
LIVE_SCENARIO_ID_PREFIX = "urbanflow-live-v1-"
LIVE_SCENARIO_REGISTRY_MAX_ENTRIES = 6
MAX_LIVE_BUILDINGS = 2_500
MAX_POLYGON_VERTICES = 512
MAX_TOTAL_POLYGON_VERTICES = 60_000
MAX_BUILDING_HEIGHT_M = 1_000.0
MAX_FLOW_GRID_CELLS = 131_072
MAX_ABSOLUTE_LOCAL_COORDINATE_M = 3_500.0
LIVE_AGENT_RADIUS_M = 1.25
LIVE_GOAL_RADIUS_M = 2.5
LIVE_LIDAR_RANGE_M = 35.0
LIVE_LIDAR_RAY_COUNT = 16


class LiveScenarioError(RuntimeError):
    pass


class LiveScenarioValidationError(ValueError):
    pass


class NoLiveScenarioError(LiveScenarioError):
    pass


class UnknownLiveScenarioError(LiveScenarioError):
    pass


def _finite_float(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    digits: int | None = None,
) -> float:
    if isinstance(value, bool):
        raise LiveScenarioValidationError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LiveScenarioValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise LiveScenarioValidationError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise LiveScenarioValidationError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise LiveScenarioValidationError(f"{name} must be at most {maximum}")
    if digits is not None:
        result = round(result, digits)
    return 0.0 if result == 0.0 else result


def _bounded_text(value: Any, name: str, maximum_length: int = 240) -> str:
    result = str(value)
    if not result or len(result) > maximum_length:
        raise LiveScenarioValidationError(
            f"{name} must contain between 1 and {maximum_length} characters"
        )
    return result


def _json_safe(value: Any, name: str, depth: int = 0) -> Any:
    if depth > 5:
        raise LiveScenarioValidationError(f"{name} metadata is nested too deeply")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if len(value) > 500:
            raise LiveScenarioValidationError(f"{name} string metadata is too long")
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _finite_float(value, name, digits=9)
    if isinstance(value, Mapping):
        if len(value) > 40:
            raise LiveScenarioValidationError(f"{name} metadata has too many fields")
        return {
            _bounded_text(key, f"{name} key", 120): _json_safe(item, f"{name}.{key}", depth + 1)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        if len(value) > 80:
            raise LiveScenarioValidationError(f"{name} metadata list is too long")
        return [_json_safe(item, f"{name}[]", depth + 1) for item in value]
    return _bounded_text(value, name, 240)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LiveScenarioValidationError("live scenario is not canonical JSON") from exc


def _canonical_polygon(raw_footprint: Any, building_index: int) -> list[list[float]]:
    if not isinstance(raw_footprint, (list, tuple)):
        raise LiveScenarioValidationError(
            f"building {building_index} footprint must be an array"
        )
    if len(raw_footprint) > MAX_POLYGON_VERTICES + 1:
        raise LiveScenarioValidationError(
            f"building {building_index} exceeds {MAX_POLYGON_VERTICES} polygon vertices"
        )
    points: list[tuple[float, float]] = []
    for point_index, raw_point in enumerate(raw_footprint):
        if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
            raise LiveScenarioValidationError(
                f"building {building_index} vertex {point_index} must be [x, y]"
            )
        point = (
            _finite_float(
                raw_point[0],
                f"building {building_index} vertex {point_index} x",
                minimum=-MAX_ABSOLUTE_LOCAL_COORDINATE_M,
                maximum=MAX_ABSOLUTE_LOCAL_COORDINATE_M,
                digits=6,
            ),
            _finite_float(
                raw_point[1],
                f"building {building_index} vertex {point_index} y",
                minimum=-MAX_ABSOLUTE_LOCAL_COORDINATE_M,
                maximum=MAX_ABSOLUTE_LOCAL_COORDINATE_M,
                digits=6,
            ),
        )
        if not points or point != points[-1]:
            points.append(point)
    if len(points) >= 2 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or len(set(points)) < 3:
        raise LiveScenarioValidationError(
            f"building {building_index} footprint must contain at least three unique vertices"
        )
    if len(points) > MAX_POLYGON_VERTICES:
        raise LiveScenarioValidationError(
            f"building {building_index} exceeds {MAX_POLYGON_VERTICES} polygon vertices"
        )

    footprint = np.asarray(points, dtype=float)
    area = polygon_signed_area(footprint)
    if abs(area) < 0.01:
        raise LiveScenarioValidationError(
            f"building {building_index} footprint area is degenerate"
        )
    if area < 0.0:
        points.reverse()

    minimum_point = min(points)
    candidate_indices = [index for index, point in enumerate(points) if point == minimum_point]
    rotations = [points[index:] + points[:index] for index in candidate_indices]
    points = min(rotations)
    return [[point[0], point[1]] for point in points]


def canonicalize_buildings(
    buildings: Iterable[Mapping[str, Any]],
    bounds_xy: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    raw_buildings = list(buildings)
    if not 1 <= len(raw_buildings) <= MAX_LIVE_BUILDINGS:
        raise LiveScenarioValidationError(
            f"live OSM scenario must contain between 1 and {MAX_LIVE_BUILDINGS} buildings"
        )
    min_x, max_x, min_y, max_y = bounds_xy
    canonical: list[dict[str, Any]] = []
    total_vertices = 0
    for index, raw in enumerate(raw_buildings):
        if not isinstance(raw, Mapping):
            raise LiveScenarioValidationError(f"building {index} must be an object")
        footprint = _canonical_polygon(raw.get("footprint"), index)
        total_vertices += len(footprint)
        if total_vertices > MAX_TOTAL_POLYGON_VERTICES:
            raise LiveScenarioValidationError(
                f"live scenario exceeds {MAX_TOTAL_POLYGON_VERTICES} total polygon vertices"
            )
        xs = [point[0] for point in footprint]
        ys = [point[1] for point in footprint]
        if max(xs) < min_x or min(xs) > max_x or max(ys) < min_y or min(ys) > max_y:
            raise LiveScenarioValidationError(
                f"building {index} lies completely outside the scenario bounds"
            )
        height = _finite_float(
            raw.get("height"),
            f"building {index} height",
            minimum=0.1,
            maximum=MAX_BUILDING_HEIGHT_M,
            digits=4,
        )
        height_source = _bounded_text(
            raw.get("height_source", "legacy_unspecified"),
            f"building {index} height_source",
            80,
        )
        geometry_identity = hashlib.sha256(
            _canonical_json_bytes({"footprint_xy_m": footprint, "height_m": height})
        ).hexdigest()[:16]
        source_element_id = raw.get("building_id")
        if source_element_id is None and isinstance(raw.get("source"), Mapping):
            source_element_id = raw["source"].get("element_id")
        source_element = (
            _bounded_text(source_element_id, f"building {index} source id", 160)
            if source_element_id is not None
            else None
        )
        canonical.append(
            {
                "building_id": f"osm-{geometry_identity}",
                "source_element_id": source_element,
                "footprint_xy_m": footprint,
                "height_m": height,
                "height_source": height_source,
            }
        )
    canonical.sort(key=lambda building: _canonical_json_bytes(building))
    return canonical


@dataclass(frozen=True)
class FrozenFlowField2D:
    nx: int
    ny: int
    bounds_xy: tuple[float, float, float, float]
    cell_size_m: float
    ux: np.ndarray
    uy: np.ndarray
    mask: np.ndarray
    digest_sha256: str

    @classmethod
    def from_payload(cls, field: Mapping[str, Any]) -> "FrozenFlowField2D":
        try:
            nx = int(field["nx"])
            ny = int(field["ny"])
            raw_bounds = field["bounds"]
        except (KeyError, TypeError, ValueError) as exc:
            raise LiveScenarioValidationError("live flow field is missing grid metadata") from exc
        if isinstance(field.get("nx"), bool) or isinstance(field.get("ny"), bool):
            raise LiveScenarioValidationError("live flow grid dimensions must be integers")
        if nx < 2 or ny < 2 or nx * ny > MAX_FLOW_GRID_CELLS:
            raise LiveScenarioValidationError(
                f"live flow grid must contain 2..{MAX_FLOW_GRID_CELLS} cells"
            )
        if not isinstance(raw_bounds, Mapping):
            raise LiveScenarioValidationError("live flow field bounds must be an object")
        bounds = (
            _finite_float(raw_bounds.get("min_x"), "field min_x", digits=6),
            _finite_float(raw_bounds.get("max_x"), "field max_x", digits=6),
            _finite_float(raw_bounds.get("min_y"), "field min_y", digits=6),
            _finite_float(raw_bounds.get("max_y"), "field max_y", digits=6),
        )
        if bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
            raise LiveScenarioValidationError("live flow field bounds must have positive area")
        if max(abs(value) for value in bounds) > MAX_ABSOLUTE_LOCAL_COORDINATE_M:
            raise LiveScenarioValidationError("live flow field bounds exceed request limits")
        count = nx * ny
        ux = np.asarray(field.get("ux"), dtype=np.float32)
        uy = np.asarray(field.get("uy"), dtype=np.float32)
        mask = np.asarray(field.get("mask"), dtype=np.uint8)
        if ux.shape != (count,) or uy.shape != (count,) or mask.shape != (count,):
            raise LiveScenarioValidationError("live flow arrays do not match grid dimensions")
        if not np.all(np.isfinite(ux)) or not np.all(np.isfinite(uy)):
            raise LiveScenarioValidationError("live flow velocity arrays must be finite")
        if np.any((mask != 0) & (mask != 1)):
            raise LiveScenarioValidationError("live flow mask must contain only 0 or 1")
        cell_size = _finite_float(
            field.get("cell_size_m", (bounds[1] - bounds[0]) / max(nx - 1, 1)),
            "field cell_size_m",
            minimum=0.1,
            maximum=80.0,
            digits=6,
        )
        digest = hashlib.sha256()
        digest.update(
            _canonical_json_bytes(
                {
                    "nx": nx,
                    "ny": ny,
                    "bounds_xy_m": list(bounds),
                    "cell_size_m": cell_size,
                    "array_dtype": "little_endian_float32_row_major_ix_ny_plus_iy",
                }
            )
        )
        digest.update(np.asarray(ux, dtype="<f4").tobytes(order="C"))
        digest.update(np.asarray(uy, dtype="<f4").tobytes(order="C"))
        digest.update(np.asarray(mask, dtype=np.uint8).tobytes(order="C"))
        ux = np.array(ux, dtype=np.float32, copy=True)
        uy = np.array(uy, dtype=np.float32, copy=True)
        mask = np.array(mask, dtype=np.uint8, copy=True)
        ux.setflags(write=False)
        uy.setflags(write=False)
        mask.setflags(write=False)
        return cls(nx, ny, bounds, cell_size, ux, uy, mask, digest.hexdigest())

    def source_metadata(self) -> dict[str, Any]:
        return {
            "kind": "registered_browser_cfd_lite_grid",
            "model": "existing_polygon_potential_flow_cfd_lite_wall_damping_wake_correction",
            "grid_digest_sha256": self.digest_sha256,
            "grid_shape": [self.nx, self.ny],
            "navier_stokes_cfd": False,
            "synthetic_hidden_flow": True,
            "real_cfd_validation": False,
            "full_grid_hidden_from_actor": True,
            "allowed_uses": ["dynamics", "reward", "metrics"],
        }


class RegisteredFlowGridWindProvider:
    """Read-only bilinear sampler matching the browser's resolved-grid sampler."""

    def __init__(self, field: FrozenFlowField2D, inlet_velocity_xy: np.ndarray) -> None:
        inlet = np.asarray(inlet_velocity_xy, dtype=float)
        if inlet.shape != (2,) or not np.all(np.isfinite(inlet)):
            raise LiveScenarioValidationError("registered inlet must be a finite 2D vector")
        self._field = field
        self._inlet = np.array(inlet, dtype=float, copy=True)
        self._inlet.setflags(write=False)

    @property
    def inlet_velocity(self) -> np.ndarray:
        return self._inlet.copy()

    def velocity_at(self, position_xy: np.ndarray, time_s: float) -> np.ndarray:
        del time_s
        point = np.asarray(position_xy, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("wind sample position must be a finite 2D vector")
        min_x, max_x, min_y, max_y = self._field.bounds_xy
        if point[0] < min_x or point[0] > max_x or point[1] < min_y or point[1] > max_y:
            return self._inlet.copy()
        gx = (point[0] - min_x) / max(max_x - min_x, 1e-9) * (self._field.nx - 1)
        gy = (point[1] - min_y) / max(max_y - min_y, 1e-9) * (self._field.ny - 1)
        x0 = int(np.clip(math.floor(gx), 0, self._field.nx - 1))
        y0 = int(np.clip(math.floor(gy), 0, self._field.ny - 1))
        x1 = min(x0 + 1, self._field.nx - 1)
        y1 = min(y0 + 1, self._field.ny - 1)
        nearest_x = int(np.clip(math.floor(gx + 0.5), 0, self._field.nx - 1))
        nearest_y = int(np.clip(math.floor(gy + 0.5), 0, self._field.ny - 1))
        if self._field.mask[nearest_x * self._field.ny + nearest_y] > 0:
            return np.zeros(2, dtype=float)
        tx = gx - x0
        ty = gy - y0

        def bilerp(values: np.ndarray) -> float:
            c00 = float(values[x0 * self._field.ny + y0])
            c10 = float(values[x1 * self._field.ny + y0])
            c01 = float(values[x0 * self._field.ny + y1])
            c11 = float(values[x1 * self._field.ny + y1])
            c0 = c00 * (1.0 - tx) + c10 * tx
            c1 = c01 * (1.0 - tx) + c11 * tx
            return c0 * (1.0 - ty) + c1 * ty

        return np.array([bilerp(self._field.ux), bilerp(self._field.uy)], dtype=float)

    def source_metadata(self) -> dict[str, Any]:
        return self._field.source_metadata()


@dataclass(frozen=True)
class LiveScenarioRecord:
    canonical_snapshot_bytes: bytes
    flow_field: FrozenFlowField2D

    @property
    def scenario_id(self) -> str:
        return str(self.snapshot()["scenario_id"])

    def snapshot(self) -> dict[str, Any]:
        return json.loads(self.canonical_snapshot_bytes.decode("utf-8"))

    def summary(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        return live_scenario_summary(snapshot)


def build_live_scenario_record(
    *,
    lat: float,
    lon: float,
    geometry_radius_m: float,
    solve_radius_m: float,
    buildings: Iterable[Mapping[str, Any]],
    weather: Mapping[str, Any],
    inlet_velocity_xy: Iterable[float],
    field: Mapping[str, Any],
    flow_source: Mapping[str, Any],
) -> LiveScenarioRecord:
    raw_buildings = list(buildings)
    latitude = _finite_float(lat, "selected latitude", minimum=-90.0, maximum=90.0, digits=8)
    longitude = _finite_float(lon, "selected longitude", minimum=-180.0, maximum=180.0, digits=8)
    geometry_radius = _finite_float(
        geometry_radius_m, "geometry radius", minimum=50.0, maximum=1_000.0, digits=3
    )
    solve_radius = _finite_float(
        solve_radius_m, "solve radius", minimum=200.0, maximum=3_000.0, digits=3
    )
    if geometry_radius > solve_radius:
        raise LiveScenarioValidationError("geometry radius cannot exceed solve radius")
    frozen_field = FrozenFlowField2D.from_payload(field)
    bounds = frozen_field.bounds_xy
    canonical_buildings = canonicalize_buildings(raw_buildings, bounds)

    inlet = np.asarray(tuple(inlet_velocity_xy), dtype=float)
    if inlet.shape != (2,) or not np.all(np.isfinite(inlet)):
        raise LiveScenarioValidationError("inlet velocity must be a finite 2D vector")
    if float(np.linalg.norm(inlet)) > 100.0:
        raise LiveScenarioValidationError("inlet velocity exceeds the 100 m/s request limit")
    inlet_values = [
        _finite_float(inlet[0], "inlet ux", digits=7),
        _finite_float(inlet[1], "inlet uy", digits=7),
    ]
    weather_source = weather.get("source") if isinstance(weather, Mapping) else None
    weather_source = weather_source if isinstance(weather_source, Mapping) else {"kind": "unspecified"}
    timestamp = weather_source.get("observation_time") or weather.get("timestamp")

    projected_crs_values = {
        str(source.get("projected_crs"))
        for raw in raw_buildings
        if isinstance(raw, Mapping)
        and isinstance((source := raw.get("source")), Mapping)
        and source.get("projected_crs") is not None
    }
    projected_crs = (
        sorted(projected_crs_values)[0]
        if len(projected_crs_values) == 1
        else "local_projected_crs_not_reported"
    )
    hidden_flow = frozen_field.source_metadata()
    content = {
        "schema_id": LIVE_SCENARIO_SCHEMA_ID,
        "schema_version": LIVE_SCENARIO_SCHEMA_VERSION,
        "location": {
            "selected_lat_deg": latitude,
            "selected_lon_deg": longitude,
            "geometry_radius_m": geometry_radius,
            "solve_radius_m": solve_radius,
        },
        "coordinate_frame": {
            "name": "urban_flighter_local_projected_xy",
            "origin": {"lat_deg": latitude, "lon_deg": longitude},
            "horizontal_units": "m",
            "vertical_units": "m",
            "x_axis": "east",
            "y_axis": "north",
            "z_axis": "up",
            "handedness": "right_handed",
            "projected_crs": projected_crs,
            "browser_3d_mapping": "scene_x=local_x, scene_y=height, scene_z=-local_y",
        },
        "bounds": {
            "min_x_m": bounds[0],
            "max_x_m": bounds[1],
            "min_y_m": bounds[2],
            "max_y_m": bounds[3],
        },
        "structure_count": len(canonical_buildings),
        "buildings": canonical_buildings,
        "inlet": {
            "velocity_xy_mps": inlet_values,
            "speed_mps": _finite_float(np.linalg.norm(inlet), "inlet speed", digits=7),
            "direction_from_north_deg": _finite_float(
                weather.get("wind_deg", 0.0),
                "weather wind direction",
                minimum=0.0,
                maximum=360.0,
                digits=6,
            ),
            "timestamp": None if timestamp is None else _bounded_text(timestamp, "inlet timestamp", 120),
            "timestamp_status": "reported_by_source" if timestamp is not None else "not_reported_by_source",
            "source": _json_safe(weather_source, "weather source"),
            "fallback": _json_safe(weather.get("fallback", {}), "weather fallback"),
        },
        "provenance": {
            "geometry": {
                "provider": "openstreetmap",
                "adapter": "osmnx",
                "footprint_geometry": "projected OSM polygon exterior",
                "height_policy": "OSM height, OSM building:levels estimate, or labelled deterministic default",
            },
            "weather": _json_safe(weather, "weather"),
            "flow_response": _json_safe(flow_source, "flow source"),
            "registration_source": "same successful POST /flow-fields/2d response returned to the UI",
        },
        "hidden_flow": hidden_flow,
        "collision_lidar_semantics": {
            "dimension": "2d_horizontal_slice",
            "building_model": "exact polygon exterior; every positive-height returned structure is solid",
            "collision_model": "swept circular agent against polygon edges/interiors and solid rectangular domain boundary",
            "agent_radius_m": LIVE_AGENT_RADIUS_M,
            "lidar_intersection_model": "exact ray-to-polygon-edge plus solid rectangular domain boundary",
            "gym_actor_lidar": {
                "ray_count": LIVE_LIDAR_RAY_COUNT,
                "max_range_m": LIVE_LIDAR_RANGE_M,
                "ordering": "counter_clockwise_vehicle_local_starting_forward",
            },
            "browser_display_lidar": {
                "ray_count": 180,
                "max_range_m": 180.0,
                "intersection_model_shared_with_gym": True,
            },
        },
        "policy_boundaries": {
            "full_flow_access": False,
            "synthetic_hidden_flow": True,
            "trained_policy_available": False,
            "real_cfd_validation_run": False,
        },
    }
    content_hash = hashlib.sha256(_canonical_json_bytes(content)).hexdigest()
    snapshot = {
        **content,
        "scenario_id": f"{LIVE_SCENARIO_ID_PREFIX}{content_hash[:24]}",
        "content_hash_sha256": content_hash,
    }
    return LiveScenarioRecord(_canonical_json_bytes(snapshot), frozen_field)


def live_scenario_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_id": snapshot["schema_id"],
        "schema_version": snapshot["schema_version"],
        "scenario_id": snapshot["scenario_id"],
        "content_hash_sha256": snapshot["content_hash_sha256"],
        "location": _json_safe(snapshot["location"], "location"),
        "coordinate_frame": _json_safe(snapshot["coordinate_frame"], "coordinate frame"),
        "bounds": _json_safe(snapshot["bounds"], "bounds"),
        "structure_count": int(snapshot["structure_count"]),
        "inlet": _json_safe(snapshot["inlet"], "inlet"),
        "provenance": _json_safe(snapshot["provenance"], "provenance"),
        "hidden_flow": _json_safe(snapshot["hidden_flow"], "hidden flow"),
        "collision_lidar_semantics": _json_safe(
            snapshot["collision_lidar_semantics"], "collision lidar semantics"
        ),
        "policy_boundaries": _json_safe(snapshot["policy_boundaries"], "policy boundaries"),
    }


def snapshot_buildings_for_flow(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "building_id": building["building_id"],
            "height": building["height_m"],
            "height_source": building["height_source"],
            "footprint": [point[:] for point in building["footprint_xy_m"]],
        }
        for building in snapshot["buildings"]
    ]


def snapshot_to_geometry(snapshot: Mapping[str, Any]) -> UrbanGeometry:
    bounds = snapshot["bounds"]
    geometry_bounds = (
        float(bounds["min_x_m"]),
        float(bounds["max_x_m"]),
        float(bounds["min_y_m"]),
        float(bounds["max_y_m"]),
    )
    prisms = tuple(
        PolygonPrism(
            obstacle_id=str(building["building_id"]),
            footprint_xy=np.asarray(building["footprint_xy_m"], dtype=float),
            height_m=float(building["height_m"]),
        )
        for building in snapshot["buildings"]
    )
    # The live Gym is the same horizontal footprint slice as the browser's 2D
    # simulator: all returned positive-height structures remain obstacles.
    return UrbanGeometry(geometry_bounds, prisms, flight_altitude_m=0.0)


def _validated_override(
    value: Iterable[float] | None,
    name: str,
    geometry: UrbanGeometry,
) -> np.ndarray | None:
    if value is None:
        return None
    point = np.asarray(tuple(value), dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise LiveScenarioValidationError(f"{name} must be a finite [x, y] vector")
    if not geometry.point_is_free(point, LIVE_AGENT_RADIUS_M, margin_m=1.0):
        raise LiveScenarioValidationError(
            f"{name} is outside bounds or collides with live scenario geometry"
        )
    return point


def derive_live_start_goal(
    geometry: UrbanGeometry,
    *,
    start_xy: Iterable[float] | None = None,
    goal_xy: Iterable[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    start = _validated_override(start_xy, "start_xy", geometry)
    goal = _validated_override(goal_xy, "goal_xy", geometry)
    if (start is None) != (goal is None):
        raise LiveScenarioValidationError("start_xy and goal_xy overrides must be provided together")
    if start is not None and goal is not None:
        if float(np.linalg.norm(goal - start)) <= LIVE_GOAL_RADIUS_M * 2.0:
            raise LiveScenarioValidationError("start_xy and goal_xy overrides are too close")
        return start, goal

    x0, x1, y0, y1 = geometry.bounds_xy
    inset = LIVE_AGENT_RADIUS_M + 3.0
    if x1 - x0 <= inset * 2.0 or y1 - y0 <= inset * 2.0:
        raise LiveScenarioValidationError("live scenario bounds are too small for a safe mission")
    grid_size = 19
    xs = np.linspace(x0 + inset, x1 - inset, grid_size)
    ys = np.linspace(y0 + inset, y1 - inset, grid_size)
    free: set[tuple[int, int]] = set()
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            point = np.array([x, y], dtype=float)
            if geometry.point_is_free(point, LIVE_AGENT_RADIUS_M, margin_m=1.0):
                free.add((ix, iy))
    if len(free) < 2:
        raise LiveScenarioValidationError("no safe start/goal pair exists in the live geometry")

    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    components: list[list[tuple[int, int]]] = []
    remaining = set(free)
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        component: list[tuple[int, int]] = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            component.append(node)
            node_point = np.array([xs[node[0]], ys[node[1]]], dtype=float)
            for dx, dy in neighbors:
                neighbor = (node[0] + dx, node[1] + dy)
                if neighbor not in remaining:
                    continue
                neighbor_point = np.array([xs[neighbor[0]], ys[neighbor[1]]], dtype=float)
                if not geometry.segment_is_free(
                    node_point,
                    neighbor_point,
                    LIVE_AGENT_RADIUS_M,
                    margin_m=1.0,
                ):
                    continue
                remaining.remove(neighbor)
                queue.append(neighbor)
        components.append(sorted(component))
    component = max(components, key=lambda nodes: (len(nodes), tuple(nodes)))
    if len(component) < 2:
        raise LiveScenarioValidationError("live free space has no connected start/goal pair")

    best_pair: tuple[tuple[int, int], tuple[int, int]] | None = None
    best_distance_squared = -1.0
    for first_index, first_node in enumerate(component[:-1]):
        first_point = np.array([xs[first_node[0]], ys[first_node[1]]], dtype=float)
        for second_node in component[first_index + 1 :]:
            second_point = np.array([xs[second_node[0]], ys[second_node[1]]], dtype=float)
            distance_squared = float(np.dot(second_point - first_point, second_point - first_point))
            pair = (first_node, second_node)
            if distance_squared > best_distance_squared + 1e-9 or (
                abs(distance_squared - best_distance_squared) <= 1e-9
                and (best_pair is None or pair < best_pair)
            ):
                best_distance_squared = distance_squared
                best_pair = pair
    if best_pair is None or math.sqrt(best_distance_squared) < geometry.diagonal_m * 0.25:
        raise LiveScenarioValidationError("live free space cannot provide a sufficiently separated mission")
    start = np.array([xs[best_pair[0][0]], ys[best_pair[0][1]]], dtype=float)
    goal = np.array([xs[best_pair[1][0]], ys[best_pair[1][1]]], dtype=float)
    return start, goal


def make_live_scenario(
    record: LiveScenarioRecord,
    *,
    seed: int,
    start_xy: Iterable[float] | None = None,
    goal_xy: Iterable[float] | None = None,
) -> UrbanFlowScenario:
    snapshot = record.snapshot()
    geometry = snapshot_to_geometry(snapshot)
    start, goal = derive_live_start_goal(
        geometry,
        start_xy=start_xy,
        goal_xy=goal_xy,
    )
    inlet = np.asarray(snapshot["inlet"]["velocity_xy_mps"], dtype=float)
    provider = RegisteredFlowGridWindProvider(record.flow_field, inlet)
    heading = math.atan2(goal[1] - start[1], goal[0] - start[0])
    return UrbanFlowScenario(
        scenario_id=str(snapshot["scenario_id"]),
        seed=int(seed),
        geometry=geometry,
        start_xy=start,
        goal_xy=goal,
        initial_heading_rad=heading,
        known_inlet_velocity_xy=inlet,
        wind_provider=provider,
        randomization={
            "enabled": False,
            "live_snapshot": True,
            "seed_changes_world": False,
            "registered_flow_grid_digest_sha256": record.flow_field.digest_sha256,
        },
    )


class LiveScenarioRegistry:
    """Bounded process-local registry with atomic current-scenario selection."""

    def __init__(self, max_entries: int = LIVE_SCENARIO_REGISTRY_MAX_ENTRIES) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        self._records: OrderedDict[str, LiveScenarioRecord] = OrderedDict()
        self._current_scenario_id: str | None = None
        self._lock = RLock()

    def register(self, record: LiveScenarioRecord) -> dict[str, Any]:
        scenario_id = record.scenario_id
        with self._lock:
            if scenario_id in self._records:
                existing = self._records.pop(scenario_id)
                if existing.canonical_snapshot_bytes != record.canonical_snapshot_bytes:
                    raise LiveScenarioError("content-addressed scenario id collision")
                record = existing
            self._records[scenario_id] = record
            self._current_scenario_id = scenario_id
            while len(self._records) > self.max_entries:
                evicted_id, _ = self._records.popitem(last=False)
                if evicted_id == self._current_scenario_id:
                    raise LiveScenarioError("registry attempted to evict the current live scenario")
            summary = record.summary()
            summary["is_current"] = True
            summary["registry_size"] = len(self._records)
            return summary

    def activate(self, scenario_id: str) -> dict[str, Any]:
        with self._lock:
            try:
                record = self._records.pop(scenario_id)
            except KeyError as exc:
                raise UnknownLiveScenarioError(
                    f"live scenario '{scenario_id}' is invalid, stale, or no longer cached"
                ) from exc
            self._records[scenario_id] = record
            self._current_scenario_id = scenario_id
            summary = record.summary()
            summary["is_current"] = True
            summary["registry_size"] = len(self._records)
            return summary

    def get_record(self, scenario_id: str | None = None) -> LiveScenarioRecord:
        with self._lock:
            resolved_id = scenario_id or self._current_scenario_id
            if resolved_id is None:
                raise NoLiveScenarioError(
                    "no live UrbanFlow scenario is loaded; successfully load a location through POST /flow-fields/2d first"
                )
            try:
                return self._records[resolved_id]
            except KeyError as exc:
                raise UnknownLiveScenarioError(
                    f"live scenario '{resolved_id}' is invalid, stale, or no longer cached"
                ) from exc

    def current_summary(self) -> dict[str, Any]:
        with self._lock:
            record = self.get_record()
            summary = record.summary()
            summary["is_current"] = True
            summary["registry_size"] = len(self._records)
            return summary

    def snapshot(self, scenario_id: str | None = None) -> dict[str, Any]:
        record = self.get_record(scenario_id)
        snapshot = record.snapshot()
        with self._lock:
            snapshot["is_current"] = record.scenario_id == self._current_scenario_id
        return snapshot

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
            self._current_scenario_id = None

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


live_scenario_registry = LiveScenarioRegistry()
