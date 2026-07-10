from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
import urllib.parse
import urllib.request

import numpy as np

from .world import Building, UrbanWorld


OVERPASS_URL = "https://overpass-api.de/api/interpreter"
ESRI_WORLD_IMAGERY_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)


@dataclass(frozen=True)
class GeoReference:
    lat: float
    lon: float
    radius_m: float
    meters_per_deg_lat: float
    meters_per_deg_lon: float

    def lonlat_to_local_xy(self, lon: float, lat: float) -> tuple[float, float]:
        x = (lon - self.lon) * self.meters_per_deg_lon
        y = (lat - self.lat) * self.meters_per_deg_lat
        return x, y


def make_georef(lat: float, lon: float, radius_m: float) -> GeoReference:
    # Good enough for sub-km urban simulator domains.
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    return GeoReference(lat, lon, radius_m, meters_per_deg_lat, meters_per_deg_lon)


def _http_post_form(url: str, data: dict[str, str], timeout_s: float = 45.0) -> bytes:
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=encoded,
        headers={
            "User-Agent": "Urban_Flighter_OSM_Prototype/0.1 (+https://github.com/VortexyAether/Urban_Flighter)",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return resp.read()


def fetch_osm_building_elements(lat: float, lon: float, radius_m: float = 300.0, timeout_s: float = 45.0) -> dict:
    """Fetch real OSM building geometry from Overpass, no API key required."""
    query = f"""
[out:json][timeout:{int(timeout_s)}];
(
  way["building"](around:{float(radius_m)},{float(lat)},{float(lon)});
  relation["building"](around:{float(radius_m)},{float(lat)},{float(lon)});
);
out body geom;
""".strip()
    raw = _http_post_form(OVERPASS_URL, {"data": query}, timeout_s=timeout_s + 5.0)
    return json.loads(raw.decode("utf-8"))


def parse_height_m(tags: dict | None, default_height_m: float = 12.0) -> float:
    tags = tags or {}
    raw = tags.get("height") or tags.get("building:height")
    if raw:
        text = str(raw).lower().replace("meters", "").replace("meter", "").replace("m", "").strip()
        text = text.split(";")[0].replace(",", ".")
        try:
            return float(text)
        except ValueError:
            pass
    levels = tags.get("building:levels") or tags.get("levels")
    if levels:
        try:
            return max(3.5, float(str(levels).split(";")[0]) * 3.3)
        except ValueError:
            pass
    return float(default_height_m)


def osm_elements_to_buildings(payload: dict, georef: GeoReference, max_buildings: int = 160) -> list[Building]:
    buildings: list[Building] = []
    seen_centers: set[tuple[int, int]] = set()
    for el in payload.get("elements", []):
        geom = el.get("geometry") or []
        if len(geom) < 3:
            continue
        pts = np.array([georef.lonlat_to_local_xy(float(p["lon"]), float(p["lat"])) for p in geom], dtype=float)
        if len(pts) < 3:
            continue
        # Keep only footprints whose centroid is inside the requested circular domain.
        centroid = pts.mean(axis=0)
        if np.linalg.norm(centroid) > georef.radius_m:
            continue
        min_xy = pts.min(axis=0)
        max_xy = pts.max(axis=0)
        size = np.maximum(max_xy - min_xy, np.array([3.0, 3.0]))
        if size[0] < 2.0 or size[1] < 2.0:
            continue
        center = (min_xy + max_xy) / 2.0
        key = (int(center[0] * 2), int(center[1] * 2))
        if key in seen_centers:
            continue
        seen_centers.add(key)
        height = parse_height_m(el.get("tags"), default_height_m=12.0)
        buildings.append(Building(center=center, size=size, height=height))
        if len(buildings) >= max_buildings:
            break
    return buildings


def fetch_osm_world(lat: float, lon: float, radius_m: float = 300.0, max_buildings: int = 160) -> tuple[UrbanWorld, dict]:
    """Build an UrbanWorld from real OSM footprints around lat/lon."""
    georef = make_georef(lat, lon, radius_m)
    started = time.time()
    payload = fetch_osm_building_elements(lat, lon, radius_m=radius_m)
    buildings = osm_elements_to_buildings(payload, georef, max_buildings=max_buildings)
    if not buildings:
        raise RuntimeError(f"Overpass returned no usable building footprints for {lat}, {lon}, r={radius_m}m")
    max_height = max(b.height for b in buildings)
    bounds = (-radius_m, radius_m, -radius_m, radius_m, 0.0, max(70.0, min(180.0, max_height + 45.0)))
    world = UrbanWorld(bounds=bounds, buildings=buildings)
    meta = {
        "source": "OpenStreetMap Overpass API",
        "overpass_url": OVERPASS_URL,
        "lat": lat,
        "lon": lon,
        "radius_m": radius_m,
        "osm_elements": len(payload.get("elements", [])),
        "usable_buildings": len(buildings),
        "max_building_height_m": float(max_height),
        "fetch_elapsed_s": round(time.time() - started, 3),
        "bounds": list(bounds),
    }
    return world, meta


def deg2num(lat_deg: float, lon_deg: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def download_satellite_tile(lat: float, lon: float, output_path: str | Path, zoom: int = 17) -> dict:
    """Download the center satellite tile from Esri World Imagery."""
    x, y = deg2num(lat, lon, zoom)
    url = ESRI_WORLD_IMAGERY_TILE_URL.format(z=zoom, x=x, y=y)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Urban_Flighter_OSM_Prototype/0.1"})
    with urllib.request.urlopen(req, timeout=30.0) as resp:
        data = resp.read()
    output_path.write_bytes(data)
    return {
        "source": "Esri World Imagery tile API",
        "url": url,
        "lat": lat,
        "lon": lon,
        "zoom": zoom,
        "tile_x": x,
        "tile_y": y,
        "path": str(output_path),
        "bytes": len(data),
    }
