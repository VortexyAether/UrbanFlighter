import osmnx as ox
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
import math
import re


DEFAULT_BUILDING_HEIGHT_M = 10.0
METRES_PER_BUILDING_LEVEL = 3.5


def _first_positive_number(value) -> float | None:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value).replace(",", "."))
    if match is None:
        return None
    try:
        parsed = float(match.group(0))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) and parsed > 0.0 else None


def _building_height(row) -> tuple[float, str]:
    if "height" in row and pd.notnull(row["height"]):
        parsed = _first_positive_number(row["height"])
        if parsed is not None:
            return parsed, "osm:height"
    if "building:levels" in row and pd.notnull(row["building:levels"]):
        levels = _first_positive_number(row["building:levels"])
        if levels is not None:
            return levels * METRES_PER_BUILDING_LEVEL, "osm:building:levels_estimate_3.5m_per_level"
    return DEFAULT_BUILDING_HEIGHT_M, "deterministic_default_missing_osm_height"


def _osm_element_id(index) -> str:
    if isinstance(index, tuple):
        return ":".join(str(part) for part in index)
    return str(index)

def fetch_buildings(lat: float, lon: float, radius: float = 300):
    """
    Fetch buildings from OSM using OSMnx (Slow but Real, No Key required).
    """
    print(f"Fetching buildings via OSMnx at {lat}, {lon} (Radius: {radius}m)")
    try:
        # Fetch geometries (Can be slow)
        tags = {"building": True}
        gdf = ox.features_from_point((lat, lon), tags=tags, dist=radius)
        
        if gdf.empty:
            print("OSMnx found no buildings.")
            return []

        # Project to local UTM (meters)
        try:
            gdf_proj = ox.project_gdf(gdf)
        except Exception as e:
            # Fallback if projection fails (sometimes happens with empty/weird geoms)
            print(f"Projection failed: {e}. Trying automatic UTM estimation.")
            gdf_proj = gdf.to_crs(epsg=3857) # Web Mercator as fallback

        # Get center point in projected coords
        # We project a point at (lat, lon) to the SAME CRS as gdf_proj
        center_series = gpd.GeoSeries([gpd.points_from_xy([lon], [lat])[0]], crs="EPSG:4326")
        center_proj = center_series.to_crs(gdf_proj.crs).iloc[0]
        center_x, center_y = center_proj.x, center_proj.y

        buildings = []
        
        projected_crs = str(gdf_proj.crs)

        # Iterate and extract polygons
        for idx, row in gdf_proj.iterrows():
            geom = row.geometry
            if geom.is_empty:
                continue

            height, height_source = _building_height(row)

            # Extract footprint(s)
            polys = []
            if isinstance(geom, Polygon):
                polys = [geom]
            elif isinstance(geom, MultiPolygon):
                polys = list(geom.geoms)

            for part_index, poly in enumerate(polys):
                # Exterior coords
                xx, yy = poly.exterior.coords.xy
                # Shift to local origin (center_x, center_y) -> (0,0)
                local_coords = []
                for x, y in zip(xx, yy):
                    local_x = float(x - center_x)
                    local_y = float(y - center_y)
                    if not math.isfinite(local_x) or not math.isfinite(local_y):
                        local_coords = []
                        break
                    local_coords.append([local_x, local_y])
                
                # Filter out degenerate polygons
                if len(local_coords) < 3:
                     continue
                
                # Circular boundary check: filter buildings outside radius
                # Check if building centroid is within circular domain
                centroid_x = sum(c[0] for c in local_coords) / len(local_coords)
                centroid_z = sum(c[1] for c in local_coords) / len(local_coords)
                dist_from_center = math.hypot(centroid_x, centroid_z)
                
                if dist_from_center > radius:
                    continue  # Skip buildings outside circular boundary

                buildings.append({
                    "building_id": f"{_osm_element_id(idx)}:part:{part_index}",
                    "height": float(height),
                    "height_source": height_source,
                    "footprint": local_coords,
                    "source": {
                        "provider": "openstreetmap",
                        "adapter": "osmnx",
                        "element_id": _osm_element_id(idx),
                        "part_index": part_index,
                        "projected_crs": projected_crs,
                    },
                })

        print(f"OSMnx: Fetched {len(buildings)} buildings.")
        return buildings

    except Exception as e:
        print(f"Error fetching buildings with OSMnx: {e}")
        return []
