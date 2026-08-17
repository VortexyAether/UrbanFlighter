from __future__ import annotations

import logging
import math
import random
from typing import Any

import requests


OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEOUT_S = 5
FALLBACK_WIND_SPEED_MPS = 5.0
FALLBACK_WIND_DIRECTION_DEG = 0.0

_WIND_UNITS = {
    "wind_speed": "m/s",
    "wind_deg": "degrees_from_north",
}
_LOGGER = logging.getLogger(__name__)


def _fallback_weather(reason: str) -> dict[str, Any]:
    """Return stable, explicitly labelled values when live weather is unavailable."""
    return {
        "wind_speed": FALLBACK_WIND_SPEED_MPS,
        "wind_deg": FALLBACK_WIND_DIRECTION_DEG,
        "description": "Deterministic fallback wind; live weather unavailable",
        "units": dict(_WIND_UNITS),
        "source": {
            "provider": "urban-flighter",
            "kind": "deterministic_fallback",
            "upstream_provider": "open-meteo",
        },
        "fallback": {
            "used": True,
            "reason": reason,
        },
    }


def get_real_weather(lat: float, lon: float) -> dict[str, Any]:
    """Fetch Open-Meteo's forecast-model current 10 m wind in metres/second.

    The legacy consumer fields (``wind_speed``, ``wind_deg`` and
    ``description``) remain present. Structured units, provenance and fallback
    metadata prevent modelled or fallback wind from being mistaken for direct
    sensor/satellite observations.
    """
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "current": "wind_speed_10m,wind_direction_10m",
        # Open-Meteo defaults to km/h, so this must always be explicit.
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "forecast_days": 1,
    }

    try:
        response = requests.get(
            OPEN_METEO_FORECAST_URL,
            params=params,
            timeout=OPEN_METEO_TIMEOUT_S,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("weather response is not an object")

        current = data.get("current")
        if not isinstance(current, dict):
            raise ValueError("weather response has no current conditions")

        speed_mps = float(current["wind_speed_10m"])
        direction_deg = float(current["wind_direction_10m"])
        if not math.isfinite(speed_mps) or speed_mps < 0.0:
            raise ValueError("weather response has invalid wind speed")
        if not math.isfinite(direction_deg):
            raise ValueError("weather response has invalid wind direction")

        current_units = data.get("current_units", {})
        if isinstance(current_units, dict):
            reported_speed_unit = current_units.get("wind_speed_10m")
            if reported_speed_unit not in (None, "m/s"):
                raise ValueError("weather response did not honour requested wind unit")

        source: dict[str, Any] = {
            "provider": "open-meteo",
            "kind": "forecast_model_current_conditions",
            "endpoint": OPEN_METEO_FORECAST_URL,
            "variable_height_m": 10.0,
            "timezone": "UTC",
        }
        if current.get("time") is not None:
            source["observation_time"] = str(current["time"])

        return {
            "wind_speed": speed_mps,
            "wind_deg": direction_deg % 360.0,
            "description": "Open-Meteo forecast-model current wind at 10 m",
            "units": dict(_WIND_UNITS),
            "source": source,
            "fallback": {
                "used": False,
                "reason": None,
            },
        }
    except requests.RequestException as exc:
        _LOGGER.warning("Open-Meteo request failed (%s)", type(exc).__name__)
        return _fallback_weather("weather_service_unavailable")
    except (KeyError, TypeError, ValueError) as exc:
        _LOGGER.warning("Open-Meteo response was unusable (%s)", type(exc).__name__)
        return _fallback_weather("invalid_weather_response")


def generate_global_wind_params() -> dict[str, Any]:
    """Return parameters for the frontend procedural wind shader/system."""
    return {
        "base_speed": random.uniform(5, 15),
        "direction": [random.uniform(-1, 1), 0, random.uniform(-1, 1)],
        "turbulence": 0.5,
    }
