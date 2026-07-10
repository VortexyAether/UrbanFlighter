from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from .potential_flow import PotentialFlowResult, trace_streamlines
from .world import UrbanWorld


def apply_wall_damping_and_wake(
    world: UrbanWorld,
    base: PotentialFlowResult,
    inlet_velocity: tuple[float, float] | np.ndarray,
    wall_length_m: float = 14.0,
    wake_strength: float = 0.55,
    wake_length_factor: float = 5.5,
) -> PotentialFlowResult:
    """Post-process potential flow for drone-risk realism.

    Adds two empirical corrections:
    - near-wall damping: speed tends toward zero close to building masks
    - downstream wake deficit: lower mean flow behind buildings

    This is not pure potential flow. It is a CFD-lite guidance/risk field.
    """
    inlet = np.asarray(inlet_velocity, dtype=float)
    speed0 = max(float(np.linalg.norm(inlet)), 1e-6)
    wind_hat = inlet / speed0
    cross_hat = np.array([-wind_hat[1], wind_hat[0]], dtype=float)
    cell = float(base.meta["cell_size_m"])
    # distance_transform_edt returns distance from fluid cell to nearest solid when input is fluid=True.
    dist_m = distance_transform_edt(~base.mask).astype(float) * cell
    damping = 1.0 - np.exp(-dist_m / max(wall_length_m, cell))
    damping = np.clip(damping, 0.0, 1.0)
    ux = base.ux * damping
    uy = base.uy * damping

    xx, yy = np.meshgrid(base.x, base.y, indexing="ij")
    deficit = np.zeros_like(ux, dtype=float)
    for b in world.buildings:
        if base.meta["altitude_m"] > b.height:
            continue
        rel_x = xx - float(b.center[0])
        rel_y = yy - float(b.center[1])
        along = rel_x * wind_hat[0] + rel_y * wind_hat[1]
        cross = rel_x * cross_hat[0] + rel_y * cross_hat[1]
        span = max(float(b.size[0]), float(b.size[1]), cell * 2.0)
        wake_len = max(span * wake_length_factor, cell * 8.0)
        wake_width = span * 0.7 + np.maximum(along, 0.0) * 0.22 + cell
        local = (along > 0.0) * np.exp(-np.maximum(along, 0.0) / wake_len) * np.exp(-(cross / np.maximum(wake_width, cell)) ** 2)
        deficit = np.maximum(deficit, local)
    factor = 1.0 - wake_strength * np.clip(deficit, 0.0, 1.0)
    ux *= factor
    uy *= factor
    ux[base.mask] = 0.0
    uy[base.mask] = 0.0
    streamlines = trace_streamlines(base.x, base.y, ux, uy, base.mask, inlet, count=max(24, int(base.meta.get("streamline_count", 24))))
    meta = {
        **base.meta,
        "model": "potential-flow-cfd-lite-wall-damping-wake-correction",
        "pure_potential_flow": False,
        "wall_damping_length_m": float(wall_length_m),
        "wake_strength": float(wake_strength),
        "wake_length_factor": float(wake_length_factor),
        "mean_speed_mps": float(np.sqrt(ux * ux + uy * uy).mean()),
        "max_speed_mps": float(np.sqrt(ux * ux + uy * uy).max()),
        "streamline_count": int(len(streamlines)),
    }
    return PotentialFlowResult(x=base.x, y=base.y, psi=base.psi, ux=ux, uy=uy, mask=base.mask, streamlines=streamlines, meta=meta)
