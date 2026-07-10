from __future__ import annotations

from pathlib import Path

import numpy as np

from .potential_flow import PotentialFlowResult
from .world import UrbanWorld


def plot_potential_flow_3d(
    world: UrbanWorld,
    slices: list[tuple[float, PotentialFlowResult]],
    output_path: str | Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Urban_Flighter 3D potential-flow CFD-lite slices")

    for b in world.buildings:
        x0, y0 = b.min_xy
        x1, y1 = b.max_xy
        z0, z1 = 0.0, b.height
        verts = [
            [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
            [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
            [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
            [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
            [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        ]
        poly = Poly3DCollection(verts, alpha=0.18, facecolor="#334155", edgecolor="#0f172a", linewidth=0.35)
        ax.add_collection3d(poly)

    colors = ["#38bdf8", "#fbbf24", "#a78bfa", "#34d399", "#fb7185"]
    for idx, (alt, result) in enumerate(slices):
        color = colors[idx % len(colors)]
        for line in result.streamlines:
            arr = np.asarray(line, dtype=float)
            if len(arr) < 2:
                continue
            z = np.full(len(arr), float(alt))
            ax.plot(arr[:, 0], arr[:, 1], z, color=color, linewidth=0.8, alpha=0.72)

        skip = max(1, len(result.x) // 16)
        xx, yy = np.meshgrid(result.x[::skip], result.y[::skip], indexing="ij")
        uu = result.ux[::skip, ::skip]
        vv = result.uy[::skip, ::skip]
        zz = np.full_like(xx, float(alt), dtype=float)
        ax.quiver(xx, yy, zz, uu, vv, np.zeros_like(uu), length=3.0, normalize=True, color=color, alpha=0.35)

    x0, x1, y0, y1, z0, z1 = world.bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(0, max(z1, max((a for a, _ in slices), default=50) + 20))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("altitude [m]")
    ax.view_init(elev=36, azim=-58)
    ax.text2D(
        0.02,
        0.03,
        "3D visualization of stacked 2D potential-flow slices. CFD-lite, not full 3D CFD.",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
