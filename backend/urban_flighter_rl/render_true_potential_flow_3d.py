from __future__ import annotations

from pathlib import Path

import numpy as np

from .potential_flow_3d import PotentialFlow3DResult
from .world import UrbanWorld


def plot_true_potential_flow_3d(world: UrbanWorld, result: PotentialFlow3DResult, output_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Urban_Flighter true 3D potential-flow CFD-lite")

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
        ax.add_collection3d(Poly3DCollection(verts, alpha=0.15, facecolor="#334155", edgecolor="#0f172a", linewidth=0.35))

    colors = plt.cm.turbo(np.linspace(0.05, 0.95, max(len(result.streamlines), 1)))
    for color, line in zip(colors, result.streamlines):
        arr = np.asarray(line, dtype=float)
        if len(arr) >= 2:
            ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], color=color, linewidth=1.0, alpha=0.78)

    skip_x = max(1, len(result.x) // 11)
    skip_y = max(1, len(result.y) // 11)
    skip_z = max(1, len(result.z) // 6)
    xx, yy, zz = np.meshgrid(result.x[::skip_x], result.y[::skip_y], result.z[::skip_z], indexing="ij")
    uu = result.ux[::skip_x, ::skip_y, ::skip_z]
    vv = result.uy[::skip_x, ::skip_y, ::skip_z]
    ww = result.uz[::skip_x, ::skip_y, ::skip_z]
    mag = np.sqrt(uu * uu + vv * vv + ww * ww)
    keep = mag > 0.25
    ax.quiver(xx[keep], yy[keep], zz[keep], uu[keep], vv[keep], ww[keep], length=3.2, normalize=True, color="#fbbf24", alpha=0.22)

    x0, x1, y0, y1, z0, z1 = world.bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.view_init(elev=34, azim=-58)
    ax.text2D(
        0.02,
        0.03,
        "3D Laplace potential-flow approximation. Includes u/v/w, but no turbulence or real wake separation.",
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
