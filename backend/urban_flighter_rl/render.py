from __future__ import annotations

from pathlib import Path
import json
import numpy as np


def plot_trajectory(env, output_path: str | Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Urban Flighter 3D Prototype: Wind-aware path baseline")

    for b in env.world.buildings:
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
        poly = Poly3DCollection(verts, alpha=0.25, facecolor="#4455aa", edgecolor="#1a1a33")
        ax.add_collection3d(poly)

    traj = np.asarray(env.trajectory)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color="#e63946", linewidth=3, label="trajectory")
    ax.scatter(*env.trajectory[0], color="#2a9d8f", s=80, label="start")
    ax.scatter(*env.goal, color="#f4a261", s=100, marker="*", label="goal")

    # Wind vectors on a sparse 3D slice.
    x0, x1, y0, y1, z0, z1 = env.world.bounds
    xs = np.linspace(x0 + 0.12 * (x1 - x0), x1 - 0.12 * (x1 - x0), 6)
    ys = np.linspace(y0 + 0.12 * (y1 - y0), y1 - 0.12 * (y1 - y0), 6)
    z = min(max(25.0, z0 + 0.35 * (z1 - z0)), z1 - 5.0)
    pts = np.array([[x, y, z] for x in xs for y in ys])
    winds = np.array([env.wind.at(p, env.t) for p in pts])
    ax.quiver(pts[:, 0], pts[:, 1], pts[:, 2], winds[:, 0], winds[:, 1], winds[:, 2],
              length=3.5, normalize=True, color="#457b9d", alpha=0.6)

    x0, x1, y0, y1, z0, z1 = env.world.bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0, z1)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper left")
    ax.view_init(elev=28, azim=-62)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_metrics(metrics: dict, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
