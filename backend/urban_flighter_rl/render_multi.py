from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_multi_drone_trajectories(world, wind, runs, output_path: str | Path, satellite_tile_path: str | Path | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Urban Flighter: multi-drone RL-style rollout over real OSM buildings")

    # Optional rough satellite context as ground texture. It is geographically contextual,
    # not yet tile-accurately warped; enough to make the data-source link visible.
    if satellite_tile_path and Path(satellite_tile_path).exists():
        try:
            import matplotlib.image as mpimg
            img = mpimg.imread(satellite_tile_path)
            x0, x1, y0, y1, z0, _ = world.bounds
            xs = np.linspace(x0, x1, img.shape[1])
            ys = np.linspace(y0, y1, img.shape[0])
            X, Y = np.meshgrid(xs, ys)
            Z = np.full_like(X, z0 - 0.5, dtype=float)
            ax.plot_surface(X, Y, Z, rstride=16, cstride=16, facecolors=img / 255.0 if img.dtype != float else img,
                            shade=False, alpha=0.42, linewidth=0)
        except Exception:
            pass

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
        poly = Poly3DCollection(verts, alpha=0.20, facecolor="#52616b", edgecolor="#1f2933", linewidths=0.25)
        ax.add_collection3d(poly)

    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(10, len(runs))))
    for i, run in enumerate(runs):
        traj = np.asarray(run.trajectory)
        color = colors[i % len(colors)]
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=2.5, color=color, label=run.drone_id)
        ax.scatter(*run.start, color=color, marker="o", s=55)
        ax.scatter(*run.goal, color=color, marker="*", s=120, edgecolor="black")
        # Show current/final drone position as a larger marker.
        ax.scatter(*traj[-1], color=color, marker="^", s=90, edgecolor="black")

    x0, x1, y0, y1, z0, z1 = world.bounds
    xs = np.linspace(x0 + 0.15 * (x1 - x0), x1 - 0.15 * (x1 - x0), 5)
    ys = np.linspace(y0 + 0.15 * (y1 - y0), y1 - 0.15 * (y1 - y0), 5)
    z = min(max(30.0, z0 + 0.4 * (z1 - z0)), z1 - 5.0)
    pts = np.array([[x, y, z] for x in xs for y in ys])
    winds = np.array([wind.at(p, 0.0) for p in pts])
    ax.quiver(pts[:, 0], pts[:, 1], pts[:, 2], winds[:, 0], winds[:, 1], winds[:, 2],
              length=7.0, normalize=True, color="#0077b6", alpha=0.55)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_zlim(z0 - 1.0, z1)
    ax.set_xlabel("local east/west x [m]")
    ax.set_ylabel("local north/south y [m]")
    ax.set_zlabel("altitude z [m]")
    ax.legend(loc="upper left", fontsize=8)
    ax.view_init(elev=34, azim=-58)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
