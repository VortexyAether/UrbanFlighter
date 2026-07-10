from __future__ import annotations

from pathlib import Path

import numpy as np

from .potential_flow import PotentialFlowResult
from .world import UrbanWorld


def plot_potential_flow(world: UrbanWorld, result: PotentialFlowResult, output_path: str | Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 9))
    speed = np.sqrt(result.ux * result.ux + result.uy * result.uy)
    img = ax.imshow(
        speed.T,
        origin="lower",
        extent=[float(result.x[0]), float(result.x[-1]), float(result.y[0]), float(result.y[-1])],
        cmap="viridis",
        alpha=0.86,
    )
    fig.colorbar(img, ax=ax, label="speed [m/s]")

    for line in result.streamlines:
        arr = np.asarray(line, dtype=float)
        ax.plot(arr[:, 0], arr[:, 1], color="white", linewidth=0.9, alpha=0.82)

    skip = max(1, len(result.x) // 28)
    xx, yy = np.meshgrid(result.x[::skip], result.y[::skip], indexing="ij")
    ax.quiver(
        xx,
        yy,
        result.ux[::skip, ::skip],
        result.uy[::skip, ::skip],
        color="#ffdd88",
        alpha=0.65,
        width=0.0022,
        scale=90,
    )

    for b in world.buildings:
        if result.meta["altitude_m"] > b.height:
            continue
        mn = b.min_xy
        size = b.size
        ax.add_patch(
            Rectangle(
                (float(mn[0]), float(mn[1])),
                float(size[0]),
                float(size[1]),
                facecolor="#111827",
                edgecolor="#e5e7eb",
                linewidth=0.5,
                alpha=0.9,
            )
        )

    ax.set_title("Urban_Flighter potential-flow CFD-lite wind slice")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.015,
        0.02,
        "Potential-flow CFD-lite. Not full CFD: no turbulence, no viscous separation.",
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
