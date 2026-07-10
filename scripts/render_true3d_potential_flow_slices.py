from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_field(field_path: Path):
    data = np.load(field_path)
    return data["x"], data["y"], data["z"], data["ux"], data["uy"], data["uz"], data["solid"]


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - float(target))))


def render_xy(x, y, z, ux, uy, uz, solid, z_value: float, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = _nearest_index(z, z_value)
    speed = np.sqrt(ux[:, :, k] ** 2 + uy[:, :, k] ** 2 + uz[:, :, k] ** 2)
    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(speed.T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="turbo")
    fig.colorbar(img, ax=ax, label="speed [m/s]")
    ax.contour(x, y, solid[:, :, k].T.astype(float), levels=[0.5], colors="white", linewidths=0.5)
    skip = max(1, len(x) // 42)
    xx, yy = np.meshgrid(x[::skip], y[::skip], indexing="ij")
    ax.quiver(xx, yy, ux[::skip, ::skip, k], uy[::skip, ::skip, k], color="black", alpha=0.55, scale=120, width=0.002)
    ax.streamplot(x, y, ux[:, :, k].T, uy[:, :, k].T, color="white", density=2.2, linewidth=0.55, arrowsize=0.45)
    ax.set_title(f"XY wind slice at z={z[k]:.1f}m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {"plane": "xy", "z_m": float(z[k]), "path": str(out)}


def render_xz(x, y, z, ux, uy, uz, solid, y_value: float, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    j = _nearest_index(y, y_value)
    speed = np.sqrt(ux[:, j, :] ** 2 + uy[:, j, :] ** 2 + uz[:, j, :] ** 2)
    fig, ax = plt.subplots(figsize=(11, 6))
    img = ax.imshow(speed.T, origin="lower", extent=[x[0], x[-1], z[0], z[-1]], aspect="auto", cmap="turbo")
    fig.colorbar(img, ax=ax, label="speed [m/s]")
    ax.contour(x, z, solid[:, j, :].T.astype(float), levels=[0.5], colors="white", linewidths=0.7)
    skip_x = max(1, len(x) // 48)
    skip_z = max(1, len(z) // 20)
    xx, zz = np.meshgrid(x[::skip_x], z[::skip_z], indexing="ij")
    ax.quiver(xx, zz, ux[::skip_x, j, ::skip_z], uz[::skip_x, j, ::skip_z], color="black", alpha=0.55, scale=110, width=0.002)
    ax.streamplot(x, z, ux[:, j, :].T, uz[:, j, :].T, color="white", density=2.0, linewidth=0.65, arrowsize=0.55)
    ax.set_title(f"XZ vertical wind slice at y={y[j]:.1f}m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z altitude [m]")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {"plane": "xz", "y_m": float(y[j]), "path": str(out)}


def render_yz(x, y, z, ux, uy, uz, solid, x_value: float, out: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    i = _nearest_index(x, x_value)
    speed = np.sqrt(ux[i, :, :] ** 2 + uy[i, :, :] ** 2 + uz[i, :, :] ** 2)
    fig, ax = plt.subplots(figsize=(10, 6))
    img = ax.imshow(speed.T, origin="lower", extent=[y[0], y[-1], z[0], z[-1]], aspect="auto", cmap="turbo")
    fig.colorbar(img, ax=ax, label="speed [m/s]")
    ax.contour(y, z, solid[i, :, :].T.astype(float), levels=[0.5], colors="white", linewidths=0.7)
    skip_y = max(1, len(y) // 44)
    skip_z = max(1, len(z) // 20)
    yy, zz = np.meshgrid(y[::skip_y], z[::skip_z], indexing="ij")
    ax.quiver(yy, zz, uy[i, ::skip_y, ::skip_z], uz[i, ::skip_y, ::skip_z], color="black", alpha=0.55, scale=110, width=0.002)
    ax.streamplot(y, z, uy[i, :, :].T, uz[i, :, :].T, color="white", density=1.8, linewidth=0.65, arrowsize=0.55)
    ax.set_title(f"YZ vertical wind slice at x={x[i]:.1f}m")
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z altitude [m]")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {"plane": "yz", "x_m": float(x[i]), "path": str(out)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Render diagnostic slices for true 3D potential-flow field")
    parser.add_argument("--field", default="results/potential_flow_gangnam_true3d_128x128x64/potential_flow_3d_field.npz")
    parser.add_argument("--out", default="results/potential_flow_gangnam_true3d_128x128x64/slices")
    parser.add_argument("--z", type=float, default=35.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--x", type=float, default=0.0)
    args = parser.parse_args()

    field = Path(args.field)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    x, y, z, ux, uy, uz, solid = _load_field(field)
    outputs = [
        render_xy(x, y, z, ux, uy, uz, solid, args.z, out / "xy_slice_z35.png"),
        render_xz(x, y, z, ux, uy, uz, solid, args.y, out / "xz_slice_y0.png"),
        render_yz(x, y, z, ux, uy, uz, solid, args.x, out / "yz_slice_x0.png"),
    ]
    summary = {"source_field": str(field), "outputs": outputs}
    summary_path = out / "slice_render_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("slice visualization complete")
    for item in outputs:
        print(f"{item['plane']}: {Path(item['path']).resolve()}")
    print(f"summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
