"""Paper figures for Urban Flighter. Numbers come from the live physics module."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from urban_flighter_physics.quadratic_air_drag import (
    QUAD_AIR_DRAG,
    evaluate_physical_drag_power,
    integrate_quadratic_air_drag,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
SHOW = ROOT / "docs" / "showcase"
DT = 1.0 / 120.0


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.28,
        }
    )


def fig_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    ax.set_xlim(0, 10.4)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    boxes = [
        (0.25, 2.7, 2.3, 1.5, "#dbeafe", "Data ingest\nOSM buildings\nOpen-Meteo inlet"),
        (2.85, 2.7, 2.5, 1.5, "#fef3c7", "Hidden flow\nCFD-lite B grid\nnot Navier-Stokes"),
        (5.65, 2.7, 2.2, 1.5, "#dcfce7", "Flyable dynamics\nquadratic v_air drag\nfixed 120 Hz"),
        (8.15, 2.7, 2.0, 1.5, "#fce7f3", "Cockpit\n2D / 3D Lite\nLiDAR maps"),
        (1.4, 0.35, 3.2, 1.7, "#ede9fe", "UrbanFlow Gym\nactor: OSM + inlet + LiDAR\nNO full-flow access\nNOT TRAINED"),
        (5.3, 0.35, 3.4, 1.7, "#fee2e2", "Honesty boundary\nscenery != physics\nTrue3D overlay != flyable\nenergy from v_air"),
    ]
    for x, y, w, h, color, text in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    facecolor=color, edgecolor="#111827", linewidth=1.1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.4)
    for x0, x1 in ((2.55, 2.85), (5.35, 5.65), (7.85, 8.15)):
        ax.annotate("", xy=(x1, 3.45), xytext=(x0, 3.45),
                    arrowprops=dict(arrowstyle="->", color="#111827", lw=1.2))
    ax.annotate("", xy=(3.0, 2.05), xytext=(3.0, 2.7),
                arrowprops=dict(arrowstyle="->", color="#111827", lw=1.2))
    ax.annotate("", xy=(7.0, 2.05), xytext=(7.0, 2.7),
                arrowprops=dict(arrowstyle="->", color="#111827", lw=1.2))
    ax.set_title("Urban Flighter: hidden flow and drag drive motion; the actor never sees the grid")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_power_curve(path: Path) -> dict:
    speeds = np.linspace(0.0, 18.0, 181)
    parasite, induced, total = [], [], []
    for v in speeds:
        p = evaluate_physical_drag_power(np.array([v, 0.0, 0.0]), np.zeros(3))
        parasite.append(p["parasite_power_w"])
        induced.append(p["induced_power_w"])
        total.append(p["total_power_w"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(speeds, parasite, color="#e03131", lw=2.0, label="parasite  0.5 rho Cd A |v_air|^3")
    ax.plot(speeds, induced, color="#4c6ef5", lw=2.0, label="induced  momentum theory")
    ax.plot(speeds, total, color="#111827", lw=2.2, label="total  + avionics/sensors")
    ax.set_xlabel("relative airspeed |v_air|  [m/s]")
    ax.set_ylabel("power  [W]")
    ax.set_title("Still-air power split  |  not blade-element / not NS")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    i_min = int(np.argmin(total))
    return {
        "hover_total_w": float(total[0]),
        "min_total_w": float(total[i_min]),
        "min_speed_mps": float(speeds[i_min]),
        "cruise11_total_w": float(total[np.argmin(np.abs(speeds - 11))]),
    }


def fig_wind_response(path: Path) -> dict:
    def integrate(wind_x: float, thrust: float = 8.0, seconds: float = 8.0):
        ground = np.zeros(2)
        wind = np.array([wind_x, 0.0])
        t, vg, va, ppara, pind = [], [], [], [], []
        steps = int(round(seconds / DT))
        energy = 0.0
        for step in range(steps + 1):
            power = evaluate_physical_drag_power(ground, wind)
            t.append(step * DT)
            vg.append(float(np.linalg.norm(ground)))
            va.append(power["relative_air_speed"])
            ppara.append(power["parasite_power_w"])
            pind.append(power["induced_power_w"])
            if step == steps:
                break
            ground = ground + np.array([thrust, 0.0]) * DT
            ground = integrate_quadratic_air_drag(ground, wind, DT)
            energy += power["total_power_w"] * DT
        return {
            "t": t, "vg": vg, "va": va, "energy": energy,
            "final_vg": vg[-1], "final_va": va[-1],
        }

    tail = integrate(6.0)
    still = integrate(0.0)
    head = integrate(-6.0)
    hover = integrate(8.0, thrust=0.0, seconds=6.0)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
    for case, color, label in (
        (tail, "#2f9e44", "tailwind +6 m/s"),
        (still, "#4c6ef5", "still air"),
        (head, "#e03131", "headwind -6 m/s"),
    ):
        axes[0].plot(case["t"], case["vg"], color=color, lw=2.0, label=label)
    axes[0].set_xlabel("t  [s]")
    axes[0].set_ylabel("ground speed  [m/s]")
    axes[0].set_title("Same thrust, different local wind")
    axes[0].legend(fontsize=8)

    axes[1].plot(hover["t"], hover["vg"], color="#7048e8", lw=2.0, label="zero thrust, wind = 8 m/s")
    axes[1].axhline(8.0, color="#868e96", ls="--", lw=1.0, label="local wind")
    axes[1].set_xlabel("t  [s]")
    axes[1].set_ylabel("ground speed  [m/s]")
    axes[1].set_title("Stick-off equilibrium is the wind")
    axes[1].legend(fontsize=8)
    fig.suptitle("Quadratic air-relative drag in the flyable loop", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return {
        "tail_final_mps": tail["final_vg"],
        "still_final_mps": still["final_vg"],
        "head_final_mps": head["final_vg"],
        "tail_energy_j": tail["energy"],
        "still_energy_j": still["energy"],
        "head_energy_j": head["energy"],
        "hover_final_mps": hover["final_vg"],
        "k_per_m": 0.5 * 1.225 * 1.05 * 0.18 / 2.5,
        "model_id": QUAD_AIR_DRAG["model_id"],
    }


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    copies = {
        SHOW / "potential_flow_toy" / "potential_flow_streamlines.png": OUT / "fig_potential_flow.png",
        SHOW / "gym_fixture_trajectories.png": OUT / "fig_trajectories.png",
        SHOW / "gym_fixture_metrics.png": OUT / "fig_metrics.png",
    }
    for src, dst in copies.items():
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)

    fig_architecture(OUT / "fig_architecture.png")
    power = fig_power_curve(OUT / "fig_power_curve.png")
    wind = fig_wind_response(OUT / "fig_wind_response.png")
    summary = {"power": power, "wind": wind, "figures": sorted(p.name for p in OUT.glob("fig_*.png"))}
    (OUT / "paper_figure_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
