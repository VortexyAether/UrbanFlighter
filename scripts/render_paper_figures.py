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
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    boxes = [
        (0.25, 3.35, 2.4, 1.45, "#e8f1fb", "1  Geometry + inlet\nOSM footprints\nOpen-Meteo $U_\\infty$"),
        (2.85, 3.35, 2.45, 1.45, "#fff4d6", "2  Hidden CFD-lite\npotential flow + wall\n+ empirical wake"),
        (5.50, 3.35, 2.45, 1.45, "#e7f6ec", "3  Dynamics\nquadratic $v_\\mathrm{air}$ drag\n120 Hz / Gym $dt$"),
        (8.15, 3.35, 2.40, 1.45, "#fde8f0", "4  Cockpit\n2D / 3D Lite\nLiDAR maps"),
        (0.25, 0.35, 5.05, 2.25, "#efe9fb", "UrbanFlow Gym actor (49-D)\nOSM geometry LiDAR + sim. radar\nknown inlet only\nNO grid  ·  NOT TRAINED"),
        (5.55, 0.35, 5.00, 2.25, "#fde8e4", "Hidden / display-only\nlocal wind → dynamics + reward\nscenery ≠ collision\nTrue 3D overlay ≠ flyable field"),
    ]
    for x, y, w, h, color, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.03,rounding_size=0.10",
                facecolor=color,
                edgecolor="#1f2937",
                linewidth=1.15,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.0)
    for x0, x1 in ((2.65, 2.85), (5.30, 5.50), (7.95, 8.15)):
        ax.annotate(
            "",
            xy=(x1, 4.08),
            xytext=(x0, 4.08),
            arrowprops=dict(arrowstyle="->", color="#111827", lw=1.25),
        )
    ax.annotate(
        "",
        xy=(2.75, 2.60),
        xytext=(4.05, 3.35),
        arrowprops=dict(arrowstyle="->", color="#111827", lw=1.15),
    )
    ax.annotate(
        "",
        xy=(8.05, 2.60),
        xytext=(6.70, 3.35),
        arrowprops=dict(arrowstyle="->", color="#111827", lw=1.15),
    )
    ax.set_title("Urban Flighter splits hidden flow from the actor channel")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_observation(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 3.6))
    ax.set_xlim(0, 8.8)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    actor = [
        (0.2, 2.15, 1.3, "#dbeafe", "odometry"),
        (1.6, 2.15, 1.5, "#dbeafe", "inlet-only\n$v_g-U_\\infty$"),
        (3.2, 2.15, 1.4, "#dbeafe", "goal"),
        (4.7, 2.15, 1.5, "#dbeafe", "16-ray\nLiDAR"),
        (6.3, 2.15, 2.25, "#dbeafe", "8-beam radar\nrange + $\\dot r$"),
    ]
    hidden = [
        (0.2, 0.35, 2.6, "#fee2e2", "CFD-lite grid"),
        (3.0, 0.35, 2.6, "#fee2e2", "exact local $w$"),
        (5.8, 0.35, 2.75, "#fee2e2", "privileged critic"),
    ]
    ax.text(0.2, 3.28, "Actor observation", fontsize=10, fontweight="bold")
    ax.text(0.2, 1.48, "Hidden from reset/step", fontsize=10, fontweight="bold")
    for x, y, w, color, text in actor:
        ax.add_patch(FancyBboxPatch((x, y), w, 0.95, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    facecolor=color, edgecolor="#111827", linewidth=1.0))
        ax.text(x + w / 2, y + 0.48, text, ha="center", va="center", fontsize=8.2)
    for x, y, w, color, text in hidden:
        ax.add_patch(FancyBboxPatch((x, y), w, 0.95, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    facecolor=color, edgecolor="#111827", linewidth=1.0))
        ax.text(x + w / 2, y + 0.48, text, ha="center", va="center", fontsize=8.2)
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
    fig_observation(OUT / "fig_observation.png")
    power = fig_power_curve(OUT / "fig_power_curve.png")
    wind = fig_wind_response(OUT / "fig_wind_response.png")
    summary = {"power": power, "wind": wind, "figures": sorted(p.name for p in OUT.glob("fig_*.png"))}
    (OUT / "paper_figure_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
