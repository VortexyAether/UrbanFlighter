"""Quadratic air-relative drag essence demo.

Produces a headwind vs tailwind vs hover-in-wind comparison with honest labels.
Not blade-element theory and not Navier-Stokes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from urban_flighter_physics.quadratic_air_drag import (
    QUAD_AIR_DRAG,
    evaluate_physical_drag_power,
    integrate_quadratic_air_drag,
    parasite_drag_per_m,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "drag_essence"
DT = 1.0 / 120.0
THRUST = 8.0  # m/s^2 along +x


def integrate_case(wind_x: float, seconds: float = 8.0, thrust_x: float = THRUST) -> dict:
    steps = int(round(seconds / DT))
    ground = np.zeros(2, dtype=float)
    wind = np.array([wind_x, 0.0], dtype=float)
    times = []
    speeds = []
    airspeeds = []
    parasite = []
    induced = []
    energy = 0.0
    for step in range(steps + 1):
        power = evaluate_physical_drag_power(ground, wind)
        times.append(step * DT)
        speeds.append(float(np.linalg.norm(ground)))
        airspeeds.append(power["relative_air_speed"])
        parasite.append(power["parasite_power_w"])
        induced.append(power["induced_power_w"])
        if step == steps:
            break
        ground = ground + np.array([thrust_x, 0.0]) * DT
        ground = integrate_quadratic_air_drag(ground, wind, DT)
        energy += power["total_power_w"] * DT
    return {
        "wind_x": wind_x,
        "final_ground_speed": speeds[-1],
        "final_air_speed": airspeeds[-1],
        "energy_j": energy,
        "times": times,
        "speeds": speeds,
        "airspeeds": airspeeds,
        "parasite_w": parasite,
        "induced_w": induced,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hover = integrate_case(8.0, seconds=6.0, thrust_x=0.0)
    tail = integrate_case(6.0)
    still = integrate_case(0.0)
    head = integrate_case(-6.0)

    metrics = {
        "model_id": QUAD_AIR_DRAG["model_id"],
        "honesty": QUAD_AIR_DRAG["honesty"],
        "k_per_m": parasite_drag_per_m(),
        "hover_in_8mps_wind_final_ground_speed": hover["final_ground_speed"],
        "tailwind_6_final_ground_speed": tail["final_ground_speed"],
        "still_final_ground_speed": still["final_ground_speed"],
        "headwind_6_final_ground_speed": head["final_ground_speed"],
        "tailwind_6_energy_j": tail["energy_j"],
        "still_energy_j": still["energy_j"],
        "headwind_6_energy_j": head["energy_j"],
        "headwind_slower_than_tailwind": head["final_ground_speed"] < tail["final_ground_speed"],
        "headwind_costs_more_than_tailwind": head["energy_j"] > tail["energy_j"],
        "hover_approaches_wind": abs(hover["final_ground_speed"] - 8.0) < 1.5,
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=140)
    fig.suptitle("Urban Flighter · quadratic air-relative drag  |  not blade-element / not NS", fontsize=11)
    for case, color, label in (
        (tail, "#2aa876", "tailwind +6 m/s"),
        (still, "#4c6ef5", "still air"),
        (head, "#e03131", "headwind -6 m/s"),
    ):
        axes[0].plot(case["times"], case["speeds"], color=color, lw=2.0, label=label)
        axes[1].plot(case["times"], case["parasite_w"], color=color, lw=2.0, label=f"{label} parasite")
        axes[1].plot(case["times"], case["induced_w"], color=color, lw=1.2, ls="--", label=f"{label} induced")
    axes[0].set_title("Ground speed under same thrust")
    axes[0].set_xlabel("t [s]")
    axes[0].set_ylabel("ground speed [m/s]")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].set_title("Power split: parasite rises, induced falls")
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("power [W]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    png_path = OUT_DIR / "urban_flighter_drag_essence.png"
    json_path = OUT_DIR / "urban_flighter_drag_essence.json"
    fig.savefig(png_path)
    plt.close(fig)

    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"png": str(png_path), "json": str(json_path), **metrics}, indent=2))


if __name__ == "__main__":
    main()
