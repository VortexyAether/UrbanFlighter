#!/usr/bin/env python3
"""Render Urban Flighter 'what it does' overview images."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "overview_explain"
_KO_FONT = Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf")
if _KO_FONT.exists():
    font_manager.fontManager.addfont(str(_KO_FONT))
    _KO_PROP = font_manager.FontProperties(fname=str(_KO_FONT))
else:
    _KO_PROP = font_manager.FontProperties(family="DejaVu Sans")
plt.rcParams["axes.unicode_minus"] = False


def _txt(ax, x, y, s, size=11, color="#e8eef5", weight="regular", ha="left", va="center"):
    return ax.text(
        x,
        y,
        s,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        fontfamily="DejaVu Sans",
        zorder=10,
    )


def _card(ax, x, y, w, h, ec="#2a3440", fc="#0d1218", lw=1.2, z=2):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=z,
    )
    ax.add_patch(p)
    return p


def _arrow(ax, x1, y1, x2, y2, color="#6b7c8f"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.4,
            color=color,
            zorder=5,
        )
    )


def render_what_it_does(path: Path) -> None:
    fig = plt.figure(figsize=(16, 10), dpi=160, facecolor="#07090c")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 100)
    ax.set_axis_off()
    ax.set_facecolor("#07090c")

    for x in range(0, 161, 8):
        ax.plot([x, x], [0, 100], color="#12171e", lw=0.6, zorder=0)
    for y in range(0, 101, 8):
        ax.plot([0, 160], [y, y], color="#12171e", lw=0.6, zorder=0)

    _txt(ax, 8, 94, "URBAN FLIGHTER", size=26, weight="bold", color="#f4f7fb")
    _txt(
        ax,
        8,
        89.5,
        "What it does — research/demo urban-drone simulator",
        size=13,
        color="#9aa8b7",
    )
    _txt(ax, 152, 94, "CFD-lite · not full CFD", size=10, color="#7f8b98", ha="right")
    _txt(
        ax,
        152,
        91,
        "SIM odometry · no loop closure",
        size=10,
        color="#7f8b98",
        ha="right",
    )

    _card(ax, 8, 76, 144, 10, ec="#3d4a5a", fc="#0b1016")
    _txt(ax, 12, 82.5, "CORE LOOP", size=9, color="#7ad0ff", weight="bold")
    _txt(
        ax,
        12,
        78.8,
        "Real city geometry (OSM)  +  live/forecast wind inlet (Open-Meteo)  →  "
        "CFD-lite wind field  →  fly a drone  →  feel wind/energy + local LiDAR maps  →  "
        "optional Gym baseline replay",
        size=11,
        color="#d7e0ea",
    )

    _card(ax, 8, 42, 46, 30, ec="#2f3b48")
    _txt(ax, 12, 68, "INPUTS", size=12, weight="bold", color="#7ad0ff")
    items_in = [
        ("OSM buildings", "real footprints + heights"),
        ("Open-Meteo wind", "current 10 m inlet"),
        ("Pilot controls", "2D WASD / Arcade·Pilot 3D"),
        ("Location pick", "lat/lon + radius domain"),
    ]
    for i, (a, b) in enumerate(items_in):
        y = 61.5 - i * 5.2
        ax.add_patch(Circle((14.2, y), 0.7, color="#7ad0ff", zorder=6))
        _txt(ax, 17, y + 0.7, a, size=11, weight="bold", color="#eef3f8")
        _txt(ax, 17, y - 1.2, b, size=9.5, color="#93a1b0")

    _card(ax, 58, 42, 46, 30, ec="#2f3b48")
    _txt(ax, 62, 68, "FLIGHT MODES", size=12, weight="bold", color="#9dffa8")
    modes = [
        ("2D", "top-down canvas, 120 Hz, 180-ray scan"),
        ("3D Lite", "Three.js city + CFD-lite sampling"),
        ("True 3D Wind", "U/V/W streamline overlay (visual)"),
        ("Flight dyn.", "B = PF + wall damp + wake  ·  not NS"),
    ]
    for i, (a, b) in enumerate(modes):
        y = 61.5 - i * 5.2
        ax.add_patch(
            FancyBboxPatch(
                (62, y - 1.6),
                38,
                3.6,
                boxstyle="round,pad=0.2,rounding_size=0.6",
                facecolor="#121a22",
                edgecolor="#334250",
                lw=1,
                zorder=5,
            )
        )
        _txt(ax, 64, y + 0.5, a, size=10.5, weight="bold", color="#b8ffc4")
        _txt(ax, 64, y - 1.0, b, size=8.8, color="#9aabbb")

    _card(ax, 108, 42, 44, 30, ec="#2f3b48")
    _txt(ax, 112, 68, "OUTPUTS / VIEWS", size=12, weight="bold", color="#ffd27a")
    outs = [
        ("Wind-affected flight", "energy + disturbance"),
        ("JET LiDAR returns", "2D rays / 3D Fibonacci shell"),
        ("Rolling sensor maps", "history + sim odometry"),
        ("Gym Inspector", "baseline replay on live OSM world"),
    ]
    for i, (a, b) in enumerate(outs):
        y = 61.5 - i * 5.2
        ax.add_patch(Circle((112.2, y), 0.7, color="#ffd27a", zorder=6))
        _txt(ax, 115, y + 0.7, a, size=11, weight="bold", color="#eef3f8")
        _txt(ax, 115, y - 1.2, b, size=9.5, color="#93a1b0")

    _arrow(ax, 54, 57, 58, 57, "#5f7388")
    _arrow(ax, 104, 57, 108, 57, "#5f7388")

    _card(ax, 8, 8, 144, 30, ec="#2f3b48", fc="#0a0f14")
    _txt(ax, 12, 34, "STACK", size=12, weight="bold", color="#c9d4e0")

    stack = [
        (
            12,
            "BACKEND",
            "#7ad0ff",
            [
                "FastAPI :8000",
                "OSMnx geometry",
                "flow_2d B solver",
                "AeroJAX snapshots",
                "UrbanFlow Gym API",
            ],
        ),
        (
            60,
            "FRONTEND",
            "#9dffa8",
            [
                "React + Vite :5173",
                "2D canvas flight",
                "Three.js 3D city",
                "movable cockpit windows",
                "sensor + energy HUD",
            ],
        ),
        (
            108,
            "HONESTY BOUNDARY",
            "#ff9d9d",
            [
                "CFD-lite ≠ Navier–Stokes",
                "True3D = overlay, not fly dyn.",
                "LiDAR = raycast prototype",
                "maps = SIM odometry",
                "Gym = NOT TRAINED",
            ],
        ),
    ]
    for x0, title, col, lines in stack:
        ax.add_patch(
            FancyBboxPatch(
                (x0, 11),
                40,
                20.5,
                boxstyle="round,pad=0.3,rounding_size=0.8",
                facecolor="#101820",
                edgecolor=col,
                lw=1.3,
                zorder=4,
            )
        )
        _txt(ax, x0 + 2, 28.2, title, size=11, weight="bold", color=col)
        for i, line in enumerate(lines):
            _txt(ax, x0 + 2.5, 24.5 - i * 2.8, "·  " + line, size=10, color="#d0dae5")

    _txt(
        ax,
        8,
        4.5,
        "Urban Flighter  ·  interactive urban drone + wind research sandbox",
        size=10,
        color="#6d7a88",
    )
    _txt(ax, 152, 4.5, "TARS render", size=9, color="#4f5b68", ha="right")

    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def render_scene(path: Path) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=160, facecolor="#05070a")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 90)
    ax.axis("off")
    ax.set_facecolor("#05070a")

    ax.add_patch(
        FancyBboxPatch(
            (6, 8),
            92,
            74,
            boxstyle="round,pad=0.5,rounding_size=1.4",
            facecolor="#0b1016",
            edgecolor="#2a3644",
            lw=1.3,
        )
    )
    ax.text(10, 76, "SCENE", color="#7ad0ff", fontsize=11, fontweight="bold")
    ax.text(
        10,
        72,
        "Real OSM city + drone + CFD-lite wind",
        color="#c5d0db",
        fontsize=13,
        fontweight="bold",
    )

    ax.add_patch(Rectangle((12, 16), 80, 48, facecolor="#101820", edgecolor="#243040", lw=1))
    buildings = [
        (18, 22, 10, 28),
        (32, 20, 12, 22),
        (48, 24, 9, 34),
        (62, 21, 14, 18),
        (20, 40, 8, 16),
        (36, 42, 11, 14),
        (54, 44, 10, 12),
        (70, 38, 12, 20),
        (78, 22, 8, 24),
    ]
    for x, y, w, h in buildings:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.05,rounding_size=0.2",
                facecolor="#1a2430",
                edgecolor="#4a5d70",
                lw=1.0,
                zorder=3,
            )
        )
        for yy in np.linspace(y + 2, y + h - 2, 4):
            ax.plot([x + 1.2, x + w - 1.2], [yy, yy], color="#2d3c4c", lw=0.6, zorder=4)

    xs = np.linspace(14, 88, 200)
    for k, y0 in enumerate([24, 30, 36, 42, 48, 54]):
        ys = y0 + 2.2 * np.sin((xs - 14) / 10.0 + k * 0.4)
        for bx, _by, bw, _bh in buildings:
            cx = bx + bw / 2
            mask = (xs > bx - 2) & (xs < bx + bw + 2)
            ys = ys.copy()
            ys[mask] += 1.8 * np.exp(-(((xs[mask] - cx) / 6) ** 2)) * (1 if k % 2 == 0 else -1)
        color = plt.cm.cool(0.25 + 0.1 * k)
        ax.plot(xs, ys, color=color, lw=1.3, alpha=0.85, zorder=2)
        ax.annotate(
            "",
            xy=(xs[-1], ys[-1]),
            xytext=(xs[-8], ys[-8]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0),
            zorder=2,
        )

    dx, dy = 44, 52
    ax.plot([dx - 2.2, dx + 2.2], [dy, dy], color="#e8f0ff", lw=2.2, zorder=8)
    ax.plot([dx, dx], [dy - 2.2, dy + 2.2], color="#e8f0ff", lw=2.2, zorder=8)
    ax.add_patch(Circle((dx, dy), 0.7, color="#9dffa8", zorder=9))
    for ang in np.linspace(0, 2 * np.pi, 18, endpoint=False):
        r = 11
        ax.plot(
            [dx, dx + r * np.cos(ang)],
            [dy, dy + r * np.sin(ang)],
            color="#ffb84d",
            lw=0.7,
            alpha=0.55,
            zorder=7,
        )
    ax.add_patch(
        Circle(
            (dx, dy),
            11,
            fill=False,
            edgecolor="#ffb84d",
            lw=0.8,
            alpha=0.35,
            linestyle="--",
            zorder=7,
        )
    )

    ax.text(14, 18.5, "OSM buildings", color="#8fa0b2", fontsize=9)
    ax.text(72, 58, "CFD-lite streamlines\n(B: PF+damp+wake)", color="#8ecbff", fontsize=9)
    ax.text(46, 63.5, "drone + JET LiDAR shell", color="#ffd27a", fontsize=9, ha="center")

    def rcard(y, title, body, accent):
        ax.add_patch(
            FancyBboxPatch(
                (104, y),
                50,
                16.5,
                boxstyle="round,pad=0.35,rounding_size=0.9",
                facecolor="#0c1219",
                edgecolor=accent,
                lw=1.3,
            )
        )
        ax.text(108, y + 13.2, title, color=accent, fontsize=11, fontweight="bold")
        ax.text(108, y + 6.5, body, color="#c9d4e0", fontsize=10, va="center", linespacing=1.35)

    rcard(
        65.5,
        "1. BUILD THE WORLD",
        "Pick a city patch.\nPull real buildings + wind inlet.\nMake a local metre-frame domain.",
        "#7ad0ff",
    )
    rcard(
        45.5,
        "2. COMPUTE WIND (lite)",
        "Potential-flow base field,\nwall damping + empirical wake.\nFast enough for interactive flight.",
        "#9dffa8",
    )
    rcard(
        25.5,
        "3. FLY & SENSE",
        "Pilot through streets/roofs.\nWind hits energy & motion.\nLocal LiDAR maps roll with history.",
        "#ffd27a",
    )
    rcard(
        5.5,
        "4. RESEARCH HOOK",
        "Same live OSM world feeds\nGym Inspector baselines.\nTraining later — currently NOT TRAINED.",
        "#ff9d9d",
    )

    ax.text(
        6,
        3.2,
        "Concept diagram  ·  not a simulator screenshot  ·  honest CFD-lite framing",
        color="#5d6a78",
        fontsize=9,
    )
    ax.text(154, 3.2, "Urban Flighter", color="#5d6a78", fontsize=9, ha="right")

    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def render_one_liner_ko(path: Path) -> None:
    fig = plt.figure(figsize=(14, 7.8), dpi=160, facecolor="#06080c")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.set_xlim(0, 140)
    ax.set_ylim(0, 78)
    ax.set_facecolor("#06080c")

    ax.text(
        8,
        70,
        "URBAN FLIGHTER",
        fontsize=28,
        color="#f3f6fa",
        fontweight="bold",
        fontproperties=_KO_PROP,
    )
    ax.text(
        8,
        61.5,
        "도시 건물 사이로 드론을 날리고,\n실제 바람장(근사)이 비행·에너지·센서맵에 영향을 주는\n연구/데모용 시뮬레이터.",
        fontsize=15.5,
        color="#d5dee8",
        linespacing=1.5,
        fontproperties=_KO_PROP,
        va="top",
    )

    pillars = [
        (8, "#7ad0ff", "REAL CITY", "OSM 건물 형상\nOpen-Meteo 바람"),
        (50, "#9dffa8", "CFD-LITE WIND", "PF + damping + wake\nfull NS 아님"),
        (92, "#ffd27a", "FLY + SENSE + GYM", "2D / 3D Lite / True3D\nLiDAR maps · baseline RL hook"),
    ]
    for x, c, t, b in pillars:
        ax.add_patch(
            FancyBboxPatch(
                (x, 16),
                36,
                28,
                boxstyle="round,pad=0.4,rounding_size=1.0",
                facecolor="#0d131a",
                edgecolor=c,
                lw=1.6,
            )
        )
        ax.text(
            x + 18,
            36,
            t,
            color=c,
            fontsize=13,
            fontweight="bold",
            ha="center",
            fontproperties=_KO_PROP,
        )
        ax.text(
            x + 18,
            26,
            b,
            color="#c7d2de",
            fontsize=12,
            ha="center",
            va="center",
            linespacing=1.4,
            fontproperties=_KO_PROP,
        )

    ax.text(
        8,
        9.5,
        '한 줄:  “실제 도시 기하 + 정직한 바람 근사 위에서 드론을 조종·관측하는 샌드박스”',
        color="#9aa8b7",
        fontsize=12,
        fontproperties=_KO_PROP,
    )
    ax.text(
        8,
        4.0,
        "Status: research/demo  ·  Gym NOT TRAINED  ·  True3D wind is visual overlay for flyable dyn.",
        color="#667483",
        fontsize=10,
        fontproperties=_KO_PROP,
    )

    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        OUT_DIR / "urban_flighter_what_it_does.png",
        OUT_DIR / "urban_flighter_how_it_works_scene.png",
        OUT_DIR / "urban_flighter_one_liner_ko.png",
    ]
    render_what_it_does(paths[0])
    render_scene(paths[1])
    render_one_liner_ko(paths[2])
    for p in paths:
        print(f"wrote {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
