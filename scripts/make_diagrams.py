"""Explanatory diagrams of the method for the README (docs/img/diagram_*.png).

Schematic only: no data is read. The error-budget bars use the shares stated in the README (squared
error of the physics chain: 0.128 -> 0.108 with oracle weather -> 0.055 with measured POA).

Run: python scripts/make_diagrams.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e6e5e1"
OUT = Path("docs/img")


def canvas(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, width * 10)
    ax.set_ylim(0, height * 10)
    ax.axis("off")
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def box(ax, x, y, w, h, text, color=BLUE, fc=None, size=9, bold=False, tc=INK):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        fc=fc or "white",
        ec=color,
        lw=1.8,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=tc,
        fontweight="bold" if bold else "normal",
        linespacing=1.35,
    )


def arrow(ax, x0, y0, x1, y1, color=MUTED):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=color, lw=1.6
        )
    )


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def pipeline() -> None:
    fig, ax = canvas(11, 3.6)
    ax.text(
        0, 34, "From raw data to an honest comparison", fontsize=12, fontweight="bold", color=INK
    )
    cols = [
        (
            1,
            "RAW DATA",
            ["Plant power and\nirradiance\n(15 min)", "GFS weather\nforecasts\n(archived)"],
            MUTED,
        ),
        (28, "PREPARE", ["Hourly series, UTC", "Quality flags", "Leak-proof\nfeatures"], BLUE),
        (
            55,
            "FORECAST",
            ["Baselines", "Physics (pvlib)", "ML  /  hybrid", "Quantiles, deep learning"],
            ORANGE,
        ),
        (
            82,
            "EVALUATE",
            ["Walk-forward\nsplits", "Metrics +\nbootstrap intervals", "Hold-out year"],
            AQUA,
        ),
    ]
    for x, title, items, color in cols:
        ax.text(x + 11, 29, title, ha="center", fontsize=9, fontweight="bold", color=color)
        n = len(items)
        h = 22 / n - 1.5
        for i, text in enumerate(items):
            y = 26 - (i + 1) * (22 / n) + 1
            box(ax, x, y, 22, h, text, color=color, size=8)
    for x in (24, 51, 78):
        arrow(ax, x, 16, x + 4, 16)
    save(fig, "diagram_pipeline.png")


def hybrid() -> None:
    fig, ax = canvas(11, 3.6)
    ax.text(
        0,
        34,
        "The hybrid model: physics does the heavy lifting, ML fixes what is left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    box(ax, 1, 20, 17, 9, "GFS weather\nforecast", MUTED)
    box(ax, 26, 20, 20, 9, "Physics model\n(pvlib chain)", BLUE, bold=True)
    box(ax, 54, 20, 20, 9, "Physics forecast\n(kW)", BLUE)
    ax.text(79, 24.5, "+", fontsize=20, ha="center", va="center", color=INK)
    box(ax, 84, 20, 20, 9, "Learned\ncorrection", ORANGE)
    box(ax, 26, 5, 20, 9, "LightGBM learns the\nresidual (error)", ORANGE, bold=True)
    box(ax, 54, 5, 50, 9, "= hybrid forecast, clipped to [0, inverter limit]", AQUA, bold=True)
    arrow(ax, 19.5, 24.5, 25, 24.5)
    arrow(ax, 47.5, 24.5, 53, 24.5)
    arrow(ax, 36, 19, 36, 15.5)
    arrow(ax, 47.5, 9.5, 53, 9.5, ORANGE)
    arrow(ax, 94, 19, 94, 15.5, ORANGE)
    ax.text(
        1,
        9.5,
        "Physics gives the right shape\nand scale; the trees learn the\nhabits of the weather forecast.",
        fontsize=8.5,
        color=MUTED,
        va="center",
    )
    save(fig, "diagram_hybrid.png")


def walkforward() -> None:
    fig, ax = canvas(11, 3.9)
    ax.text(
        0,
        37,
        "Walk-forward testing: train on the past, test on the next month, move forward",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    months = 23
    unit = 100 / months
    rows = [(4 + k, 4.35 + k, 5.35 + k) for k in range(7)]  # train end, test start, test end
    for i, (te, ts, tend) in enumerate(rows):
        y = 29 - i * 3.5
        ax.add_patch(plt.Rectangle((0, y), te * unit, 2.2, fc=BLUE, alpha=0.85, lw=0))
        ax.add_patch(plt.Rectangle((te * unit, y), 0.35 * unit, 2.2, fc=YELLOW, lw=0))  # embargo
        ax.add_patch(plt.Rectangle((ts * unit, y), 1 * unit, 2.2, fc=ORANGE, lw=0))
    ax.add_patch(plt.Rectangle((11 * unit, 1.5), 12 * unit, 2.4, fc=AQUA, lw=0))
    ax.text(
        11 * unit + 6 * unit,
        2.7,
        "2025 hold-out (not used for decisions)",
        ha="center",
        va="center",
        fontsize=8.5,
        color="white",
        fontweight="bold",
    )
    ax.text(0, 5.2, "Feb 2024", fontsize=8, color=MUTED)
    ax.text(11 * unit, 5.2, "Jan 2025", fontsize=8, color=MUTED)
    ax.text(100, 5.2, "Dec 2025", fontsize=8, color=MUTED, ha="right")
    for label, color, x in (
        ("training window (grows each month)", BLUE, 0),
        ("2-day embargo", YELLOW, 38),
        ("test month", ORANGE, 58),
    ):
        ax.add_patch(
            plt.Rectangle((x, 34.0 - 0.0), 1.6, 1.4, fc=color, lw=0, transform=ax.transData)
        )
        ax.text(x + 2.4, 34.7, label, fontsize=8.5, va="center", color=INK)
    save(fig, "diagram_walkforward.png")


def timeline() -> None:
    fig, ax = canvas(11, 3.5)
    ax.text(
        0,
        33,
        "When a forecast is issued and what it may use",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    # day-ahead
    ax.text(0, 26, "Day-ahead", fontsize=10, fontweight="bold", color=BLUE)
    ax.add_patch(plt.Rectangle((0, 20.5), 100, 0.4, fc=GRID, lw=0))
    ax.add_patch(plt.Rectangle((30, 19.5), 2.2, 2.4, fc=BLUE, lw=0))
    ax.text(
        31, 24.2, "10:00 local, day D-1:\nforecast issued", fontsize=8.5, ha="center", color=BLUE
    )
    ax.add_patch(plt.Rectangle((50, 19.0), 34, 3.4, fc=ORANGE, alpha=0.85, lw=0))
    ax.text(
        67,
        20.7,
        "all 24 hours of day D",
        fontsize=8.5,
        ha="center",
        va="center",
        color="white",
        fontweight="bold",
    )
    ax.text(
        0,
        17.2,
        "Allowed: GFS forecast published ~2 days before the hour, sun position, plant history up to 10:00 D-1.",
        fontsize=8.2,
        color=MUTED,
    )
    # intraday
    ax.text(0, 11.5, "Intraday", fontsize=10, fontweight="bold", color=BLUE)
    ax.add_patch(plt.Rectangle((0, 6.5), 100, 0.4, fc=GRID, lw=0))
    ax.add_patch(plt.Rectangle((20, 5.5), 2.2, 2.4, fc=BLUE, lw=0))
    ax.text(21, 10.2, "end of hour t:\nforecast issued", fontsize=8.5, ha="center", color=BLUE)
    for k in range(1, 7):
        ax.add_patch(
            plt.Rectangle(
                (22 + k * 11 - 9, 5.2), 9, 3.0, fc=ORANGE, alpha=0.5 + 0.08 * (6 - k), lw=0
            )
        )
        ax.text(
            22 + k * 11 - 4.5,
            6.7,
            f"t+{k}",
            fontsize=8,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
        )
    ax.text(
        0,
        2.0,
        "Allowed: the newest GFS run (published before issue), plant measurements up to t-1 (known 1 h late).",
        fontsize=8.2,
        color=MUTED,
    )
    save(fig, "diagram_timeline.png")


def budget() -> None:
    fig, ax = canvas(11, 2.8)
    ax.text(
        0,
        25,
        "Where the day-ahead error comes from (share of squared error, physics chain)",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    parts = [
        (30, "GFS forecast\nerror ~30%", ORANGE),
        (50, "reanalysis irradiance vs\nplant sensor ~50%", YELLOW),
        (20, "plant model\n~20%", BLUE),
    ]
    x = 0
    for share, label, color in parts:
        ax.add_patch(plt.Rectangle((x, 9), share, 8, fc=color, lw=0))
        ax.text(
            x + share / 2,
            13,
            label,
            ha="center",
            va="center",
            fontsize=8.8,
            color="white" if color != YELLOW else INK,
            fontweight="bold",
        )
        x += share
    ax.text(
        0,
        5,
        "nRMSE:  0.128 (GFS forecast)  ->  0.108 (observed weather)  ->  0.055 (plant's own irradiance sensor)",
        fontsize=8.5,
        color=MUTED,
    )
    ax.text(
        0,
        1.5,
        "About 80% of the error is in the irradiance input, so better weather input is the main lever.",
        fontsize=8.5,
        color=INK,
    )
    save(fig, "diagram_error_budget.png")


if __name__ == "__main__":
    for make in (pipeline, hybrid, walkforward, timeline, budget):
        make()
