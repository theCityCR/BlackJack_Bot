#!/usr/bin/env python3
"""Plot learning-curve CSV files as a simple SVG line chart (stdlib only)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SERIES_COLORS = (
    "#3182ce",
    "#d69e2e",
    "#38a169",
    "#e53e3e",
    "#805ad5",
    "#dd6b20",
)

WIDTH = 800
HEIGHT = 480
MARGIN_LEFT = 70
MARGIN_RIGHT = 170
MARGIN_TOP = 40
MARGIN_BOTTOM = 60
PLOT_WIDTH = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
PLOT_HEIGHT = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM


def _load_series(path: Path) -> tuple[str, list[int], list[float]]:
    episodes: list[int] = []
    rewards: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["eval_reward"]))
    return path.stem, episodes, rewards


def _scale(value: float, vmin: float, vmax: float, out_min: float, out_max: float) -> float:
    if vmax == vmin:
        return (out_min + out_max) / 2
    return out_min + (value - vmin) / (vmax - vmin) * (out_max - out_min)


def plot_learning_curves(csv_paths: list[Path | str], output_svg: Path | str) -> None:
    """Render episode vs eval_reward polylines for each CSV to an SVG file."""
    paths = [Path(path) for path in csv_paths]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found: {path}")

    series = [_load_series(path) for path in paths]
    all_episodes = [episode for _, episodes, _ in series for episode in episodes]
    all_rewards = [reward for _, _, rewards in series for reward in rewards]
    if not all_episodes:
        raise ValueError("No data rows found in the provided CSV files.")

    x_min, x_max = min(all_episodes), max(all_episodes)
    y_min, y_max = min(all_rewards), max(all_rewards)
    if y_min == y_max:
        y_min -= 0.05
        y_max += 0.05

    polylines: list[str] = []
    legend_rows: list[str] = []
    for index, (label, episodes, rewards) in enumerate(series):
        color = SERIES_COLORS[index % len(SERIES_COLORS)]
        points = []
        for episode, reward in zip(episodes, rewards, strict=True):
            x = _scale(episode, x_min, x_max, MARGIN_LEFT, MARGIN_LEFT + PLOT_WIDTH)
            y = _scale(
                reward,
                y_min,
                y_max,
                MARGIN_TOP + PLOT_HEIGHT,
                MARGIN_TOP,
            )
            points.append(f"{x:.1f},{y:.1f}")
        polylines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" '
            f'points="{" ".join(points)}"/>'
        )
        legend_y = MARGIN_TOP + index * 22
        legend_rows.append(
            f'<line x1="{WIDTH - MARGIN_RIGHT + 10}" y1="{legend_y}" '
            f'x2="{WIDTH - MARGIN_RIGHT + 35}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
            f'<text x="{WIDTH - MARGIN_RIGHT + 42}" y="{legend_y + 5}">{label}</text>'
        )

    x_axis_y = MARGIN_TOP + PLOT_HEIGHT
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
<rect width="100%" height="100%" fill="#ffffff"/>
<style>text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 13px; fill: #1a202c; }}</style>
<text x="{MARGIN_LEFT}" y="24" font-size="18" font-weight="700">Learning curves (eval reward)</text>
<line x1="{MARGIN_LEFT}" y1="{MARGIN_TOP}" x2="{MARGIN_LEFT}" y2="{x_axis_y}" stroke="#718096" stroke-width="1.5"/>
<line x1="{MARGIN_LEFT}" y1="{x_axis_y}" x2="{MARGIN_LEFT + PLOT_WIDTH}" y2="{x_axis_y}" stroke="#718096" stroke-width="1.5"/>
<text x="{MARGIN_LEFT + PLOT_WIDTH / 2:.1f}" y="{HEIGHT - 18}" text-anchor="middle">episode</text>
<text x="18" y="{MARGIN_TOP + PLOT_HEIGHT / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {MARGIN_TOP + PLOT_HEIGHT / 2:.1f})">eval_reward</text>
{"".join(polylines)}
{"".join(legend_rows)}
</svg>
'''
    Path(output_svg).write_text(svg, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot learning-curve CSV files as an SVG line chart."
    )
    parser.add_argument(
        "csv_paths",
        nargs="+",
        help="One or more learning_curve.csv files to plot",
    )
    parser.add_argument(
        "--output",
        default="learning_curves.svg",
        help="Output SVG path (default: learning_curves.svg)",
    )
    args = parser.parse_args(argv)

    try:
        plot_learning_curves(args.csv_paths, args.output)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
