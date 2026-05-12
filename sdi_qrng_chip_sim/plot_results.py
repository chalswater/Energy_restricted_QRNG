from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH = 1400
HEIGHT = 840
BG = "#f7f8fa"
FG = "#1f2328"
GRID = "#d6dce3"
FRAME = "#9aa6b2"
COLORS = {
    "baseline_fixed_no_cal": "#8c564b",
    "baseline_fixed_periodic_cal": "#ff7f0e",
    "dynamic_no_cal": "#1f77b4",
    "proposed_dynamic_event_cal": "#2ca02c",
}
FALLBACK_COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#6a3d9a", "#8c564b")


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_line_chart(
    path: Path,
    series: dict[str, list[tuple[float, float]]],
    title: str,
    ylabel: str,
    *,
    xlabel: str = "block index",
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    f_title = font(30)
    f_axis = font(20)
    f_legend = font(18)

    left, top, right, bottom = 92, 86, WIDTH - 48, HEIGHT - 96
    draw.text((left, 34), title, fill=FG, font=f_title)
    draw.rectangle((left, top, right, bottom), outline=FRAME, width=2)

    all_x = [x for pts in series.values() for x, _ in pts]
    all_y = [y for pts in series.values() for _, y in pts]
    xmin, xmax = min(all_x), max(all_x)
    ymin = min(all_y) if y_min is None else y_min
    ymax = max(all_y) if y_max is None else y_max
    if xmax <= xmin:
        xmax = xmin + 1
    if ymax <= ymin:
        ymax = ymin + 1.0

    for i in range(6):
        frac = i / 5
        x = left + frac * (right - left)
        y = bottom - frac * (bottom - top)
        draw.line((x, top, x, bottom), fill=GRID, width=1)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((left - 76, y - 10), f"{ymin + frac * (ymax - ymin):.3f}", fill=FG, font=f_axis)

    draw.text((left, bottom + 36), xlabel, fill=FG, font=f_axis)
    draw.text((16, top - 34), ylabel, fill=FG, font=f_axis)

    series_items = list(series.items())
    color_by_name = {
        name: COLORS.get(name, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])
        for idx, (name, _) in enumerate(series_items)
    }

    for name, pts in series_items:
        color = color_by_name[name]
        scaled = []
        for x, y in pts:
            sx = left + (x - xmin) / (xmax - xmin) * (right - left)
            sy = bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
            scaled.append((sx, sy))
        if len(scaled) >= 2:
            draw.line(scaled, fill=color, width=3)
        elif scaled:
            x, y = scaled[0]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

    legend_x = right - 390
    legend_y = top + 18
    for name in series:
        color = color_by_name[name]
        draw.rectangle((legend_x, legend_y + 4, legend_x + 28, legend_y + 16), fill=color)
        draw.text((legend_x + 38, legend_y), name, fill=FG, font=f_legend)
        legend_y += 26

    img.save(path)


def draw_stacked_line_charts(
    path: Path,
    panels: list[dict[str, object]],
    title: str,
    *,
    xlabel: str = "block index",
) -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    f_title = font(28)
    f_axis = font(18)
    f_legend = font(16)

    draw.text((92, 28), title, fill=FG, font=f_title)
    panel_height = 300
    panel_boxes = [
        (92, 86, WIDTH - 48, 86 + panel_height),
        (92, 456, WIDTH - 48, 456 + panel_height),
    ]

    for panel_idx, panel in enumerate(panels):
        box = panel_boxes[panel_idx]
        left, top, right, bottom = box
        series = panel["series"]  # type: ignore[assignment]
        ylabel = str(panel["ylabel"])
        y_min = panel.get("y_min")  # type: ignore[union-attr]
        y_max = panel.get("y_max")  # type: ignore[union-attr]
        draw.rectangle(box, outline=FRAME, width=2)
        draw.text((left, top - 28), str(panel["title"]), fill=FG, font=f_axis)

        all_x = [x for pts in series.values() for x, _ in pts]
        all_y = [y for pts in series.values() for _, y in pts]
        xmin, xmax = min(all_x), max(all_x)
        ymin = min(all_y) if y_min is None else float(y_min)
        ymax = max(all_y) if y_max is None else float(y_max)
        if xmax <= xmin:
            xmax = xmin + 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0

        for i in range(5):
            frac = i / 4
            y = bottom - frac * (bottom - top)
            draw.line((left, y, right, y), fill=GRID, width=1)
            draw.text((left - 74, y - 9), f"{ymin + frac * (ymax - ymin):.3f}", fill=FG, font=f_axis)

        draw.text((16, top + 6), ylabel, fill=FG, font=f_axis)
        if panel_idx == len(panels) - 1:
            draw.text((left, bottom + 30), xlabel, fill=FG, font=f_axis)

        series_items = list(series.items())
        color_by_name = {
            name: COLORS.get(name, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])
            for idx, (name, _) in enumerate(series_items)
        }
        for name, pts in series_items:
            color = color_by_name[name]
            scaled = []
            for x, y in pts:
                sx = left + (x - xmin) / (xmax - xmin) * (right - left)
                sy = bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
                scaled.append((sx, sy))
            if len(scaled) >= 2:
                draw.line(scaled, fill=color, width=3)

        legend_x = right - 340
        legend_y = top + 12
        for name in series:
            color = color_by_name[name]
            draw.rectangle((legend_x, legend_y + 4, legend_x + 24, legend_y + 15), fill=color)
            draw.text((legend_x + 34, legend_y), name, fill=FG, font=f_legend)
            legend_y += 22

    img.save(path)


def draw_bar_chart(path: Path, summaries: list[dict], key: str, title: str, ylabel: str) -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)
    f_title = font(30)
    f_axis = font(18)

    left, top, right, bottom = 92, 86, WIDTH - 48, HEIGHT - 170
    draw.text((left, 34), title, fill=FG, font=f_title)
    draw.rectangle((left, top, right, bottom), outline=FRAME, width=2)

    values = [float(s[key]) for s in summaries]
    ymax = max(max(values) * 1.15, 0.01)
    names = [str(s["strategy"]) for s in summaries]
    bar_gap = 28
    bar_w = (right - left - bar_gap * (len(names) + 1)) / len(names)

    for i in range(6):
        frac = i / 5
        y = bottom - frac * (bottom - top)
        draw.line((left, y, right, y), fill=GRID, width=1)
        draw.text((left - 70, y - 10), f"{frac * ymax:.3f}", fill=FG, font=f_axis)
    draw.text((16, top - 34), ylabel, fill=FG, font=f_axis)

    for idx, (name, value) in enumerate(zip(names, values)):
        x0 = left + bar_gap + idx * (bar_w + bar_gap)
        x1 = x0 + bar_w
        y0 = bottom - value / ymax * (bottom - top)
        color = COLORS.get(name, "#333333")
        draw.rectangle((x0, y0, x1, bottom), fill=color)
        draw.text((x0, y0 - 26), f"{value:.3f}", fill=FG, font=f_axis)
        draw.text((x0, bottom + 18), name.replace("_", "\n"), fill=FG, font=f_axis)

    img.save(path)


def plot_all(all_records: dict[str, list[dict]], summaries: list[dict], output_dir: Path) -> list[Path]:
    ensure_output_dir(output_dir)
    paths: list[Path] = []

    for metric, ylabel, filename, y_min, y_max in (
        ("h_min", "single-round h_min", "h_min_vs_block.png", 0.0, None),
        ("omega_high", "energy upper bound", "omega_high_vs_block.png", 0.0, None),
        ("certified_rate", "certified output rate", "certified_rate_vs_block.png", 0.0, None),
    ):
        series = {
            name: [(float(r["block"]), float(r[metric])) for r in records]
            for name, records in all_records.items()
        }
        path = output_dir / filename
        draw_line_chart(
            path,
            series,
            filename.replace("_", " ").replace(".png", ""),
            ylabel,
            y_min=y_min,
            y_max=y_max,
        )
        paths.append(path)

    for key, filename, ylabel in (
        ("stop_rate", "stop_rate_comparison.png", "STOP blocks / all blocks"),
        ("unsafe_block_rate", "unsafe_block_rate_comparison.png", "unsafe fixed-output blocks"),
    ):
        path = output_dir / filename
        draw_bar_chart(path, summaries, key, filename.replace("_", " ").replace(".png", ""), ylabel)
        paths.append(path)

    return paths
