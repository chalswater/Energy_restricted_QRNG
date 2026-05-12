from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .config import DEFAULT_OUTPUT_DIR, STRATEGIES, SimulationConfig
from .plot_results import draw_line_chart
from .simulation import run_strategy


PROPOSED = next(strategy for strategy in STRATEGIES if strategy.name == "proposed_dynamic_event_cal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SDI QRNG chip parameter scans.")
    parser.add_argument("--num-blocks", type=int, default=120)
    parser.add_argument("--block-size", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def run_dark_count_scan(base_cfg: SimulationConfig) -> list[dict]:
    rows: list[dict] = []
    for p_dark in np.logspace(-6, -3, 9):
        cfg = replace(base_cfg, p_dark0=float(p_dark))
        _, summary = run_strategy(cfg, PROPOSED)
        rows.append({"p_dark0": float(p_dark), **summary})
    return rows


def run_eta_scan(base_cfg: SimulationConfig) -> list[dict]:
    rows: list[dict] = []
    for eta in np.linspace(0.08, 0.42, 9):
        cfg = replace(base_cfg, eta0=float(eta))
        _, summary = run_strategy(cfg, PROPOSED)
        rows.append({"eta0": float(eta), **summary})
    return rows


def plot_scan(rows: list[dict], x_key: str, output_dir: Path, prefix: str) -> list[Path]:
    paths: list[Path] = []
    for metric, ylabel in (
        ("certified_rate", "certified output rate"),
        ("stop_rate", "STOP blocks / all blocks"),
        ("mean_h_min", "mean single-round h_min"),
    ):
        series = {
            metric: [(float(row[x_key]), float(row[metric])) for row in rows],
        }
        path = output_dir / f"{prefix}_{metric}.png"
        draw_line_chart(
            path,
            series,
            f"{prefix} {metric}",
            ylabel,
            xlabel=x_key,
            y_min=0.0,
            y_max=None,
        )
        paths.append(path)
    return paths


def main() -> None:
    args = parse_args()
    cfg = SimulationConfig().with_overrides(num_blocks=args.num_blocks, block_size=args.block_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dark_rows = run_dark_count_scan(cfg)
    eta_rows = run_eta_scan(cfg)
    payload = {"dark_count_scan": dark_rows, "eta_scan": eta_rows}

    scan_path = args.output_dir / "sdi_qrng_chip_scans.json"
    scan_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    figures = []
    figures.extend(plot_scan(dark_rows, "p_dark0", args.output_dir, "dark_count_scan"))
    figures.extend(plot_scan(eta_rows, "eta0", args.output_dir, "eta_scan"))

    print(json.dumps({"scan_path": str(scan_path), "figures": [str(p) for p in figures]}, indent=2))


if __name__ == "__main__":
    main()

