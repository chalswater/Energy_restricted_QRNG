from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_OUTPUT_DIR, SIM_OUTPUT_SCHEMA_VERSION, SimulationConfig
from .plot_results import plot_all
from .simulation import config_as_dict, run_all_strategies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SDI QRNG chip block-level simulation.")
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = SimulationConfig().with_overrides(
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        seed=args.seed,
    )
    cfg.validate()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records, summaries = run_all_strategies(cfg)

    records_path = output_dir / "sdi_qrng_chip_records.json"
    summary_path = output_dir / "sdi_qrng_chip_summary.json"
    records_path.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    summary_payload = {
        "schema_version": SIM_OUTPUT_SCHEMA_VERSION,
        "config": config_as_dict(cfg),
        "summaries": summaries,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    figure_paths = [] if args.no_plots else plot_all(all_records, summaries, output_dir)

    print(json.dumps(
        {
            "summary_path": str(summary_path),
            "records_path": str(records_path),
            "figures": [str(p) for p in figure_paths],
            "summaries": summaries,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
