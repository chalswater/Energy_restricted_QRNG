from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import DEFAULT_OUTPUT_DIR, STRATEGIES, SimulationConfig
from .plot_results import draw_line_chart, draw_stacked_line_charts
from .reproducibility import metric_stats, write_manifest
from .simulation import run_strategy


STRATEGY_BY_NAME = {strategy.name: strategy for strategy in STRATEGIES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate patent-oriented SDI QRNG simulation figures.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "patent_figures")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--seed-step", type=int, default=17)
    return parser.parse_args()


def replicate_stats(cfg: SimulationConfig, strategy_names: list[str], seeds: list[int]) -> dict[str, dict[str, dict[str, float]]]:
    metrics = ("certified_rate", "stop_rate", "unsafe_block_rate", "unsafe_emitted_rate", "mean_h_min", "mean_omega_high")
    rows: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in metrics}
        for name in strategy_names
    }
    for seed in seeds:
        seeded_cfg = replace(cfg, seed=seed)
        for name in strategy_names:
            _, summary = run_strategy(seeded_cfg, STRATEGY_BY_NAME[name])
            for metric in metrics:
                rows[name][metric].append(float(summary[metric]))
    return {
        name: {metric: metric_stats(values) for metric, values in metric_rows.items()}
        for name, metric_rows in rows.items()
    }


def scenario_drift_fixed_vs_dynamic(output_dir: Path, seeds: list[int]) -> dict:
    cfg = replace(
        SimulationConfig(),
        num_blocks=180,
        block_size=8_000,
        sigma_mu=0.005,
        rho_mu=0.998,
        fixed_extractor_rate=0.025,
    )
    fixed_records, fixed_summary = run_strategy(cfg, STRATEGY_BY_NAME["baseline_fixed_no_cal"])
    dynamic_records, dynamic_summary = run_strategy(cfg, STRATEGY_BY_NAME["dynamic_no_cal"])

    path = output_dir / "patent_fig_1_drift_fixed_vs_dynamic.png"
    draw_line_chart(
        path,
        {
            "fixed emitted rate": [(float(r["block"]), float(r["emitted_rate"])) for r in fixed_records],
            "fixed certified rate": [(float(r["block"]), float(r["certified_rate"])) for r in fixed_records],
            "dynamic certified rate": [(float(r["block"]), float(r["certified_rate"])) for r in dynamic_records],
        },
        "Patent Fig. 1 Drift: Fixed Compression vs Dynamic Compression",
        "output bits / raw bit",
        y_min=0.0,
    )
    return {
        "figure": str(path),
        "scenario": "drift_fixed_vs_dynamic",
        "description": "在光源功率漂移条件下，固定压缩率持续按预设速率输出，而动态压缩率随认证熵逐块调整输出长度。",
        "config": {
            "num_blocks": cfg.num_blocks,
            "block_size": cfg.block_size,
            "sigma_mu": cfg.sigma_mu,
            "rho_mu": cfg.rho_mu,
            "fixed_extractor_rate": cfg.fixed_extractor_rate,
        },
        "summaries": [fixed_summary, dynamic_summary],
        "replicate_stats": replicate_stats(cfg, ["baseline_fixed_no_cal", "dynamic_no_cal"], seeds),
    }


def scenario_anomaly_event_calibration(output_dir: Path, seeds: list[int]) -> dict:
    cfg = replace(
        SimulationConfig(),
        num_blocks=160,
        block_size=8_000,
        mu_initial=(0.06, 0.34),
        sigma_mu=0.004,
        rho_mu=0.997,
        omega_max=0.30,
        mu_target=0.25,
    )
    no_cal_records, no_cal_summary = run_strategy(cfg, STRATEGY_BY_NAME["dynamic_no_cal"])
    proposed_records, proposed_summary = run_strategy(cfg, STRATEGY_BY_NAME["proposed_dynamic_event_cal"])

    path = output_dir / "patent_fig_2_anomaly_event_calibration.png"
    cap_series = [(float(r["block"]), cfg.omega_max) for r in no_cal_records]
    draw_stacked_line_charts(
        path,
        [
            {
                "title": "Energy upper bound",
                "ylabel": "omega_high",
                "series": {
                    "no calibration omega_high": [(float(r["block"]), float(r["omega_high"])) for r in no_cal_records],
                    "event calibration omega_high": [(float(r["block"]), float(r["omega_high"])) for r in proposed_records],
                    "security cap": cap_series,
                },
                "y_min": 0.0,
            },
            {
                "title": "Certified output rate",
                "ylabel": "certified bits / raw bit",
                "series": {
                    "no calibration certified rate": [(float(r["block"]), float(r["certified_rate"])) for r in no_cal_records],
                    "event calibration certified rate": [(float(r["block"]), float(r["certified_rate"])) for r in proposed_records],
                },
                "y_min": 0.0,
            },
        ],
        "Patent Fig. 2 Anomaly: No Calibration vs Event-Triggered Calibration",
    )
    return {
        "figure": str(path),
        "scenario": "anomaly_event_calibration",
        "description": "当片上监测能量超过安全包络时，事件触发自校准降低 omega_high，并恢复认证随机数输出。",
        "config": {
            "num_blocks": cfg.num_blocks,
            "block_size": cfg.block_size,
            "mu_initial": list(cfg.mu_initial),
            "sigma_mu": cfg.sigma_mu,
            "omega_max": cfg.omega_max,
            "mu_target": cfg.mu_target,
        },
        "summaries": [no_cal_summary, proposed_summary],
        "replicate_stats": replicate_stats(cfg, ["dynamic_no_cal", "proposed_dynamic_event_cal"], seeds),
    }


def scenario_unsafe_blocks(output_dir: Path, seeds: list[int]) -> dict:
    cfg = replace(
        SimulationConfig(),
        num_blocks=180,
        block_size=8_000,
        sigma_mu=0.004,
        rho_mu=0.998,
        fixed_extractor_rate=0.030,
    )
    fixed_records, fixed_summary = run_strategy(cfg, STRATEGY_BY_NAME["baseline_fixed_no_cal"])
    proposed_records, proposed_summary = run_strategy(cfg, STRATEGY_BY_NAME["proposed_dynamic_event_cal"])

    fixed_cumulative = cumulative_fraction([bool(r["unsafe_output"]) for r in fixed_records])
    proposed_cumulative = cumulative_fraction([bool(r["unsafe_output"]) for r in proposed_records])
    path = output_dir / "patent_fig_3_unsafe_block_safety.png"
    draw_line_chart(
        path,
        {
            "fixed compression unsafe fraction": fixed_cumulative,
            "proposed unsafe fraction": proposed_cumulative,
            "fixed emitted rate": [(float(r["block"]), float(r["emitted_rate"])) for r in fixed_records],
            "proposed certified rate": [(float(r["block"]), float(r["certified_rate"])) for r in proposed_records],
        },
        "Patent Fig. 3 Safety: Fixed Output Produces Unsafe Blocks",
        "fraction or output rate",
        y_min=0.0,
    )
    return {
        "figure": str(path),
        "scenario": "unsafe_block_safety",
        "description": "固定压缩率可能输出超过熵下界可认证的比特数；本方案通过动态输出控制避免 unsafe block。",
        "config": {
            "num_blocks": cfg.num_blocks,
            "block_size": cfg.block_size,
            "sigma_mu": cfg.sigma_mu,
            "fixed_extractor_rate": cfg.fixed_extractor_rate,
        },
        "summaries": [fixed_summary, proposed_summary],
        "replicate_stats": replicate_stats(cfg, ["baseline_fixed_no_cal", "proposed_dynamic_event_cal"], seeds),
    }


def cumulative_fraction(flags: list[bool]) -> list[tuple[float, float]]:
    count = 0
    points: list[tuple[float, float]] = []
    for idx, flag in enumerate(flags):
        count += int(flag)
        points.append((float(idx), count / float(idx + 1)))
    return points


def write_report(output_dir: Path, payload: dict) -> Path:
    report_path = output_dir / "patent_figure_report_zh.md"
    lines = [
        "# SDI QRNG 芯片专利图表仿真说明",
        "",
        "本报告对应“一种具有片上自校准和动态熵提取控制的半设备无关量子随机数芯片及其随机数生成方法”的实施例仿真图。",
        "",
        "## 图表列表",
        "",
    ]
    for idx, item in enumerate(payload["figures"], start=1):
        lines.extend(
            [
                f"### 图 {idx}: {item['scenario']}",
                "",
                f"- 文件：`{item['figure']}`",
                f"- 说明：{item['description']}",
                "- 关键结果：",
            ]
        )
        for summary in item["summaries"]:
            lines.append(
                f"  - `{summary['strategy']}`: certified_rate={summary['certified_rate']:.6f}, "
                f"stop_rate={summary['stop_rate']:.6f}, unsafe_block_rate={summary['unsafe_block_rate']:.6f}, "
                f"unsafe_emitted_rate={summary['unsafe_emitted_rate']:.6f}, calibration_events={summary['calibration_events']}"
            )
        lines.append("- 多 seed 统计：")
        for strategy_name, stats in item["replicate_stats"].items():
            cert = stats["certified_rate"]
            unsafe = stats["unsafe_block_rate"]
            lines.append(
                f"  - `{strategy_name}`: certified_rate_mean={cert['mean']:.6f} ± {cert['ci95_half_width']:.6f}, "
                f"unsafe_block_rate_mean={unsafe['mean']:.6f} ± {unsafe['ci95_half_width']:.6f}"
            )
        lines.append("")
    lines.extend(
        [
            "## 专利说明书可用结论",
            "",
            "1. 在漂移环境下，动态熵提取根据每个 block 的认证熵改变输出长度，避免固定压缩率过度输出。",
            "2. 在能量异常环境下，事件触发自校准把片上监测得到的能量上界拉回安全包络内，并恢复认证输出。",
            "3. 固定输出方案会出现 unsafe block；本方案在同等输入数据下通过动态输出长度控制避免 unsafe block。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_seed = SimulationConfig().seed
    seeds = [base_seed + idx * args.seed_step for idx in range(args.replicates)]

    figures = [
        scenario_drift_fixed_vs_dynamic(args.output_dir, seeds),
        scenario_anomaly_event_calibration(args.output_dir, seeds),
        scenario_unsafe_blocks(args.output_dir, seeds),
    ]
    payload = {"seeds": seeds, "figures": figures}
    summary_path = args.output_dir / "patent_figure_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path = write_report(args.output_dir, payload)
    manifest_path = args.output_dir / "patent_figure_manifest.json"
    generated_files = [str(item["figure"]) for item in figures] + [str(summary_path), str(report_path), str(manifest_path)]
    write_manifest(
        manifest_path,
        cfg=SimulationConfig(),
        generated_files=generated_files,
        seeds=seeds,
        command="python3 -m sdi_qrng_chip_sim.patent_figures",
    )

    print(json.dumps(
        {
            "summary_path": str(summary_path),
            "report_path": str(report_path),
            "manifest_path": str(manifest_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
