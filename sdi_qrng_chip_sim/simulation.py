from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .calibration import pending_calibration_delay, should_calibrate, update_mu_control
from .config import STRATEGIES, SimulationConfig, Strategy
from .detector_model import estimate_dark_count_prob, generate_block
from .entropy_estimator import EntropyCurve, estimate_block_entropy
from .extractor_control import choose_mode, compute_strategy_output_length
from .physics_model import DriftState, apply_physical_drift, update_drift


def run_strategy(
    cfg: SimulationConfig,
    strategy: Strategy,
    *,
    entropy_curve: EntropyCurve | None = None,
) -> tuple[list[dict[str, float | int | str | bool]], dict[str, float | int | str]]:
    cfg.validate()
    rng = np.random.default_rng(cfg.seed)
    curve = entropy_curve or EntropyCurve(cfg)

    mu_values = cfg.mu_initial_array.copy()
    drift = DriftState(temperature_c=cfg.temp0_c)
    pending_mu_updates: list[tuple[int, np.ndarray]] = []
    seed_pool_bits = int(cfg.seed_pool_initial_bits)

    total_raw = 0
    total_emitted = 0
    total_certified = 0
    total_seed_consumed = 0
    total_seed_refreshed = 0
    stop_blocks = 0
    unsafe_blocks = 0
    calibration_events = 0
    calibration_applied_events = 0
    seed_starved_blocks = 0
    records: list[dict[str, float | int | str | bool]] = []

    for block_index in range(cfg.num_blocks):
        due_updates = [update for update in pending_mu_updates if update[0] <= block_index]
        pending_mu_updates = [update for update in pending_mu_updates if update[0] > block_index]
        if due_updates:
            mu_values = due_updates[-1][1].copy()
            calibration_applied_events += len(due_updates)

        drift = update_drift(drift, cfg, rng)
        mu_drifted, eta_current, p_dark_current = apply_physical_drift(mu_values, drift, cfg)

        xs, outcomes, monitor_values = generate_block(
            cfg,
            mu_drifted,
            eta_current,
            p_dark_current,
            drift.temperature_c,
            rng,
        )
        est = estimate_block_entropy(xs, outcomes, monitor_values, cfg, curve)
        dark_count = estimate_dark_count_prob(max(1000, cfg.block_size // 10), p_dark_current, rng)

        decision = compute_strategy_output_length(
            cfg.block_size,
            est.h_min,
            cfg,
            strategy,
        )
        mode = choose_mode(est.h_min, cfg)
        emitted_m = decision.emitted_bits
        certified_m = decision.certified_bits
        stop_blocked_bits = 0

        if mode == "STOP":
            stop_blocks += 1
            certified_m = 0
            if strategy.dynamic_extraction:
                stop_blocked_bits = emitted_m
                emitted_m = 0

        seed_required_bits = max(0, cfg.block_size + emitted_m - 1) if emitted_m > 0 else 0
        seed_consumed_bits = 0
        seed_refreshed_bits = 0
        seed_starved = seed_required_bits > seed_pool_bits
        if seed_starved:
            seed_starved_blocks += 1
            stop_blocked_bits += emitted_m
            emitted_m = 0
            certified_m = 0
            seed_required_bits = 0
        else:
            seed_consumed_bits = seed_required_bits
            seed_pool_bits -= seed_consumed_bits

        unsafe = emitted_m > certified_m
        unsafe_emitted_bits = max(0, emitted_m - certified_m)

        if certified_m > 0:
            seed_refreshed_bits = int(certified_m * cfg.seed_refresh_fraction)
            seed_pool_bits += seed_refreshed_bits

        total_seed_consumed += seed_consumed_bits
        total_seed_refreshed += seed_refreshed_bits

        if unsafe:
            unsafe_blocks += 1

        if should_calibrate(block_index, est.omega_high, est.h_min, dark_count, cfg, strategy):
            calibration_events += 1
            updated_mu_values = update_mu_control(mu_values, est.omega_high, est.h_min, cfg)
            delay = pending_calibration_delay(cfg)
            if delay == 0:
                mu_values = updated_mu_values
                calibration_applied_events += 1
            else:
                pending_mu_updates.append((block_index + delay, updated_mu_values))

        total_raw += cfg.block_size
        total_emitted += emitted_m
        total_certified += certified_m

        records.append(
            {
                "strategy": strategy.name,
                "block": block_index,
                "W_hat": est.W_hat,
                "W_low": est.W_low,
                "omega_high": est.omega_high,
                "h_min": est.h_min,
                "emitted_bits": emitted_m,
                "certified_bits": certified_m,
                "safe_bits": decision.safe_bits,
                "unsafe_emitted_bits": unsafe_emitted_bits,
                "stop_blocked_bits": stop_blocked_bits,
                "seed_required_bits": seed_required_bits,
                "seed_consumed_bits": seed_consumed_bits,
                "seed_refreshed_bits": seed_refreshed_bits,
                "seed_pool_bits": seed_pool_bits,
                "seed_starved": seed_starved,
                "emitted_rate": emitted_m / cfg.block_size,
                "certified_rate": certified_m / cfg.block_size,
                "safe_rate": decision.safe_bits / cfg.block_size,
                "mode": mode,
                "stop_certified_output": mode == "STOP",
                "unsafe_output": unsafe,
                "unsafe_fixed_output": unsafe,
                "calibration_events_so_far": calibration_events,
                "calibration_applied_events_so_far": calibration_applied_events,
                "pending_calibration_updates": len(pending_mu_updates),
                "eta": eta_current,
                "p_dark": p_dark_current,
                "temperature_c": drift.temperature_c,
                "mod_bias": drift.delta_mod_bias,
                "dark_count_est": dark_count,
                "mu0": float(mu_drifted[0]),
                "mu1": float(mu_drifted[1]) if mu_drifted.size > 1 else float(mu_drifted[0]),
            }
        )

    summary: dict[str, float | int | str] = {
        "strategy": strategy.name,
        "num_blocks": cfg.num_blocks,
        "block_size": cfg.block_size,
        "raw_bits": total_raw,
        "emitted_bits": total_emitted,
        "certified_bits": total_certified,
        "unsafe_emitted_bits": int(sum(int(r["unsafe_emitted_bits"]) for r in records)),
        "stop_blocked_bits": int(sum(int(r["stop_blocked_bits"]) for r in records)),
        "seed_consumed_bits": total_seed_consumed,
        "seed_refreshed_bits": total_seed_refreshed,
        "seed_final_pool_bits": seed_pool_bits,
        "emitted_rate": total_emitted / max(total_raw, 1),
        "certified_rate": total_certified / max(total_raw, 1),
        "unsafe_emitted_rate": float(sum(int(r["unsafe_emitted_bits"]) for r in records) / max(total_raw, 1)),
        "stop_rate": stop_blocks / max(cfg.num_blocks, 1),
        "unsafe_block_rate": unsafe_blocks / max(cfg.num_blocks, 1),
        "seed_starved_rate": seed_starved_blocks / max(cfg.num_blocks, 1),
        "calibration_events": calibration_events,
        "calibration_applied_events": calibration_applied_events,
        "mean_h_min": float(np.mean([r["h_min"] for r in records])),
        "mean_omega_high": float(np.mean([r["omega_high"] for r in records])),
        "mean_temperature_c": float(np.mean([r["temperature_c"] for r in records])),
    }
    return records, summary


def run_all_strategies(cfg: SimulationConfig) -> tuple[dict[str, list[dict]], list[dict]]:
    all_records: dict[str, list[dict]] = {}
    summaries: list[dict] = []
    for strategy in STRATEGIES:
        records, summary = run_strategy(cfg, strategy)
        all_records[strategy.name] = records
        summaries.append(summary)
    return all_records, summaries
def config_as_dict(cfg: SimulationConfig) -> dict:
    data = asdict(cfg)
    data["mu_initial"] = list(cfg.mu_initial)
    return data
