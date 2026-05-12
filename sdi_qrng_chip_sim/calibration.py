from __future__ import annotations

import numpy as np

from .config import SimulationConfig, Strategy


def update_mu_control(
    mu_values: np.ndarray,
    omega_high: float,
    h_min: float,
    cfg: SimulationConfig,
) -> np.ndarray:
    """Keep the monitored max energy near target while preserving contrast."""
    scale = 1.0

    if omega_high > cfg.omega_max:
        scale_error = omega_high - cfg.mu_target
        scale = float(np.exp(-cfg.Kp_mu * scale_error))
    elif h_min < cfg.h_warn and omega_high < cfg.omega_max:
        headroom = max(cfg.omega_max - omega_high, 0.0) / max(cfg.omega_max - cfg.mu_target, 1e-12)
        scale = 1.0 + 0.35 * cfg.Kp_mu * min(headroom, 1.0)
    else:
        scale_error = omega_high - cfg.mu_target
        scale = float(np.exp(-0.05 * cfg.Kp_mu * scale_error))

    new_values = mu_values * scale
    return np.clip(new_values, cfg.mu_min, cfg.mu_max)


def pending_calibration_delay(cfg: SimulationConfig) -> int:
    return max(int(cfg.calibration_latency_blocks), 0)


def should_calibrate(
    block_index: int,
    omega_high: float,
    h_min: float,
    dark_count: float,
    cfg: SimulationConfig,
    strategy: Strategy,
) -> bool:
    if not strategy.calibration:
        return False

    periodic = strategy.fixed_periodic and block_index % cfg.calibration_interval == 0
    event = (
        strategy.event_triggered
        and (
            omega_high > cfg.omega_max
            or h_min < cfg.h_warn
            or dark_count > cfg.dark_count_max
        )
    )
    return periodic or event
