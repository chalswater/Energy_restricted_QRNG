from __future__ import annotations

import numpy as np

from .config import SimulationConfig


def click_probability(mu: np.ndarray, eta: float, p_dark: float) -> np.ndarray:
    p_click = 1.0 - (1.0 - p_dark) * np.exp(-eta * mu)
    return np.clip(p_click, 0.0, 1.0)


def generate_block(
    cfg: SimulationConfig,
    mu_values: np.ndarray,
    eta: float,
    p_dark: float,
    temperature_c: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate one prepare-and-measure block with click/no-click detection."""
    xs = rng.integers(0, cfg.num_inputs, size=cfg.block_size, dtype=np.int16)
    mu_rounds = mu_values[xs]
    probs = click_probability(mu_rounds, eta, p_dark)
    outcomes = sample_detector_outcomes(probs, cfg, rng)

    tap_actual = cfg.tap_nominal_ratio * (1.0 + cfg.tap_ratio_relative_error)
    temp_scale = 1.0 + cfg.monitor_temp_coeff * (temperature_c - cfg.temp0_c)
    optical_tap_signal = tap_actual * mu_rounds
    monitor_current = (
        cfg.monitor_gain
        * temp_scale
        * (optical_tap_signal + cfg.monitor_pd_nonlinearity * optical_tap_signal**2)
        + cfg.monitor_offset
    )
    nominal_denominator = max(cfg.monitor_gain * cfg.tap_nominal_ratio, 1e-12)
    monitor_current += rng.normal(0.0, cfg.sigma_mon * nominal_denominator, cfg.block_size)
    monitor_values = monitor_current / nominal_denominator
    monitor_values = np.clip(monitor_values, 0.0, cfg.mu_max)

    return xs, outcomes, monitor_values


def sample_detector_outcomes(
    probs: np.ndarray,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    clicks = rng.random(probs.size) < probs

    if cfg.afterpulse_prob > 0.0:
        previous_click = np.zeros(probs.size, dtype=bool)
        previous_click[1:] = clicks[:-1]
        afterpulses = previous_click & (rng.random(probs.size) < cfg.afterpulse_prob)
        clicks |= afterpulses

    dead = int(max(cfg.detector_dead_time_rounds, 0))
    if dead > 0 and clicks.size:
        suppressed_until = -1
        for idx in np.flatnonzero(clicks):
            if idx <= suppressed_until:
                clicks[idx] = False
            else:
                suppressed_until = idx + dead

    if cfg.tdc_bin_nonuniformity > 0.0 and cfg.tdc_bin_count > 1:
        click_idx = np.flatnonzero(clicks)
        if click_idx.size:
            bins = rng.integers(0, cfg.tdc_bin_count, size=click_idx.size)
            phase = 2.0 * np.pi * bins / cfg.tdc_bin_count
            keep_prob = 1.0 - cfg.tdc_bin_nonuniformity * (1.0 + np.sin(phase)) / 2.0
            clicks[click_idx] &= rng.random(click_idx.size) < keep_prob

    return clicks.astype(np.uint8)


def estimate_dark_count_prob(num_windows: int, p_dark: float, rng: np.random.Generator) -> float:
    clicks = rng.binomial(num_windows, np.clip(p_dark, 0.0, 1.0))
    return float(clicks / max(num_windows, 1))


def estimate_detector_efficiency(p_click: float, p_dark: float, mu_test: float) -> float:
    if mu_test <= 0.0:
        return 0.0
    return float(np.clip((p_click - p_dark) / mu_test, 0.0, 1.0))
