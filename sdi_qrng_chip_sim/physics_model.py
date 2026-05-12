from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig


@dataclass
class DriftState:
    delta_mu: float = 0.0
    delta_eta: float = 0.0
    delta_dark: float = 0.0
    delta_mod_bias: float = 0.0
    temperature_c: float = 25.0


def update_drift(state: DriftState, cfg: SimulationConfig, rng: np.random.Generator) -> DriftState:
    temp_pull = (1.0 - cfg.rho_temp) * (cfg.ambient_temp_c - state.temperature_c)
    return DriftState(
        delta_mu=cfg.rho_mu * state.delta_mu + cfg.sigma_mu * float(rng.normal()),
        delta_eta=cfg.rho_eta * state.delta_eta + cfg.sigma_eta * float(rng.normal()),
        delta_dark=cfg.rho_dark * state.delta_dark + cfg.sigma_dark * float(rng.normal()),
        delta_mod_bias=cfg.rho_mod_bias * state.delta_mod_bias + cfg.sigma_mod_bias * float(rng.normal()),
        temperature_c=state.temperature_c + temp_pull + cfg.sigma_temp_c * float(rng.normal()),
    )


def apply_physical_drift(
    mu_values: np.ndarray,
    state: DriftState,
    cfg: SimulationConfig,
) -> tuple[np.ndarray, float, float]:
    temp_delta = state.temperature_c - cfg.temp0_c
    temp_mu_scale = 1.0 + cfg.laser_temp_coeff * temp_delta
    mu_drifted = (mu_values + state.delta_mu) * temp_mu_scale

    if mu_drifted.size >= 2:
        high = float(max(mu_drifted[1], cfg.mu_min))
        extinction_floor = high / (10.0 ** (cfg.mod_extinction_ratio_db / 10.0))
        leakage = high * cfg.mod_bias_to_mu_fraction * max(state.delta_mod_bias, 0.0)
        signal_loss = high * cfg.mod_bias_to_mu_fraction * max(-state.delta_mod_bias, 0.0)
        mu_drifted[0] = max(float(mu_drifted[0]), extinction_floor + leakage)
        mu_drifted[1] = high - signal_loss

    mu_drifted = np.clip(mu_drifted, cfg.mu_min, cfg.mu_max)
    eta_current = float(np.clip((cfg.eta0 + state.delta_eta) * (1.0 + cfg.eta_temp_coeff * temp_delta), 0.01, 0.9))
    p_dark_temp = cfg.p_dark0 * float(np.exp(cfg.dark_temp_coeff * temp_delta))
    p_dark_current = float(np.clip(p_dark_temp + state.delta_dark, 0.0, 1e-2))
    return mu_drifted, eta_current, p_dark_current
