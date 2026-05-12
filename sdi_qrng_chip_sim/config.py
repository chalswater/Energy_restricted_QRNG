from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT / "figures"
DEFAULT_LUT_PATH = ROOT / "lut_generation" / "lut_data.npz"
SIM_OUTPUT_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class SimulationConfig:
    num_blocks: int = 300
    block_size: int = 10_000
    num_inputs: int = 2
    seed: int = 20260512

    # Two-intensity prepare-and-measure MVP for the LUT-backed witness:
    # W = P(click | signal) - P(click | decoy).
    mu_initial: tuple[float, ...] = (0.05, 0.25)
    mu_min: float = 0.001
    mu_max: float = 0.5
    eta0: float = 0.22
    p_dark0: float = 1e-5

    rho_mu: float = 0.995
    sigma_mu: float = 0.002
    rho_eta: float = 0.995
    sigma_eta: float = 0.001
    rho_dark: float = 0.995
    sigma_dark: float = 1e-6

    # Temperature and analog drift model.
    temp0_c: float = 25.0
    ambient_temp_c: float = 25.0
    rho_temp: float = 0.990
    sigma_temp_c: float = 0.015
    laser_temp_coeff: float = 0.002
    eta_temp_coeff: float = -0.003
    dark_temp_coeff: float = 0.080

    # Modulator nonidealities.
    mod_extinction_ratio_db: float = 22.0
    rho_mod_bias: float = 0.995
    sigma_mod_bias: float = 0.002
    mod_bias_to_mu_fraction: float = 0.18

    # On-chip tap monitor path. monitor_values are estimated mean photon
    # numbers after dividing by nominal calibration gain; the upper-bound
    # estimator separately adds systematic margins for these uncertainties.
    tap_nominal_ratio: float = 0.02
    tap_ratio_relative_error: float = -0.01
    tap_ratio_error_bound: float = 0.02
    sigma_mon: float = 0.002
    monitor_gain: float = 1.0
    monitor_pd_nonlinearity: float = 0.025
    monitor_pd_nonlinearity_bound: float = 0.03
    monitor_temp_coeff: float = 0.001
    monitor_offset: float = 0.0

    # Detector timing impairments.
    detector_dead_time_rounds: int = 2
    afterpulse_prob: float = 0.002
    tdc_bin_count: int = 32
    tdc_bin_nonuniformity: float = 0.015

    # Control targets. omega_max is the hard health boundary; mu_target is the
    # target for the maximum monitored input energy.
    mu_target: float = 0.25
    omega_max: float = 0.30
    Kp_mu: float = 0.18

    # Runtime entropy estimation uses a precomputed h_min(W, omega) LUT.
    entropy_lut_path: str = str(DEFAULT_LUT_PATH)

    eps_W: float = 1e-10
    eps_omega: float = 1e-10
    eps_ext: float = 1e-12
    lambda_margin: int = 64
    witness_correction_scale: float = 0.35

    h_normal: float = 0.035
    h_warn: float = 0.010
    h_stop: float = 0.0
    dark_count_max: float = 2e-4

    fixed_extractor_rate: float = 0.03
    calibration_interval: int = 25
    calibration_latency_blocks: int = 2

    # Toeplitz seed budget model. The seed is public but must be independent of
    # the current raw block, so the simulator tracks seed consumption explicitly.
    seed_pool_initial_bits: int = 4_000_000
    seed_refresh_fraction: float = 0.02
    seed_low_watermark_bits: int = 250_000

    def with_overrides(
        self,
        *,
        num_blocks: int | None = None,
        block_size: int | None = None,
        seed: int | None = None,
    ) -> "SimulationConfig":
        updates = {}
        if num_blocks is not None:
            updates["num_blocks"] = num_blocks
        if block_size is not None:
            updates["block_size"] = block_size
        if seed is not None:
            updates["seed"] = seed
        return replace(self, **updates)

    @property
    def mu_initial_array(self) -> np.ndarray:
        return np.array(self.mu_initial, dtype=np.float64)

    def validate(self) -> None:
        if self.num_blocks < 0:
            raise ValueError("num_blocks must be nonnegative")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.num_inputs != len(self.mu_initial):
            raise ValueError("num_inputs must match len(mu_initial)")
        if self.num_inputs < 2:
            raise ValueError("at least two inputs are required for the click-contrast witness")
        if not (0.0 < self.eps_W < 1.0 and 0.0 < self.eps_omega < 1.0 and 0.0 < self.eps_ext < 1.0):
            raise ValueError("security epsilons must be in (0, 1)")
        if self.tap_nominal_ratio <= 0.0:
            raise ValueError("tap_nominal_ratio must be positive")
        if self.seed_pool_initial_bits < 0:
            raise ValueError("seed_pool_initial_bits must be nonnegative")


@dataclass(frozen=True)
class Strategy:
    name: str
    dynamic_extraction: bool
    calibration: bool
    event_triggered: bool = True
    fixed_periodic: bool = False


STRATEGIES: tuple[Strategy, ...] = (
    Strategy("baseline_fixed_no_cal", dynamic_extraction=False, calibration=False),
    Strategy(
        "baseline_fixed_periodic_cal",
        dynamic_extraction=False,
        calibration=True,
        event_triggered=False,
        fixed_periodic=True,
    ),
    Strategy("dynamic_no_cal", dynamic_extraction=True, calibration=False),
    Strategy("proposed_dynamic_event_cal", dynamic_extraction=True, calibration=True),
)
