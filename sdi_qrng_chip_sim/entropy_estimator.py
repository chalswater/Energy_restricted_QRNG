from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SimulationConfig
from .lut_validation import load_and_validate_lut


@dataclass(frozen=True)
class EntropyEstimate:
    conditional_p: np.ndarray
    W_hat: float
    W_low: float
    omega_high: float
    h_min: float


class EntropyCurve:
    """Runtime entropy curve.

    The chip runtime path is intentionally LUT-only: the expensive security
    calculation is performed offline and stored as h_min(W, omega).
    """

    def __init__(self, cfg: SimulationConfig, lut_path: str | Path | None = None) -> None:
        self.cfg = cfg
        self.W_grid: np.ndarray
        self.omega_grid: np.ndarray
        self.h_grid: np.ndarray
        self.metadata: dict[str, str] = {}

        resolved_lut_path = Path(lut_path or cfg.entropy_lut_path)
        if not resolved_lut_path.exists():
            raise FileNotFoundError(
                f"Entropy LUT not found at {resolved_lut_path}. "
                "Run: python3 -m sdi_qrng_chip_sim.lut_generation.generate_lut"
            )
        self.W_grid, self.omega_grid, self.h_grid, self.metadata = load_and_validate_lut(resolved_lut_path)

    def __call__(self, W_low: float, omega_high: float) -> float:
        return float(max(0.0, bilinear_lookup(W_low, omega_high, self.W_grid, self.omega_grid, self.h_grid)))


def estimate_conditional_distribution(
    xs: np.ndarray,
    outcomes: np.ndarray,
    num_inputs: int,
) -> np.ndarray:
    P = np.zeros((num_inputs, 2), dtype=np.float64)
    for x in range(num_inputs):
        mask = xs == x
        count_x = int(np.sum(mask))
        if count_x == 0:
            P[x, :] = 0.5
        else:
            p1 = float(np.mean(outcomes[mask]))
            P[x, 1] = p1
            P[x, 0] = 1.0 - p1
    return P


def compute_witness(P: np.ndarray) -> float:
    """Protocol witness W = P(click | high-energy input) - P(click | decoy)."""
    if P.shape[0] < 2:
        return 0.0
    return max(0.0, float(P[1, 1] - P[0, 1]))


def finite_size_correction_W(block_size: int, eps_W: float, C_W: float) -> float:
    return float(C_W * np.sqrt(np.log(1.0 / eps_W) / (2.0 * block_size)))


def estimate_energy_upper(
    xs: np.ndarray,
    monitor_values: np.ndarray,
    cfg: SimulationConfig,
) -> float:
    highs: list[float] = []
    for x in range(cfg.num_inputs):
        vals = monitor_values[xs == x]
        if vals.size == 0:
            continue
        mean_mu = float(np.mean(vals))
        # Conservative mean bound for Gaussian monitor noise. The real product
        # version should replace this with a calibrated monitor error model.
        statistical_delta = cfg.sigma_mon * np.sqrt(2.0 * np.log(1.0 / cfg.eps_omega) / vals.size)
        systematic_delta = (
            abs(mean_mu) * cfg.tap_ratio_error_bound
            + cfg.monitor_pd_nonlinearity_bound * mean_mu * mean_mu * cfg.tap_nominal_ratio
            + abs(mean_mu) * abs(cfg.monitor_temp_coeff) * 5.0
        )
        highs.append(mean_mu + float(statistical_delta + systematic_delta))
    if not highs:
        return 0.0
    return float(max(highs))


def estimate_block_entropy(
    xs: np.ndarray,
    outcomes: np.ndarray,
    monitor_values: np.ndarray,
    cfg: SimulationConfig,
    entropy_curve: EntropyCurve,
) -> EntropyEstimate:
    P = estimate_conditional_distribution(xs, outcomes, cfg.num_inputs)
    W_hat = compute_witness(P)
    delta_W = finite_size_correction_W(cfg.block_size, cfg.eps_W, cfg.witness_correction_scale)
    W_low = max(0.0, W_hat - delta_W)
    omega_high = estimate_energy_upper(xs, monitor_values, cfg)
    h_min = entropy_curve(W_low, omega_high)
    return EntropyEstimate(P, W_hat, W_low, omega_high, h_min)


def bilinear_lookup(
    W: float,
    omega: float,
    W_grid: np.ndarray,
    omega_grid: np.ndarray,
    h_grid: np.ndarray,
) -> float:
    Wc = float(np.clip(W, W_grid[0], W_grid[-1]))
    oc = float(np.clip(omega, omega_grid[0], omega_grid[-1]))

    i = int(np.searchsorted(W_grid, Wc, side="right") - 1)
    j = int(np.searchsorted(omega_grid, oc, side="right") - 1)
    i = min(max(i, 0), len(W_grid) - 2)
    j = min(max(j, 0), len(omega_grid) - 2)

    W0, W1 = float(W_grid[i]), float(W_grid[i + 1])
    o0, o1 = float(omega_grid[j]), float(omega_grid[j + 1])
    t = 0.0 if W1 == W0 else (Wc - W0) / (W1 - W0)
    u = 0.0 if o1 == o0 else (oc - o0) / (o1 - o0)

    h00 = float(h_grid[i, j])
    h10 = float(h_grid[i + 1, j])
    h01 = float(h_grid[i, j + 1])
    h11 = float(h_grid[i + 1, j + 1])
    return (1 - t) * (1 - u) * h00 + t * (1 - u) * h10 + (1 - t) * u * h01 + t * u * h11
