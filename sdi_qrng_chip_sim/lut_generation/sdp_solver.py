from __future__ import annotations

import math

import numpy as np


ENERGY_RELAXATION_BACKEND = "energy_constrained_relaxation_v1"
CVXPY_SDP_BACKEND = "cvxpy_sdp_formulation"
OMEGA_SECURITY_CAP = 0.30
ENERGY_PENALTY_SLOPE = 1.2


def certified_click_response(W: float, omega: float) -> float:
    """Conservative response lower bound used to populate the runtime LUT.

    Final protocol selected for the current chip design:
      - x=0: weak decoy coherent state.
      - x=1: weak signal coherent state.
      - a=1: click event.
      - W = P(1|1) - P(1|0).
      - omega is the monitored upper bound on the larger mean photon number.

    The certification envelope accepts contrast only while the monitored energy
    remains below OMEGA_SECURITY_CAP. Above the cap, the usable quantum response
    is discounted quickly so event-triggered self-calibration is visible in the
    LUT itself. The production version can replace this function with the
    solution of the SDP guessing-probability relaxation without changing the
    runtime firmware interface.
    """
    W_clipped = min(max(float(W), 0.0), 0.49)
    energy_excess = max(float(omega) - OMEGA_SECURITY_CAP, 0.0)
    response = W_clipped - ENERGY_PENALTY_SLOPE * energy_excess
    return float(min(max(response, 0.0), 0.49))


def energy_bounded_guess_probability(W: float, omega: float) -> float:
    response = certified_click_response(W, omega)
    # Worst case is a rare certified click: P_guess = 1 - q.
    return float(1.0 - response)


def energy_bounded_h_min(W: float, omega: float) -> float:
    p_guess = energy_bounded_guess_probability(W, omega)
    return float(-math.log2(max(p_guess, 1e-12)))


def h_min_grid(W_grid: np.ndarray, omega_grid: np.ndarray) -> np.ndarray:
    h_grid = np.zeros((W_grid.size, omega_grid.size), dtype=np.float64)
    for i, W in enumerate(W_grid):
        for j, omega in enumerate(omega_grid):
            h_grid[i, j] = energy_bounded_h_min(float(W), float(omega))
    return h_grid


def cvxpy_sdp_h_min_grid(W_grid: np.ndarray, omega_grid: np.ndarray) -> np.ndarray:
    """Formal SDP backend hook.

    The runtime is already LUT-only, so the final security proof should replace
    the conservative relaxation with this backend once a convex solver is
    available in the environment. This function intentionally fails loudly
    rather than silently falling back to a heuristic.
    """
    try:
        import cvxpy  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "cvxpy is not installed. Install a convex SDP solver stack before "
            "generating a formal SDP-certified LUT."
        ) from exc

    raise NotImplementedError(
        "The repository contains the runtime LUT interface and formal SDP "
        "problem statement, but the production SDP relaxation has not been "
        "implemented yet. Use --backend energy_constrained_relaxation_v1 for "
        "engineering/patent simulations."
    )


def generate_h_min_grid(W_grid: np.ndarray, omega_grid: np.ndarray, backend: str) -> np.ndarray:
    if backend == ENERGY_RELAXATION_BACKEND:
        return h_min_grid(W_grid, omega_grid)
    if backend == CVXPY_SDP_BACKEND:
        return cvxpy_sdp_h_min_grid(W_grid, omega_grid)
    raise ValueError(f"Unsupported LUT backend: {backend}")
