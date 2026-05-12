from __future__ import annotations

from pathlib import Path

import numpy as np


def validate_lut_arrays(
    W_grid: np.ndarray,
    omega_grid: np.ndarray,
    h_grid: np.ndarray,
    *,
    monotonic_tolerance: float = 1e-12,
) -> None:
    if W_grid.ndim != 1 or omega_grid.ndim != 1:
        raise ValueError("LUT grids must be one-dimensional")
    if W_grid.size < 2 or omega_grid.size < 2:
        raise ValueError("LUT grids need at least two points on each axis")
    if not np.all(np.diff(W_grid) > 0.0):
        raise ValueError("W_grid must be strictly increasing")
    if not np.all(np.diff(omega_grid) > 0.0):
        raise ValueError("omega_grid must be strictly increasing")
    if h_grid.shape != (W_grid.size, omega_grid.size):
        raise ValueError(
            f"h_grid shape {h_grid.shape} does not match ({W_grid.size}, {omega_grid.size})"
        )
    if not np.all(np.isfinite(h_grid)):
        raise ValueError("h_grid contains NaN or infinite values")
    if float(np.min(h_grid)) < -monotonic_tolerance:
        raise ValueError("h_grid contains negative entropy values")

    if np.any(np.diff(h_grid, axis=0) < -monotonic_tolerance):
        raise ValueError("h_grid must be nondecreasing in W")
    if np.any(np.diff(h_grid, axis=1) > monotonic_tolerance):
        raise ValueError("h_grid must be nonincreasing in omega")


def load_and_validate_lut(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    data = np.load(path)
    W_grid = data["W_grid"].astype(np.float64)
    omega_grid = data["omega_grid"].astype(np.float64)
    h_grid = data["h_grid"].astype(np.float64)
    validate_lut_arrays(W_grid, omega_grid, h_grid)

    metadata: dict[str, str] = {}
    if "metadata" in data:
        metadata["description"] = str(data["metadata"])
    if "backend" in data:
        metadata["backend"] = str(data["backend"])
    return W_grid, omega_grid, h_grid, metadata
