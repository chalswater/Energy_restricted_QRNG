from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..lut_validation import validate_lut_arrays
from .sdp_solver import ENERGY_RELAXATION_BACKEND, generate_h_min_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an SDI h_min LUT.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "lut_data.npz")
    parser.add_argument("--W-points", type=int, default=81)
    parser.add_argument("--omega-points", type=int, default=81)
    parser.add_argument("--backend", default=ENERGY_RELAXATION_BACKEND)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    W_grid = np.linspace(0.0, 0.16, args.W_points)
    omega_grid = np.linspace(0.0, 0.6, args.omega_points)
    grid = generate_h_min_grid(W_grid, omega_grid, args.backend)
    validate_lut_arrays(W_grid, omega_grid, grid)
    metadata = np.array(
        "two-intensity click protocol; runtime LUT maps (W_low, omega_high) to h_min",
        dtype="U128",
    )
    backend = np.array(args.backend, dtype="U64")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, W_grid=W_grid, omega_grid=omega_grid, h_grid=grid, metadata=metadata, backend=backend)
    print(args.output)


if __name__ == "__main__":
    main()
