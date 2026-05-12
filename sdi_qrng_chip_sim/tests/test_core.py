from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from sdi_qrng_chip_sim.config import STRATEGIES, SimulationConfig
from sdi_qrng_chip_sim.calibration import update_mu_control
from sdi_qrng_chip_sim.entropy_estimator import bilinear_lookup, compute_witness, estimate_energy_upper
from sdi_qrng_chip_sim.extractor_control import toeplitz_hash
from sdi_qrng_chip_sim.lut_validation import load_and_validate_lut, validate_lut_arrays
from sdi_qrng_chip_sim.simulation import run_strategy


STRATEGY_BY_NAME = {strategy.name: strategy for strategy in STRATEGIES}


class CoreSecuritySemanticsTest(unittest.TestCase):
    def test_witness_is_directional(self) -> None:
        good = np.array([[0.98, 0.02], [0.94, 0.06]])
        reversed_inputs = np.array([[0.92, 0.08], [0.96, 0.04]])

        self.assertAlmostEqual(compute_witness(good), 0.04)
        self.assertEqual(compute_witness(reversed_inputs), 0.0)

    def test_energy_upper_uses_one_sided_gaussian_mean_bound(self) -> None:
        cfg = replace(
            SimulationConfig(),
            eps_omega=1e-10,
            sigma_mon=0.002,
            num_inputs=2,
            tap_ratio_error_bound=0.0,
            monitor_pd_nonlinearity_bound=0.0,
            monitor_temp_coeff=0.0,
        )
        xs = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int16)
        monitor_values = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2])

        omega_high = estimate_energy_upper(xs, monitor_values, cfg)
        expected_delta = cfg.sigma_mon * math.sqrt(2.0 * math.log(1.0 / cfg.eps_omega) / 4)
        self.assertAlmostEqual(omega_high, 0.2 + expected_delta)

    def test_toeplitz_hash_reference(self) -> None:
        raw = np.array([1, 0, 1], dtype=np.uint8)
        seed = np.array([1, 1, 0, 1], dtype=np.uint8)
        # Matrix rows are [1, 0, 1] and [1, 1, 0] under this implementation.
        out = toeplitz_hash(raw, seed, 2)
        np.testing.assert_array_equal(out, np.array([0, 1], dtype=np.uint8))

    def test_lut_validation_rejects_nonmonotone_grid(self) -> None:
        W_grid = np.array([0.0, 0.1])
        omega_grid = np.array([0.0, 0.2])
        bad_grid = np.array([[0.0, 0.1], [0.2, 0.3]])
        with self.assertRaises(ValueError):
            validate_lut_arrays(W_grid, omega_grid, bad_grid)

    def test_lut_round_trip_validation(self) -> None:
        W_grid = np.array([0.0, 0.1])
        omega_grid = np.array([0.0, 0.2])
        h_grid = np.array([[0.0, 0.0], [0.1, 0.05]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lut.npz"
            np.savez(path, W_grid=W_grid, omega_grid=omega_grid, h_grid=h_grid)
            loaded_W, loaded_omega, loaded_h, _ = load_and_validate_lut(path)
        np.testing.assert_array_equal(loaded_W, W_grid)
        np.testing.assert_array_equal(loaded_omega, omega_grid)
        np.testing.assert_array_equal(loaded_h, h_grid)

    def test_lut_boundary_interpolation_clips_to_grid(self) -> None:
        W_grid = np.array([0.0, 0.1])
        omega_grid = np.array([0.0, 0.2])
        h_grid = np.array([[0.0, 0.0], [0.2, 0.1]])

        self.assertAlmostEqual(bilinear_lookup(-1.0, 0.1, W_grid, omega_grid, h_grid), 0.0)
        self.assertAlmostEqual(bilinear_lookup(1.0, 0.0, W_grid, omega_grid, h_grid), 0.2)
        self.assertAlmostEqual(bilinear_lookup(0.05, 0.1, W_grid, omega_grid, h_grid), 0.075)

    def test_calibration_control_reduces_high_energy(self) -> None:
        cfg = SimulationConfig()
        mu_values = np.array([0.05, 0.35], dtype=np.float64)
        current = mu_values
        for _ in range(10):
            current = update_mu_control(current, omega_high=float(np.max(current)), h_min=0.02, cfg=cfg)
        self.assertLess(current[1], mu_values[1])
        self.assertLess(abs(float(np.max(current)) - cfg.mu_target), abs(float(np.max(mu_values)) - cfg.mu_target))

    def test_invalid_config_raises(self) -> None:
        cfg = replace(SimulationConfig(), block_size=0)
        with self.assertRaises(ValueError):
            run_strategy(cfg, STRATEGY_BY_NAME["dynamic_no_cal"])

    def test_seed_starvation_blocks_output(self) -> None:
        class ConstantEntropy:
            def __call__(self, W_low: float, omega_high: float) -> float:
                return 0.2

        cfg = replace(
            SimulationConfig(),
            num_blocks=1,
            block_size=1000,
            seed_pool_initial_bits=0,
        )
        _, summary = run_strategy(cfg, STRATEGY_BY_NAME["dynamic_no_cal"], entropy_curve=ConstantEntropy())
        self.assertEqual(summary["emitted_bits"], 0)
        self.assertEqual(summary["certified_bits"], 0)
        self.assertEqual(summary["seed_starved_rate"], 1.0)

    def test_stop_semantics_distinguish_emitted_and_certified_bits(self) -> None:
        cfg = replace(
            SimulationConfig(),
            num_blocks=2,
            block_size=1000,
            h_warn=1.0,
            h_normal=2.0,
            fixed_extractor_rate=0.01,
        )
        fixed_records, fixed_summary = run_strategy(cfg, STRATEGY_BY_NAME["baseline_fixed_no_cal"])
        dynamic_records, dynamic_summary = run_strategy(cfg, STRATEGY_BY_NAME["dynamic_no_cal"])

        self.assertEqual(fixed_summary["stop_rate"], 1.0)
        self.assertGreater(fixed_summary["emitted_bits"], 0)
        self.assertEqual(fixed_summary["certified_bits"], 0)
        self.assertGreater(fixed_summary["unsafe_emitted_bits"], 0)
        self.assertTrue(all(record["unsafe_output"] for record in fixed_records))

        self.assertEqual(dynamic_summary["stop_rate"], 1.0)
        self.assertEqual(dynamic_summary["emitted_bits"], 0)
        self.assertEqual(dynamic_summary["certified_bits"], 0)
        self.assertTrue(all(not record["unsafe_output"] for record in dynamic_records))


if __name__ == "__main__":
    unittest.main()
