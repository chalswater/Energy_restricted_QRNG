# SDI QRNG Chip Simulation MVP

This directory implements a runnable block-level simulation for:

```text
on-chip monitoring
  -> conservative SDI entropy estimate
  -> dynamic extractor output length
  -> health modes and calibration events
```

Runtime entropy estimation is LUT-based. The default file is:

```text
lut_generation/lut_data.npz
```

The default LUT is generated from the selected two-intensity click protocol and
a conservative energy-constrained relaxation backend:

```text
energy_constrained_relaxation_v1
```

The runtime code does not use the old linear placeholder. A formal SDP hook
(`cvxpy_sdp_formulation`) is present and fails loudly unless a solver stack and
the final optical relaxation are implemented, so engineering figures cannot be
mistaken for a completed proof.

## Engineering Impairments

The block-level model includes product-facing nonidealities:

```text
tap coupler ratio error
monitor photodiode nonlinearity
modulator extinction ratio and bias drift
detector dead time and afterpulsing
TDC bin nonuniformity
temperature drift and temperature-dependent source/detector response
calibration latency
Toeplitz seed pool consumption and refresh
```

These are controlled by `SimulationConfig` and recorded in per-block metadata.

## Run

Use a Python environment with `numpy` and `Pillow`.

```bash
python3 -m sdi_qrng_chip_sim.main
```

In Codex desktop, the bundled runtime has the required packages:

```bash
/Users/sichen/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m sdi_qrng_chip_sim.main
```

Outputs are written to:

```text
sdi_qrng_chip_sim/figures/
```

Key files:

- `sdi_qrng_chip_summary.json`
- `sdi_qrng_chip_records.json`
- `h_min_vs_block.png`
- `omega_high_vs_block.png`
- `certified_rate_vs_block.png`
- `stop_rate_comparison.png`
- `unsafe_block_rate_comparison.png`

## Compared Strategies

```text
baseline_fixed_no_cal
baseline_fixed_periodic_cal
dynamic_no_cal
proposed_dynamic_event_cal
```

For fixed-output baselines, the simulator records emitted bits and separately
tracks whether the chosen fixed output length exceeds the conservative safe
length for that block.

## Next Upgrade

Replace `lut_generation/sdp_solver.py` with the selected SDI relaxation and run:

```bash
python3 -m sdi_qrng_chip_sim.lut_generation.generate_lut
```

Then pass the resulting LUT into `EntropyCurve` from the control firmware or
system simulator.

## Parameter Scans

Run dark-count and detector-efficiency scans for the proposed strategy:

```bash
python3 -m sdi_qrng_chip_sim.experiments
```

This writes `sdi_qrng_chip_scans.json` and scan figures to the same output
directory.

## Patent Figures

Generate the three patent-oriented embodiment figures:

```bash
python3 -m sdi_qrng_chip_sim.patent_figures
```

This writes:

```text
figures/patent_figures/patent_fig_1_drift_fixed_vs_dynamic.png
figures/patent_figures/patent_fig_2_anomaly_event_calibration.png
figures/patent_figures/patent_fig_3_unsafe_block_safety.png
figures/patent_figures/patent_figure_report_zh.md
figures/patent_figures/patent_figure_summary.json
figures/patent_figures/patent_figure_manifest.json
```

The patent summary includes 10-seed confidence statistics by default. The
manifest records seeds, LUT hash, source hashes, and generated file paths.

## Tests

```bash
python3 -m unittest discover sdi_qrng_chip_sim/tests
```

## Release Artifacts

```text
requirements.txt
.github/workflows/ci.yml
scripts/reproduce_sdi_qrng_chip_sim.sh
sdi_qrng_chip_sim/schema/simulation_summary.schema.json
sdi_qrng_chip_sim/schema/patent_figure_summary.schema.json
```
