#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m sdi_qrng_chip_sim.lut_generation.generate_lut
"${PYTHON_BIN}" -m unittest discover sdi_qrng_chip_sim/tests
"${PYTHON_BIN}" -m sdi_qrng_chip_sim.main
"${PYTHON_BIN}" -m sdi_qrng_chip_sim.experiments --num-blocks 80 --block-size 5000
"${PYTHON_BIN}" -m sdi_qrng_chip_sim.patent_figures --replicates 10

