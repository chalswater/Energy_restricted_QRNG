from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig, Strategy


@dataclass(frozen=True)
class OutputDecision:
    emitted_bits: int
    certified_bits: int
    safe_bits: int
    unsafe_output: bool


def compute_safe_output_length(block_size: int, h_min: float, cfg: SimulationConfig) -> int:
    H = block_size * h_min
    penalty = 2.0 * math.log2(1.0 / cfg.eps_ext) + cfg.lambda_margin
    return max(0, int(math.floor(H - penalty)))


def compute_strategy_output_length(
    block_size: int,
    h_min: float,
    cfg: SimulationConfig,
    strategy: Strategy,
) -> OutputDecision:
    safe_m = compute_safe_output_length(block_size, h_min, cfg)
    if strategy.dynamic_extraction:
        return OutputDecision(
            emitted_bits=safe_m,
            certified_bits=safe_m,
            safe_bits=safe_m,
            unsafe_output=False,
        )

    fixed_m = max(0, int(math.floor(block_size * cfg.fixed_extractor_rate)))
    unsafe = fixed_m > safe_m
    certified_m = fixed_m if not unsafe else 0
    return OutputDecision(
        emitted_bits=fixed_m,
        certified_bits=certified_m,
        safe_bits=safe_m,
        unsafe_output=unsafe,
    )


def choose_mode(h_min: float, cfg: SimulationConfig) -> str:
    if h_min >= cfg.h_normal:
        return "NORMAL"
    if h_min >= cfg.h_warn:
        return "PROTECTION"
    return "STOP"


def toeplitz_hash(raw_bits: np.ndarray, seed_bits: np.ndarray, output_len: int) -> np.ndarray:
    """Reference Toeplitz extractor for small blocks.

    seed_bits length must be at least len(raw_bits) + output_len - 1.
    The system simulation normally computes m_k only; hardware studies can call
    this function on reduced block sizes.
    """
    n = int(raw_bits.size)
    m = int(output_len)
    if m <= 0:
        return np.zeros(0, dtype=np.uint8)
    if seed_bits.size < n + m - 1:
        raise ValueError("Toeplitz seed is too short")

    raw = raw_bits.astype(np.uint8)
    seed = seed_bits.astype(np.uint8)
    out = np.zeros(m, dtype=np.uint8)
    for row in range(m):
        coeffs = seed[m - 1 - row : m - 1 - row + n]
        out[row] = int(np.bitwise_xor.reduce(coeffs & raw))
    return out
