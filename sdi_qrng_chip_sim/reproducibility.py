from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .config import ROOT, SimulationConfig


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    return hashes


def metric_stats(values: Iterable[float]) -> dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "ci95_half_width": 0.0}
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    ci95 = 1.96 * std / (len(vals) ** 0.5) if len(vals) > 1 else 0.0
    return {"mean": mean, "std": std, "ci95_half_width": ci95}


def write_manifest(
    path: str | Path,
    *,
    cfg: SimulationConfig,
    generated_files: list[str],
    seeds: list[int],
    command: str,
) -> None:
    manifest = {
        "command": command,
        "seeds": seeds,
        "config": {**asdict(cfg), "mu_initial": list(cfg.mu_initial)},
        "lut_sha256": sha256_file(cfg.entropy_lut_path),
        "source_hashes": source_hashes(),
        "generated_files": generated_files,
    }
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
