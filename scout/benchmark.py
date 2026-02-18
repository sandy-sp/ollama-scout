"""
benchmark.py - Dry-run inference speed estimator based on hardware profile.
"""
from dataclasses import dataclass

from .hardware import HardwareProfile
from .recommender import Recommendation


@dataclass
class BenchmarkEstimate:
    model_name: str
    run_mode: str
    tokens_per_sec: float
    rating: str


def estimate_speed(rec: Recommendation, hw: HardwareProfile) -> BenchmarkEstimate:
    """Estimate inference speed for a model based on hardware and run mode."""
    size = rec.variant.size_gb
    if size <= 0:
        size = 1.0  # avoid division by zero

    vram = hw.best_vram_gb
    ram = hw.ram_gb
    threads = hw.cpu_threads

    mode = rec.run_mode

    if mode == "GPU":
        tps = (vram / size) * 40
        tps = min(tps, 120.0)
    elif mode == "CPU+GPU":
        total = vram + ram
        tps = (total / size) * 12
        tps = min(tps, 45.0)
    else:  # CPU
        tps = (threads / size) * 4
        tps = min(tps, 20.0)

    tps = round(tps, 1)

    if tps >= 60:
        rating = "Fast"
    elif tps >= 25:
        rating = "Moderate"
    else:
        rating = "Slow"

    return BenchmarkEstimate(
        model_name=f"{rec.model.name}:{rec.variant.tag}",
        run_mode=mode,
        tokens_per_sec=tps,
        rating=rating,
    )


def get_benchmark_estimates(
    recs: list[Recommendation],
    hw: HardwareProfile,
    top_n: int = 3,
) -> list[BenchmarkEstimate]:
    """Estimate speeds for the top N recommendations."""
    estimates = []
    for rec in recs[:top_n]:
        estimates.append(estimate_speed(rec, hw))
    return estimates
