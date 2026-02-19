"""
benchmark.py - Inference speed estimator (formula-based and real timing).
"""
import shutil
import subprocess
import time
from dataclasses import dataclass, field

from .hardware import HardwareProfile
from .recommender import Recommendation


@dataclass
class BenchmarkEstimate:
    model_name: str
    run_mode: str
    tokens_per_sec: float
    rating: str
    is_real: bool = field(default=False)


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


def run_real_benchmark(model_name: str, prompt: str = "Hello, how are you?") -> float | None:
    """Run a real benchmark by invoking `ollama run` and measuring wall-clock time.

    Returns estimated tokens_per_sec, or None if ollama is unavailable,
    the model isn't pulled, or the run fails/times out.
    """
    if not shutil.which("ollama"):
        return None

    # Check if model is pulled
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
        # ollama list output has model names in first column
        pulled_names = []
        for line in result.stdout.strip().splitlines()[1:]:  # skip header
            parts = line.split()
            if parts:
                pulled_names.append(parts[0])
        # Match with or without tag (e.g. "llama3.2:latest" or "llama3.2")
        base_name = model_name.split(":")[0]
        if not any(base_name == p.split(":")[0] for p in pulled_names):
            return None
    except (subprocess.TimeoutExpired, OSError):
        return None

    # Run the model and measure time
    try:
        start = time.monotonic()
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            capture_output=True, text=True, timeout=60,
        )
        elapsed = time.monotonic() - start

        if result.returncode != 0 or not result.stdout.strip():
            return None

        # Estimate tokens: ~0.75 tokens per word is a rough average
        words = len(result.stdout.split())
        estimated_tokens = max(int(words * 0.75), 1)
        tps = round(estimated_tokens / max(elapsed, 0.1), 1)
        return tps
    except (subprocess.TimeoutExpired, OSError):
        return None


def benchmark_pulled_models(
    pulled_models: list[str],
    hw: HardwareProfile,
) -> list[BenchmarkEstimate]:
    """Run real benchmarks on already-pulled models.

    Returns BenchmarkEstimate list with is_real=True.
    """
    estimates = []
    for model_name in pulled_models:
        tps = run_real_benchmark(model_name)
        if tps is None:
            continue

        if tps >= 60:
            rating = "Fast"
        elif tps >= 25:
            rating = "Moderate"
        else:
            rating = "Slow"

        # Determine run mode based on hardware
        run_mode = "GPU" if hw.best_vram_gb > 0 else "CPU"

        estimates.append(BenchmarkEstimate(
            model_name=model_name,
            run_mode=run_mode,
            tokens_per_sec=tps,
            rating=rating,
            is_real=True,
        ))
    return estimates
