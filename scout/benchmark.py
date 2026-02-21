"""
benchmark.py - Real benchmark timing via `ollama run`.
"""
import shutil
import subprocess
import time
from dataclasses import dataclass

from .hardware import HardwareProfile


@dataclass
class BenchmarkEstimate:
    model_name: str
    run_mode: str
    tokens_per_sec: float
    rating: str


def benchmark_model(model_name: str, prompt: str = "Hello, how are you?") -> float | None:
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
        pulled_names = []
        for line in result.stdout.strip().splitlines()[1:]:  # skip header
            parts = line.split()
            if parts:
                pulled_names.append(parts[0])
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

    Returns a BenchmarkEstimate list.
    """
    estimates = []
    for model_name in pulled_models:
        tps = benchmark_model(model_name)
        if tps is None:
            continue

        if tps >= 60:
            rating = "Fast"
        elif tps >= 25:
            rating = "Moderate"
        else:
            rating = "Slow"

        run_mode = "GPU" if hw.best_vram_gb > 0 else "CPU"

        estimates.append(BenchmarkEstimate(
            model_name=model_name,
            run_mode=run_mode,
            tokens_per_sec=tps,
            rating=rating,
        ))
    return estimates
