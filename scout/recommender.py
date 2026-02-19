"""
recommender.py - Match hardware profile to compatible Ollama models/variants.
"""
from dataclasses import dataclass

from .hardware import HardwareProfile
from .ollama_api import ModelVariant, OllamaModel


@dataclass
class Recommendation:
    model: OllamaModel
    variant: ModelVariant
    score: int           # higher = better fit
    run_mode: str        # "GPU", "CPU", "CPU+GPU"
    fit_label: str       # "Excellent", "Good", "Possible"
    note: str = ""


def _score_variant(
    variant: ModelVariant,
    hw: HardwareProfile,
) -> tuple[int, str, str, str]:
    """
    Returns (score, fit_label, run_mode, note).
    Logic:
    - Apple Silicon unified memory: treat total RAM as VRAM (GPU acceleration via Metal)
    - If VRAM >= model size → full GPU (best)
    - If VRAM < model size but RAM + VRAM >= model size → partial offload (ok)
    - If no GPU but RAM >= model size → CPU only (slow but possible)
    - Otherwise → not recommended
    """
    size = variant.size_gb
    vram = hw.best_vram_gb
    ram = hw.ram_gb

    if size == 0:
        return 0, "Unknown", "?", "Size unknown"

    # Apple Silicon unified memory: VRAM and RAM are the same pool
    if hw.is_unified_memory:
        usable = max(ram - 4.0, 0)  # reserve 4GB for macOS + apps
        if usable >= size:
            score = 100 - int(size)
            return score, "Excellent", "GPU", f"Fits in unified memory ({ram}GB total)"
        if usable >= size * 0.7:
            score = 60 - int((size - usable) * 5)
            return max(score, 20), "Good", "GPU", "Tight fit in unified memory, may swap"
        return (
            -1, "Too Large", "N/A",
            f"Needs ~{size}GB, unified memory: "
            f"{ram}GB (usable: {usable:.0f}GB)",
        )

    # Discrete GPU path
    usable_ram = max(ram - 2.0, 0)

    if vram >= size:
        score = 100 - int(size)
        return score, "Excellent", "GPU", f"Fits fully in VRAM ({vram}GB)"

    if vram > 0 and (vram + usable_ram) >= size:
        offload_gb = size - vram
        score = 60 - int(offload_gb * 5)
        return max(score, 20), "Good", "CPU+GPU", f"~{offload_gb:.1f}GB offloaded to RAM"

    if usable_ram >= size:
        score = 40 - int(size * 2)
        tps = max((hw.cpu_threads / size) * 4, 0.1)
        time_sec = round(200 / tps)
        if time_sec <= 10:
            note = "CPU-only (fast enough)"
        elif time_sec > 60:
            minutes = round(time_sec / 60)
            note = f"CPU-only (~{minutes}m, consider a smaller model)"
        else:
            note = f"CPU-only (~{time_sec}s for 200 tokens)"
        return max(score, 5), "Possible", "CPU", note

    return (
        -1, "Too Large", "N/A",
        f"Needs ~{size}GB, available: "
        f"{vram}GB VRAM / {usable_ram:.0f}GB RAM",
    )


def get_recommendations(
    models: list[OllamaModel],
    hw: HardwareProfile,
    use_case_filter: str = "all",
    pulled_models: list[str] = None,
    top_n: int = 15,
) -> list[Recommendation]:
    pulled_models = pulled_models or []
    recs: list[Recommendation] = []

    for model in models:
        # Filter by use case
        if use_case_filter != "all" and use_case_filter not in model.use_cases:
            continue

        # Mark if already pulled
        model.pulled = model.name in pulled_models

        # Find best compatible variant
        best: tuple[int, ModelVariant, str, str, str] | None = None
        for variant in model.tags:
            score, fit_label, run_mode, note = _score_variant(variant, hw)
            if score < 0:
                continue
            if best is None or score > best[0]:
                best = (score, variant, fit_label, run_mode, note)

        if best:
            score, variant, fit_label, run_mode, note = best
            # Boost score if already pulled
            if model.pulled:
                score += 10
            recs.append(Recommendation(
                model=model,
                variant=variant,
                score=score,
                run_mode=run_mode,
                fit_label=fit_label,
                note=note,
            ))

    recs.sort(key=lambda r: r.score, reverse=True)
    return recs[:top_n]


def group_by_use_case(recs: list[Recommendation]) -> dict[str, list[Recommendation]]:
    groups: dict[str, list[Recommendation]] = {
        "coding": [],
        "reasoning": [],
        "chat": [],
    }
    for rec in recs:
        for uc in rec.model.use_cases:
            if uc in groups:
                groups[uc].append(rec)
    # Dedupe within each group (keep top 5)
    for key in groups:
        seen = set()
        deduped = []
        for r in groups[key]:
            if r.model.name not in seen:
                seen.add(r.model.name)
                deduped.append(r)
        groups[key] = deduped[:5]
    return groups
