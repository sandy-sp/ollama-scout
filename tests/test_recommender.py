"""Tests for scout.recommender module."""
from scout.hardware import GPUInfo, HardwareProfile
from scout.ollama_api import ModelVariant, OllamaModel
from scout.recommender import _score_variant, get_recommendations, group_by_use_case


def _make_hw(vram_gb=10.0, ram_gb=32.0, unified=False):
    gpus = [GPUInfo(name="Test GPU", vram_mb=int(vram_gb * 1024))] if vram_gb > 0 else []
    return HardwareProfile(
        os="Linux",
        cpu_name="Test CPU",
        cpu_cores=8,
        cpu_threads=16,
        ram_gb=ram_gb,
        gpus=gpus,
        is_unified_memory=unified,
    )


def _make_model(name, param_size, size_gb, use_cases=None):
    variant = ModelVariant(
        tag=f"{param_size.lower()}",
        size_gb=size_gb,
        quantization="Q4_K_M",
        param_size=param_size,
    )
    return OllamaModel(
        name=name,
        description=f"Test {name}",
        tags=[variant],
        use_cases=use_cases or ["chat"],
    )


class TestScoreVariant:
    def test_excellent_fit_when_vram_sufficient(self):
        hw = _make_hw(vram_gb=10.0, ram_gb=32.0)
        variant = ModelVariant(tag="3b", size_gb=2.0, quantization="Q4_K_M", param_size="3B")
        score, fit, mode, note = _score_variant(variant, hw)
        assert fit == "Excellent"
        assert mode == "GPU"
        assert score > 0

    def test_good_fit_with_cpu_offload(self):
        hw = _make_hw(vram_gb=6.0, ram_gb=32.0)
        variant = ModelVariant(tag="13b", size_gb=8.0, quantization="Q4_K_M", param_size="13B")
        score, fit, mode, note = _score_variant(variant, hw)
        assert fit == "Good"
        assert mode == "CPU+GPU"

    def test_too_large_when_insufficient(self):
        hw = _make_hw(vram_gb=10.0, ram_gb=32.0)
        variant = ModelVariant(tag="70b", size_gb=80.0, quantization="Q4_K_M", param_size="70B")
        score, fit, mode, note = _score_variant(variant, hw)
        assert fit == "Too Large"
        assert score < 0

    def test_cpu_only_note_has_time_estimate(self):
        hw = _make_hw(vram_gb=0, ram_gb=32.0)
        variant = ModelVariant(tag="7b", size_gb=4.0, quantization="Q4_K_M", param_size="7B")
        score, fit, mode, note = _score_variant(variant, hw)
        assert fit == "Possible"
        assert mode == "CPU"
        assert "CPU-only" in note
        # Should have a time estimate, not the old generic message
        assert "may be slow" not in note

    def test_cpu_only_fast_enough_for_tiny_model(self):
        hw_big = HardwareProfile(
            os="Linux", cpu_name="Test", cpu_cores=16,
            cpu_threads=32, ram_gb=32.0, gpus=[],
        )
        variant = ModelVariant(tag="1b", size_gb=0.7, quantization="Q4_K_M", param_size="1B")
        score, fit, mode, note = _score_variant(variant, hw_big)
        assert "fast enough" in note

    def test_cpu_only_large_model_suggests_smaller(self):
        hw = HardwareProfile(
            os="Linux", cpu_name="Test", cpu_cores=2,
            cpu_threads=4, ram_gb=32.0, gpus=[],
        )
        variant = ModelVariant(
            tag="13b", size_gb=8.0,
            quantization="Q4_K_M", param_size="13B",
        )
        score, fit, mode, note = _score_variant(variant, hw)
        assert "consider a smaller model" in note


class TestMultiGPU:
    def test_multi_gpu_fits_across_two_gpus(self):
        hw = HardwareProfile(
            os="Linux", cpu_name="Test", cpu_cores=8,
            cpu_threads=16, ram_gb=32.0,
            gpus=[
                GPUInfo(name="GPU 0", vram_mb=8 * 1024),
                GPUInfo(name="GPU 1", vram_mb=8 * 1024),
            ],
        )
        # 12GB model: doesn't fit in single 8GB GPU, fits across 16GB combined
        variant = ModelVariant(
            tag="13b", size_gb=12.0,
            quantization="Q4_K_M", param_size="13B",
        )
        score, fit, mode, note = _score_variant(variant, hw)
        assert fit == "Excellent"
        assert mode == "Multi-GPU"
        assert "2 GPUs" in note

    def test_single_gpu_preferred_when_it_fits(self):
        hw = HardwareProfile(
            os="Linux", cpu_name="Test", cpu_cores=8,
            cpu_threads=16, ram_gb=32.0,
            gpus=[
                GPUInfo(name="GPU 0", vram_mb=10 * 1024),
                GPUInfo(name="GPU 1", vram_mb=10 * 1024),
            ],
        )
        # 4GB model fits in single 10GB GPU — should prefer GPU, not Multi-GPU
        variant = ModelVariant(
            tag="7b", size_gb=4.0,
            quantization="Q4_K_M", param_size="7B",
        )
        score, fit, mode, note = _score_variant(variant, hw)
        assert fit == "Excellent"
        assert mode == "GPU"


class TestGetRecommendations:
    def test_excludes_too_large_models(self):
        hw = _make_hw(vram_gb=10.0, ram_gb=32.0)
        models = [
            _make_model("small-model", "3B", 2.0),
            _make_model("medium-model", "7B", 4.0),
            _make_model("huge-model", "70B", 80.0),
        ]
        recs = get_recommendations(models, hw)
        rec_names = [r.model.name for r in recs]
        assert "small-model" in rec_names
        assert "medium-model" in rec_names
        assert "huge-model" not in rec_names

    def test_3b_and_7b_are_excellent(self):
        hw = _make_hw(vram_gb=10.0, ram_gb=32.0)
        models = [
            _make_model("small-model", "3B", 2.0),
            _make_model("medium-model", "7B", 4.0),
        ]
        recs = get_recommendations(models, hw)
        for rec in recs:
            assert rec.fit_label == "Excellent"


class TestGroupByUseCase:
    def test_returns_correct_keys(self):
        hw = _make_hw(vram_gb=10.0, ram_gb=32.0)
        models = [
            _make_model("code-model", "7B", 4.0, use_cases=["coding"]),
            _make_model("chat-model", "7B", 4.0, use_cases=["chat"]),
            _make_model("reason-model", "7B", 4.0, use_cases=["reasoning"]),
        ]
        recs = get_recommendations(models, hw)
        grouped = group_by_use_case(recs)

        assert "coding" in grouped
        assert "chat" in grouped
        assert "reasoning" in grouped

    def test_models_sorted_into_correct_groups(self):
        hw = _make_hw(vram_gb=10.0, ram_gb=32.0)
        models = [
            _make_model("code-model", "7B", 4.0, use_cases=["coding"]),
            _make_model("chat-model", "7B", 4.0, use_cases=["chat"]),
        ]
        recs = get_recommendations(models, hw)
        grouped = group_by_use_case(recs)

        coding_names = [r.model.name for r in grouped["coding"]]
        chat_names = [r.model.name for r in grouped["chat"]]
        assert "code-model" in coding_names
        assert "chat-model" in chat_names

    def test_no_per_group_cap(self):
        """group_by_use_case should not cap groups at 5 models."""
        hw = _make_hw(vram_gb=10.0, ram_gb=32.0)
        # Create 7 distinct chat models
        models = [
            _make_model(f"chat-model-{i}", "7B", 4.0, use_cases=["chat"])
            for i in range(7)
        ]
        recs = get_recommendations(models, hw, top_n=20)
        grouped = group_by_use_case(recs)
        # All 7 models should appear — no cap of 5
        assert len(grouped["chat"]) == 7
