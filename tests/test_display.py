"""Tests for scout.display module."""
from io import StringIO

from rich.console import Console

from scout.benchmark import BenchmarkEstimate
from scout.display import (
    print_banner,
    print_benchmark,
    print_footer,
    print_hardware_summary,
    print_legend,
    print_model_comparison,
    print_recommendations_flat,
    print_recommendations_grouped,
)
from scout.hardware import GPUInfo, HardwareProfile
from scout.ollama_api import ModelVariant, OllamaModel
from scout.recommender import Recommendation


def _make_hw():
    return HardwareProfile(
        os="Linux",
        cpu_name="Test CPU",
        cpu_cores=8,
        cpu_threads=16,
        ram_gb=32.0,
        gpus=[GPUInfo(name="Test GPU", vram_mb=10240)],
    )


def _make_rec():
    variant = ModelVariant(
        tag="7b", size_gb=4.0,
        quantization="Q4_K_M", param_size="7B",
    )
    model = OllamaModel(
        name="test-model", description="A test model",
        tags=[variant], use_cases=["chat"],
    )
    return Recommendation(
        model=model, variant=variant,
        score=90, run_mode="GPU",
        fit_label="Excellent", note="Fits in VRAM",
    )


def _capture(fn, *args, **kwargs):
    """Run a display function and capture its Rich output."""
    buf = StringIO()
    test_console = Console(file=buf, force_terminal=True, width=120)
    import scout.display as display_mod
    original = display_mod.console
    display_mod.console = test_console
    try:
        fn(*args, **kwargs)
    finally:
        display_mod.console = original
    return buf.getvalue()


class TestDisplayFunctions:
    def test_print_banner_does_not_raise(self):
        output = _capture(print_banner)
        assert "ollama" in output
        assert "scout" in output

    def test_print_hardware_summary_shows_components(self):
        hw = _make_hw()
        output = _capture(print_hardware_summary, hw)
        assert "Test CPU" in output
        assert "32.0 GB" in output
        assert "Test GPU" in output

    def test_print_hardware_summary_no_gpu(self):
        hw = HardwareProfile(
            os="Linux", cpu_name="Test CPU",
            cpu_cores=4, cpu_threads=8, ram_gb=16.0, gpus=[],
        )
        output = _capture(print_hardware_summary, hw)
        assert "None detected" in output

    def test_print_recommendations_grouped(self):
        rec = _make_rec()
        grouped = {"coding": [], "reasoning": [], "chat": [rec]}
        output = _capture(
            print_recommendations_grouped, grouped, [],
        )
        assert "test-model" in output
        assert "Excellent" in output

    def test_print_recommendations_flat(self):
        rec = _make_rec()
        output = _capture(print_recommendations_flat, [rec])
        assert "test-model" in output

    def test_print_benchmark_shows_estimates(self):
        est = BenchmarkEstimate(
            model_name="test:7b", run_mode="GPU",
            tokens_per_sec=80.0, rating="Fast",
        )
        output = _capture(print_benchmark, [est])
        assert "80.0" in output
        assert "Fast" in output

    def test_print_legend_does_not_raise(self):
        output = _capture(print_legend)
        assert "Excellent" in output
        assert "GPU" in output

    def test_print_footer_does_not_raise(self):
        output = _capture(print_footer)
        assert "--help" in output

    def test_print_benchmark_shows_source_column(self):
        est_real = BenchmarkEstimate(
            model_name="test:7b", run_mode="GPU",
            tokens_per_sec=80.0, rating="Fast", is_real=True,
        )
        est_formula = BenchmarkEstimate(
            model_name="other:7b", run_mode="CPU",
            tokens_per_sec=8.0, rating="Slow", is_real=False,
        )
        output = _capture(print_benchmark, [est_real, est_formula])
        assert "Real" in output
        assert "Est." in output


class TestPrintModelComparison:
    def test_comparison_with_two_valid_models(self):
        d1 = {
            "name": "llama3.2", "description": "A chat model",
            "tag": "3b", "size_gb": 2.0, "param_size": "3B",
            "quantization": "Q4_K_M", "fit_label": "Excellent",
            "run_mode": "GPU", "score": 98, "est_tps": 80.0,
            "pulled": True,
        }
        d2 = {
            "name": "mistral", "description": "A reasoning model",
            "tag": "7b", "size_gb": 4.1, "param_size": "7B",
            "quantization": "Q4_K_M", "fit_label": "Good",
            "run_mode": "CPU+GPU", "score": 60, "est_tps": 35.0,
            "pulled": False,
        }
        output = _capture(print_model_comparison, d1, d2)
        assert "llama3.2" in output
        assert "mistral" in output
        assert "Verdict" in output

    def test_comparison_with_one_missing_model(self):
        d1 = {
            "name": "llama3.2", "description": "A chat model",
            "tag": "3b", "size_gb": 2.0, "param_size": "3B",
            "quantization": "Q4_K_M", "fit_label": "Excellent",
            "run_mode": "GPU", "score": 98, "est_tps": 80.0,
            "pulled": False,
        }
        output = _capture(print_model_comparison, d1, None)
        assert "llama3.2" in output
        assert "Not found" in output
