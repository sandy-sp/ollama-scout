"""Tests for scout.benchmark module."""
import subprocess
from unittest.mock import patch

from scout.benchmark import (
    BenchmarkEstimate,
    benchmark_pulled_models,
    estimate_speed,
    get_benchmark_estimates,
    run_real_benchmark,
)
from scout.hardware import GPUInfo, HardwareProfile
from scout.ollama_api import ModelVariant, OllamaModel
from scout.recommender import Recommendation


def _make_hw(vram_gb=10.0, ram_gb=32.0, threads=16):
    gpus = [GPUInfo(name="Test GPU", vram_mb=int(vram_gb * 1024))]
    return HardwareProfile(
        os="Linux",
        cpu_name="Test CPU",
        cpu_cores=8,
        cpu_threads=threads,
        ram_gb=ram_gb,
        gpus=gpus,
    )


def _make_rec(name="test-model", size_gb=4.0, run_mode="GPU"):
    variant = ModelVariant(
        tag="7b", size_gb=size_gb,
        quantization="Q4_K_M", param_size="7B",
    )
    model = OllamaModel(
        name=name, description="Test",
        tags=[variant], use_cases=["chat"],
    )
    return Recommendation(
        model=model, variant=variant,
        score=90, run_mode=run_mode,
        fit_label="Excellent", note="test",
    )


class TestEstimateSpeed:
    def test_gpu_mode(self):
        hw = _make_hw(vram_gb=10.0)
        rec = _make_rec(size_gb=4.0, run_mode="GPU")
        est = estimate_speed(rec, hw)

        assert isinstance(est, BenchmarkEstimate)
        assert est.run_mode == "GPU"
        assert est.tokens_per_sec > 0
        # (10/4)*40 = 100, should be Fast
        assert est.rating == "Fast"

    def test_gpu_mode_cap_at_120(self):
        hw = _make_hw(vram_gb=24.0)
        rec = _make_rec(size_gb=1.0, run_mode="GPU")
        est = estimate_speed(rec, hw)

        # (24/1)*40 = 960, capped at 120
        assert est.tokens_per_sec == 120.0

    def test_cpu_gpu_mode(self):
        hw = _make_hw(vram_gb=6.0, ram_gb=32.0)
        rec = _make_rec(size_gb=8.0, run_mode="CPU+GPU")
        est = estimate_speed(rec, hw)

        assert est.run_mode == "CPU+GPU"
        assert est.tokens_per_sec > 0
        assert est.tokens_per_sec <= 45.0

    def test_cpu_mode(self):
        hw = _make_hw(vram_gb=0, ram_gb=16.0, threads=8)
        rec = _make_rec(size_gb=4.0, run_mode="CPU")
        est = estimate_speed(rec, hw)

        assert est.run_mode == "CPU"
        # (8/4)*4 = 8, Slow
        assert est.tokens_per_sec == 8.0
        assert est.rating == "Slow"

    def test_cpu_mode_cap_at_20(self):
        hw = _make_hw(vram_gb=0, ram_gb=64.0, threads=64)
        rec = _make_rec(size_gb=1.0, run_mode="CPU")
        est = estimate_speed(rec, hw)

        assert est.tokens_per_sec == 20.0

    def test_zero_size_handled(self):
        hw = _make_hw()
        rec = _make_rec(size_gb=0.0, run_mode="GPU")
        est = estimate_speed(rec, hw)
        assert est.tokens_per_sec > 0

    def test_moderate_rating(self):
        hw = _make_hw(vram_gb=8.0, ram_gb=32.0)
        rec = _make_rec(size_gb=8.0, run_mode="CPU+GPU")
        est = estimate_speed(rec, hw)
        # (8+32)/8*12 = 60, capped at 45 => Moderate
        assert est.rating == "Moderate"


class TestGetBenchmarkEstimates:
    def test_returns_correct_count(self):
        hw = _make_hw()
        recs = [_make_rec(f"model-{i}") for i in range(5)]
        estimates = get_benchmark_estimates(recs, hw, top_n=3)
        assert len(estimates) == 3

    def test_returns_all_when_fewer_than_top_n(self):
        hw = _make_hw()
        recs = [_make_rec("model-1")]
        estimates = get_benchmark_estimates(recs, hw, top_n=5)
        assert len(estimates) == 1

    def test_formula_estimates_have_is_real_false(self):
        hw = _make_hw()
        recs = [_make_rec("model-1")]
        estimates = get_benchmark_estimates(recs, hw, top_n=1)
        assert estimates[0].is_real is False


class TestRunRealBenchmark:
    @patch("scout.benchmark.shutil.which", return_value=None)
    def test_returns_none_when_ollama_not_installed(self, mock_which):
        result = run_real_benchmark("llama3.2:latest")
        assert result is None

    @patch("scout.benchmark.subprocess.run")
    @patch("scout.benchmark.shutil.which", return_value="/usr/bin/ollama")
    def test_returns_none_when_model_not_pulled(self, mock_which, mock_run):
        # ollama list returns output without our model
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ollama", "list"],
            returncode=0,
            stdout=(
                "NAME           ID     SIZE   MODIFIED\n"
                "mistral:latest abc123 4.1GB  2 days ago\n"
            ),
        )
        result = run_real_benchmark("llama3.2:latest")
        assert result is None

    @patch("scout.benchmark.time.monotonic", side_effect=[0.0, 5.0])
    @patch("scout.benchmark.subprocess.run")
    @patch("scout.benchmark.shutil.which", return_value="/usr/bin/ollama")
    def test_returns_float_on_success(self, mock_which, mock_run, mock_time):
        # First call: ollama list (model is pulled)
        list_result = subprocess.CompletedProcess(
            args=["ollama", "list"],
            returncode=0,
            stdout=(
                "NAME             ID     SIZE   MODIFIED\n"
                "llama3.2:latest  abc123 2.0GB  1 day ago\n"
            ),
        )
        # Second call: ollama run (produces output)
        run_result = subprocess.CompletedProcess(
            args=["ollama", "run", "llama3.2:latest", "Hello, how are you?"],
            returncode=0,
            stdout="Hello! I'm doing well, thank you for asking. How can I help you today?",
        )
        mock_run.side_effect = [list_result, run_result]

        result = run_real_benchmark("llama3.2:latest")
        assert isinstance(result, float)
        assert result > 0

    @patch("scout.benchmark.subprocess.run")
    @patch("scout.benchmark.shutil.which", return_value="/usr/bin/ollama")
    def test_returns_none_on_timeout(self, mock_which, mock_run):
        # ollama list succeeds
        list_result = subprocess.CompletedProcess(
            args=["ollama", "list"],
            returncode=0,
            stdout=(
                "NAME             ID     SIZE   MODIFIED\n"
                "llama3.2:latest  abc123 2.0GB  1 day ago\n"
            ),
        )
        timeout_err = subprocess.TimeoutExpired(cmd="ollama", timeout=60)
        mock_run.side_effect = [list_result, timeout_err]

        result = run_real_benchmark("llama3.2:latest")
        assert result is None


class TestBenchmarkPulledModels:
    @patch("scout.benchmark.run_real_benchmark", return_value=45.0)
    def test_returns_real_estimates(self, mock_bench):
        hw = _make_hw(vram_gb=10.0)
        estimates = benchmark_pulled_models(["llama3.2:latest"], hw)
        assert len(estimates) == 1
        assert estimates[0].is_real is True
        assert estimates[0].tokens_per_sec == 45.0
        assert estimates[0].run_mode == "GPU"

    @patch("scout.benchmark.run_real_benchmark", return_value=None)
    def test_skips_failed_benchmarks(self, mock_bench):
        hw = _make_hw(vram_gb=0)
        estimates = benchmark_pulled_models(["llama3.2:latest"], hw)
        assert len(estimates) == 0
