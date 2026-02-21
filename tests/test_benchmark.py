"""Tests for scout.benchmark module."""
import subprocess
from unittest.mock import patch

from scout.benchmark import (
    BenchmarkEstimate,
    benchmark_model,
    benchmark_pulled_models,
)
from scout.hardware import GPUInfo, HardwareProfile


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


class TestBenchmarkModel:
    @patch("scout.benchmark.shutil.which", return_value=None)
    def test_returns_none_when_ollama_not_installed(self, mock_which):
        result = benchmark_model("llama3.2:latest")
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
        result = benchmark_model("llama3.2:latest")
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

        result = benchmark_model("llama3.2:latest")
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

        result = benchmark_model("llama3.2:latest")
        assert result is None

    @patch("scout.benchmark.subprocess.run")
    @patch("scout.benchmark.shutil.which", return_value="/usr/bin/ollama")
    def test_returns_none_when_ollama_list_fails(self, mock_which, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ollama", "list"], returncode=1, stdout="",
        )
        result = benchmark_model("llama3.2:latest")
        assert result is None

    @patch("scout.benchmark.time.monotonic", side_effect=[0.0, 10.0])
    @patch("scout.benchmark.subprocess.run")
    @patch("scout.benchmark.shutil.which", return_value="/usr/bin/ollama")
    def test_returns_none_when_run_fails(self, mock_which, mock_run, mock_time):
        list_result = subprocess.CompletedProcess(
            args=["ollama", "list"],
            returncode=0,
            stdout=(
                "NAME             ID     SIZE   MODIFIED\n"
                "llama3.2:latest  abc123 2.0GB  1 day ago\n"
            ),
        )
        run_result = subprocess.CompletedProcess(
            args=["ollama", "run", "llama3.2:latest", "Hello, how are you?"],
            returncode=1,
            stdout="",
        )
        mock_run.side_effect = [list_result, run_result]
        result = benchmark_model("llama3.2:latest")
        assert result is None


class TestBenchmarkPulledModels:
    @patch("scout.benchmark.benchmark_model", return_value=45.0)
    def test_returns_estimates(self, mock_bench):
        hw = _make_hw(vram_gb=10.0)
        estimates = benchmark_pulled_models(["llama3.2:latest"], hw)
        assert len(estimates) == 1
        assert estimates[0].tokens_per_sec == 45.0
        assert estimates[0].run_mode == "GPU"

    @patch("scout.benchmark.benchmark_model", return_value=None)
    def test_skips_failed_benchmarks(self, mock_bench):
        hw = _make_hw(vram_gb=0)
        estimates = benchmark_pulled_models(["llama3.2:latest"], hw)
        assert len(estimates) == 0

    @patch("scout.benchmark.benchmark_model", return_value=70.0)
    def test_fast_rating(self, mock_bench):
        hw = _make_hw(vram_gb=10.0)
        estimates = benchmark_pulled_models(["model:7b"], hw)
        assert estimates[0].rating == "Fast"

    @patch("scout.benchmark.benchmark_model", return_value=30.0)
    def test_moderate_rating(self, mock_bench):
        hw = _make_hw(vram_gb=10.0)
        estimates = benchmark_pulled_models(["model:7b"], hw)
        assert estimates[0].rating == "Moderate"

    @patch("scout.benchmark.benchmark_model", return_value=10.0)
    def test_slow_rating(self, mock_bench):
        hw = _make_hw(vram_gb=10.0)
        estimates = benchmark_pulled_models(["model:7b"], hw)
        assert estimates[0].rating == "Slow"

    @patch("scout.benchmark.benchmark_model", return_value=25.0)
    def test_cpu_mode_when_no_vram(self, mock_bench):
        hw = _make_hw(vram_gb=0)
        estimates = benchmark_pulled_models(["model:7b"], hw)
        assert estimates[0].run_mode == "CPU"


class TestBenchmarkEstimateDataclass:
    def test_can_create_benchmark_estimate(self):
        est = BenchmarkEstimate(
            model_name="test:7b",
            run_mode="GPU",
            tokens_per_sec=80.0,
            rating="Fast",
        )
        assert est.model_name == "test:7b"
        assert est.tokens_per_sec == 80.0
        assert est.rating == "Fast"
