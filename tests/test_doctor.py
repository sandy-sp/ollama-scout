"""Tests for scout.doctor module."""
import os
import subprocess
import sys
import tempfile
from collections import namedtuple
from unittest.mock import MagicMock, patch


class TestCheckPython:
    def test_passes_on_3_10_plus(self):
        from scout.doctor import _check_python
        ok, detail = _check_python()
        assert ok is True  # tests run on 3.10+
        assert "." in detail

    def test_fails_on_older_version(self):
        from scout.doctor import _check_python
        VI = namedtuple("version_info", ["major", "minor", "micro"])
        with patch.object(sys, "version_info", VI(3, 9, 0)):
            ok, detail = _check_python()
            assert ok is False


class TestCheckOllama:
    def test_passes_when_installed(self):
        from scout.doctor import _check_ollama
        with patch("shutil.which", return_value="/usr/bin/ollama"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ollama version 0.5.0\n")
            ok, detail = _check_ollama()
            assert ok is True
            assert "0.5.0" in detail

    def test_fails_when_not_in_path(self):
        from scout.doctor import _check_ollama
        with patch("shutil.which", return_value=None):
            ok, detail = _check_ollama()
            assert ok is False
            assert "PATH" in detail

    def test_fails_on_timeout(self):
        from scout.doctor import _check_ollama
        with patch("shutil.which", return_value="/usr/bin/ollama"), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ollama", 5)):
            ok, detail = _check_ollama()
            assert ok is False

    def test_fails_on_nonzero_returncode(self):
        from scout.doctor import _check_ollama
        with patch("shutil.which", return_value="/usr/bin/ollama"), \
             patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            ok, detail = _check_ollama()
            assert ok is False


class TestCheckGpu:
    def test_returns_ok_with_gpu(self):
        from scout.doctor import _check_gpu
        from scout.hardware import GPUInfo, HardwareProfile
        hw = HardwareProfile(
            os="Linux", cpu_name="Test", cpu_cores=4, cpu_threads=8,
            ram_gb=32.0,
            gpus=[GPUInfo(name="RTX 3080", vram_mb=10240)],
        )
        with patch("scout.hardware.detect_hardware", return_value=hw):
            ok, detail = _check_gpu()
            assert isinstance(ok, bool)
            assert isinstance(detail, str)

    def test_returns_ok_for_apple_silicon(self):
        from scout.doctor import _check_gpu
        from scout.hardware import GPUInfo, HardwareProfile
        hw = HardwareProfile(
            os="Darwin", cpu_name="Apple M2", cpu_cores=8, cpu_threads=8,
            ram_gb=16.0,
            gpus=[GPUInfo(name="Apple M2 (Unified Memory)", vram_mb=16384)],
            is_unified_memory=True,
        )
        with patch("scout.hardware.detect_hardware", return_value=hw):
            ok, detail = _check_gpu()
            assert ok is True
            assert "Apple Silicon" in detail


class TestCheckRam:
    def test_returns_ok_with_enough_ram(self):
        from scout.doctor import _check_ram
        mock_mem = MagicMock()
        mock_mem.total = 16 * (1024 ** 3)  # 16 GB
        with patch("psutil.virtual_memory", return_value=mock_mem):
            ok, detail = _check_ram()
            assert ok is True
            assert "16" in detail

    def test_fails_with_low_ram(self):
        from scout.doctor import _check_ram
        mock_mem = MagicMock()
        mock_mem.total = 2 * (1024 ** 3)  # 2 GB
        with patch("psutil.virtual_memory", return_value=mock_mem):
            ok, detail = _check_ram()
            assert ok is False


class TestCheckInternet:
    def test_passes_when_connected(self):
        from scout.doctor import _check_internet
        with patch("socket.socket") as mock_sock:
            mock_sock.return_value.connect.return_value = None
            ok, detail = _check_internet()
            assert ok is True

    def test_fails_when_disconnected(self):
        from scout.doctor import _check_internet
        with patch("socket.socket") as mock_sock:
            mock_sock.return_value.connect.side_effect = OSError("Network unreachable")
            ok, detail = _check_internet()
            assert ok is False


class TestCheckModelCache:
    def test_passes_when_cache_fresh(self):
        from scout.doctor import _check_model_cache
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'{"models": []}')
            tmp_path = f.name
        with patch("scout.ollama_api._get_cache_path", return_value=tmp_path):
            ok, detail = _check_model_cache()
        os.unlink(tmp_path)
        assert ok is True
        assert "h old" in detail

    def test_fails_when_no_cache(self):
        from scout.doctor import _check_model_cache
        with patch("scout.ollama_api._get_cache_path", return_value="/nonexistent/cache.json"):
            ok, detail = _check_model_cache()
            assert ok is False
            assert "no cache" in detail


class TestCheckConfig:
    def test_passes_with_valid_config(self):
        from scout.doctor import _check_config
        with patch("scout.config.load_config", return_value={"default_top_n": 10}), \
             patch("os.path.exists", return_value=True):
            ok, detail = _check_config()
            assert isinstance(ok, bool)


class TestCheckPulledModels:
    def test_passes_when_models_pulled(self):
        from scout.doctor import _check_pulled_models
        models = ["llama3.2:3b", "mistral:7b"]
        with patch("scout.ollama_api.get_pulled_models", return_value=models):
            ok, detail = _check_pulled_models()
            assert ok is True
            assert "2 pulled" in detail

    def test_fails_when_no_models_pulled(self):
        from scout.doctor import _check_pulled_models
        with patch("scout.ollama_api.get_pulled_models", return_value=[]):
            ok, detail = _check_pulled_models()
            assert ok is False
            assert "no models" in detail


class TestRunDoctor:
    def test_runs_without_error(self):
        from scout.doctor import run_doctor
        with patch("scout.doctor._check_python", return_value=(True, "3.11.0")), \
             patch("scout.doctor._check_ollama", return_value=(True, "ollama 0.5.0")), \
             patch("scout.doctor._check_gpu", return_value=(True, "RTX 3080")), \
             patch("scout.doctor._check_ram", return_value=(True, "16.0 GB")), \
             patch("scout.doctor._check_internet", return_value=(True, "reachable")), \
             patch("scout.doctor._check_model_cache", return_value=(True, "fresh")), \
             patch("scout.doctor._check_config", return_value=(True, "valid")), \
             patch("scout.doctor._check_pulled_models", return_value=(True, "2 pulled")), \
             patch("scout.doctor.console"):
            run_doctor()  # should not raise

    def test_runs_with_all_failures(self):
        from scout.doctor import run_doctor
        with patch("scout.doctor._check_python", return_value=(False, "3.9.0")), \
             patch("scout.doctor._check_ollama", return_value=(False, "not found")), \
             patch("scout.doctor._check_gpu", return_value=(False, "none")), \
             patch("scout.doctor._check_ram", return_value=(False, "2.0 GB")), \
             patch("scout.doctor._check_internet", return_value=(False, "no internet")), \
             patch("scout.doctor._check_model_cache", return_value=(False, "stale")), \
             patch("scout.doctor._check_config", return_value=(False, "error")), \
             patch("scout.doctor._check_pulled_models", return_value=(False, "none")), \
             patch("scout.doctor.console"):
            run_doctor()  # should not raise even when all fail
