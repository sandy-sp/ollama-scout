"""Tests for scout.hardware module."""
import json
from unittest.mock import MagicMock, patch

from scout.hardware import (
    GPUInfo,
    HardwareProfile,
    _detect_cpu_windows_ps,
    _detect_gpus_windows_ps,
    _detect_ram_gb,
    _detect_ram_windows_ps,
    detect_hardware,
)


class TestDetectHardware:
    @patch("scout.hardware._detect_ram_gb", return_value=32.0)
    @patch(
        "scout.hardware._detect_gpus_nvidia",
        return_value=[GPUInfo(name="RTX 3080", vram_mb=10240)],
    )
    @patch("scout.hardware._detect_cpu_linux", return_value=("AMD Ryzen 9 5900X", 12, 24))
    @patch("scout.hardware.platform")
    def test_returns_hardware_profile_with_correct_types(
        self, mock_platform, mock_cpu, mock_gpu, mock_ram
    ):
        mock_platform.system.return_value = "Linux"
        hw = detect_hardware()

        assert isinstance(hw, HardwareProfile)
        assert isinstance(hw.os, str)
        assert isinstance(hw.cpu_name, str)
        assert isinstance(hw.cpu_cores, int)
        assert isinstance(hw.cpu_threads, int)
        assert isinstance(hw.ram_gb, float)
        assert isinstance(hw.gpus, list)
        assert isinstance(hw.is_unified_memory, bool)

        assert hw.os == "Linux"
        assert hw.cpu_name == "AMD Ryzen 9 5900X"
        assert hw.cpu_cores == 12
        assert hw.cpu_threads == 24
        assert hw.ram_gb == 32.0
        assert len(hw.gpus) == 1
        assert hw.gpus[0].vram_gb == 10.0
        assert hw.is_unified_memory is False

    @patch("scout.hardware._detect_ram_gb", return_value=16.0)
    @patch("scout.hardware._detect_gpus_nvidia", return_value=[])
    @patch("scout.hardware._detect_gpus_macos", return_value=[])
    @patch("scout.hardware._is_apple_silicon", return_value=True)
    @patch("scout.hardware._detect_cpu_macos", return_value=("Apple M2", 8, 8))
    @patch("scout.hardware.platform")
    def test_apple_silicon_unified_memory(
        self, mock_platform, mock_cpu, mock_apple, mock_macos_gpu, mock_nvidia, mock_ram
    ):
        mock_platform.system.return_value = "Darwin"
        hw = detect_hardware()

        assert hw.is_unified_memory is True
        assert hw.best_vram_gb == 16.0
        assert len(hw.gpus) == 1
        assert "Unified Memory" in hw.gpus[0].name


class TestDetectRamGb:
    @patch("scout.hardware.platform")
    def test_returns_positive_float(self, mock_platform):
        """On the current system, _detect_ram_gb should return a real positive value."""
        mock_platform.system.return_value = "Linux"
        ram = _detect_ram_gb()
        assert isinstance(ram, float)
        assert ram > 0


class TestWindowsPowerShellFallback:
    @patch("scout.hardware.subprocess.run")
    def test_detect_gpus_windows_ps(self, mock_run):
        ps_output = json.dumps([
            {"Name": "NVIDIA RTX 4090", "AdapterRAM": 25769803776},
        ])
        mock_run.return_value = MagicMock(stdout=ps_output, returncode=0)
        gpus = _detect_gpus_windows_ps()
        assert len(gpus) == 1
        assert gpus[0].name == "NVIDIA RTX 4090"
        assert gpus[0].vram_mb == 24576

    @patch("scout.hardware.subprocess.run")
    def test_detect_gpus_windows_ps_single_gpu(self, mock_run):
        """PowerShell returns a dict (not list) for a single GPU."""
        ps_output = json.dumps({"Name": "Intel UHD 630", "AdapterRAM": 1073741824})
        mock_run.return_value = MagicMock(stdout=ps_output, returncode=0)
        gpus = _detect_gpus_windows_ps()
        assert len(gpus) == 1
        assert gpus[0].name == "Intel UHD 630"

    @patch("scout.hardware.subprocess.run")
    def test_detect_cpu_windows_ps(self, mock_run):
        ps_output = json.dumps({
            "Name": "Intel Core i9-13900K",
            "NumberOfCores": 24,
            "NumberOfLogicalProcessors": 32,
        })
        mock_run.return_value = MagicMock(stdout=ps_output, returncode=0)
        name, cores, threads = _detect_cpu_windows_ps()
        assert name == "Intel Core i9-13900K"
        assert cores == 24
        assert threads == 32

    @patch("scout.hardware.subprocess.run")
    def test_detect_ram_windows_ps(self, mock_run):
        ps_output = json.dumps({"TotalPhysicalMemory": 34359738368})  # 32 GB
        mock_run.return_value = MagicMock(stdout=ps_output, returncode=0)
        ram = _detect_ram_windows_ps()
        assert ram == 32.0

    @patch("scout.hardware.subprocess.run", side_effect=Exception("powershell not found"))
    def test_detect_gpus_windows_ps_handles_error(self, mock_run):
        gpus = _detect_gpus_windows_ps()
        assert gpus == []

    @patch("scout.hardware.subprocess.run", side_effect=Exception("powershell not found"))
    def test_detect_ram_windows_ps_handles_error(self, mock_run):
        ram = _detect_ram_windows_ps()
        assert ram == 0.0
