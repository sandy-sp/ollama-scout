"""Tests for scout.hardware module."""
import json  # noqa: F401
from unittest.mock import MagicMock, mock_open, patch

from scout.hardware import (
    GPUInfo,
    HardwareProfile,
    _detect_cpu_linux,
    _detect_cpu_macos,
    _detect_cpu_windows,
    _detect_cpu_windows_ps,
    _detect_gpus_amd_linux,
    _detect_gpus_macos,
    _detect_gpus_nvidia,
    _detect_gpus_windows_ps,
    _detect_gpus_windows_wmi,
    _detect_ram_gb,
    _detect_ram_windows_ps,
    _is_apple_silicon,
    detect_hardware,
)

# ---------------------------------------------------------------------------
# HardwareProfile properties
# ---------------------------------------------------------------------------

class TestHardwareProfileProperties:
    def test_total_vram_gb_unified_memory(self):
        hw = HardwareProfile(
            os="Darwin", cpu_name="Apple M2", cpu_cores=8, cpu_threads=8,
            ram_gb=16.0, gpus=[], is_unified_memory=True,
        )
        assert hw.total_vram_gb == 16.0

    def test_total_vram_gb_multi_gpu(self):
        hw = HardwareProfile(
            os="Linux", cpu_name="Test", cpu_cores=8, cpu_threads=16,
            ram_gb=32.0,
            gpus=[
                GPUInfo(name="GPU 0", vram_mb=8192),
                GPUInfo(name="GPU 1", vram_mb=8192),
            ],
        )
        assert hw.total_vram_gb == 16.0

    def test_combined_vram_gb_equals_total(self):
        hw = HardwareProfile(
            os="Linux", cpu_name="Test", cpu_cores=8, cpu_threads=16,
            ram_gb=32.0,
            gpus=[
                GPUInfo(name="GPU 0", vram_mb=10240),
                GPUInfo(name="GPU 1", vram_mb=10240),
            ],
        )
        assert hw.combined_vram_gb == hw.total_vram_gb

    def test_best_vram_gb_no_gpus(self):
        hw = HardwareProfile(
            os="Linux", cpu_name="Test", cpu_cores=4, cpu_threads=8,
            ram_gb=16.0, gpus=[],
        )
        assert hw.best_vram_gb == 0.0

    def test_best_vram_gb_multiple_gpus_picks_max(self):
        hw = HardwareProfile(
            os="Linux", cpu_name="Test", cpu_cores=8, cpu_threads=16,
            ram_gb=32.0,
            gpus=[
                GPUInfo(name="GPU 0", vram_mb=8192),   # 8GB
                GPUInfo(name="GPU 1", vram_mb=24576),  # 24GB
            ],
        )
        assert hw.best_vram_gb == 24.0

    def test_multi_gpu_property(self):
        single_hw = HardwareProfile(
            os="Linux", cpu_name="Test", cpu_cores=4, cpu_threads=8,
            ram_gb=16.0, gpus=[GPUInfo(name="GPU 0", vram_mb=8192)],
        )
        multi_hw = HardwareProfile(
            os="Linux", cpu_name="Test", cpu_cores=8, cpu_threads=16,
            ram_gb=32.0,
            gpus=[GPUInfo(name="GPU 0", vram_mb=8192), GPUInfo(name="GPU 1", vram_mb=8192)],
        )
        assert single_hw.multi_gpu is False
        assert multi_hw.multi_gpu is True


# ---------------------------------------------------------------------------
# _is_apple_silicon
# ---------------------------------------------------------------------------

class TestIsAppleSilicon:
    def test_returns_false_on_linux(self):
        # We're on Linux; platform.system() != "Darwin" → immediate False
        result = _is_apple_silicon()
        assert result is False

    @patch("scout.hardware.platform.processor", return_value="arm")
    @patch("scout.hardware.platform.system", return_value="Darwin")
    def test_returns_true_for_arm_processor(self, mock_sys, mock_proc):
        assert _is_apple_silicon() is True

    @patch("scout.hardware.platform.processor", return_value="AppleSilicon")
    @patch("scout.hardware.platform.system", return_value="Darwin")
    def test_returns_true_for_apple_in_processor(self, mock_sys, mock_proc):
        assert _is_apple_silicon() is True

    @patch("scout.hardware.subprocess.run")
    @patch("scout.hardware.platform.processor", return_value="i386")
    @patch("scout.hardware.platform.system", return_value="Darwin")
    def test_uses_sysctl_fallback_returns_true(self, mock_sys, mock_proc, mock_run):
        mock_run.return_value = MagicMock(stdout="1\n", returncode=0)
        assert _is_apple_silicon() is True

    @patch("scout.hardware.subprocess.run")
    @patch("scout.hardware.platform.processor", return_value="i386")
    @patch("scout.hardware.platform.system", return_value="Darwin")
    def test_uses_sysctl_fallback_returns_false(self, mock_sys, mock_proc, mock_run):
        mock_run.return_value = MagicMock(stdout="0\n", returncode=0)
        assert _is_apple_silicon() is False

    @patch("scout.hardware.subprocess.run", side_effect=Exception("no sysctl"))
    @patch("scout.hardware.platform.processor", return_value="i386")
    @patch("scout.hardware.platform.system", return_value="Darwin")
    def test_returns_false_on_sysctl_error(self, mock_sys, mock_proc, mock_run):
        assert _is_apple_silicon() is False


# ---------------------------------------------------------------------------
# _detect_gpus_nvidia
# ---------------------------------------------------------------------------

class TestDetectGpusNvidia:
    @patch("scout.hardware.shutil.which", return_value=None)
    def test_returns_empty_when_nvidia_smi_not_found(self, mock_which):
        assert _detect_gpus_nvidia() == []

    @patch("scout.hardware.subprocess.run")
    @patch("scout.hardware.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_returns_gpu_info_on_success(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(
            stdout="NVIDIA RTX 3080, 10240\nNVIDIA RTX 3080 Ti, 12288\n",
            returncode=0,
        )
        gpus = _detect_gpus_nvidia()
        assert len(gpus) == 2
        assert gpus[0].name == "NVIDIA RTX 3080"
        assert gpus[0].vram_mb == 10240
        assert gpus[1].vram_mb == 12288

    @patch("scout.hardware.subprocess.run", side_effect=Exception("nvidia-smi failed"))
    @patch("scout.hardware.shutil.which", return_value="/usr/bin/nvidia-smi")
    def test_returns_empty_on_error(self, mock_which, mock_run):
        assert _detect_gpus_nvidia() == []


# ---------------------------------------------------------------------------
# _detect_gpus_amd_linux
# ---------------------------------------------------------------------------

class TestDetectGpusAmdLinux:
    @patch("scout.hardware.shutil.which", return_value=None)
    def test_returns_empty_when_no_rocm_smi(self, mock_which):
        assert _detect_gpus_amd_linux() == []

    @patch("scout.hardware.subprocess.run")
    @patch("scout.hardware.shutil.which", return_value="/usr/bin/rocm-smi")
    def test_parses_rocm_output(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(
            stdout=(
                "GPU, Total, ...\n"
                "GPU 0, Total, 8388608\n"
            ),
            returncode=0,
        )
        gpus = _detect_gpus_amd_linux()
        assert len(gpus) == 1
        assert gpus[0].name == "AMD GPU (ROCm)"
        assert gpus[0].vram_mb == 8192  # 8388608 // 1024

    @patch("scout.hardware.subprocess.run", side_effect=Exception("rocm-smi failed"))
    @patch("scout.hardware.shutil.which", return_value="/usr/bin/rocm-smi")
    def test_returns_empty_on_error(self, mock_which, mock_run):
        assert _detect_gpus_amd_linux() == []


# ---------------------------------------------------------------------------
# _detect_gpus_macos
# ---------------------------------------------------------------------------

class TestDetectGpusMacos:
    @patch("scout.hardware.subprocess.run")
    def test_parses_system_profiler_gb_output(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout=(
                "      Chipset Model: AMD Radeon Pro 5500M\n"
                "      VRAM (Total): 8 GB\n"
            ),
            returncode=0,
        )
        gpus = _detect_gpus_macos()
        assert len(gpus) == 1
        assert gpus[0].name == "AMD Radeon Pro 5500M"
        assert gpus[0].vram_mb == 8 * 1024

    @patch("scout.hardware.subprocess.run")
    def test_parses_system_profiler_mb_output(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout=(
                "      Chipset Model: Intel Iris Pro\n"
                "      VRAM (Total): 1536 MB\n"
            ),
            returncode=0,
        )
        gpus = _detect_gpus_macos()
        assert len(gpus) == 1
        assert gpus[0].vram_mb == 1536

    @patch("scout.hardware.subprocess.run", side_effect=Exception("no system_profiler"))
    def test_returns_empty_on_error(self, mock_run):
        assert _detect_gpus_macos() == []


# ---------------------------------------------------------------------------
# _detect_gpus_windows_wmi
# ---------------------------------------------------------------------------

class TestDetectGpusWindowsWmi:
    @patch("scout.hardware.subprocess.run")
    def test_parses_wmic_output(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout=(
                "Node,AdapterRAM,Name\n"
                "DESKTOP,8589934592,NVIDIA GeForce RTX 3080\n"
            ),
            returncode=0,
        )
        gpus = _detect_gpus_windows_wmi()
        assert len(gpus) == 1
        assert gpus[0].name == "NVIDIA GeForce RTX 3080"
        assert gpus[0].vram_mb == 8192  # 8589934592 // (1024*1024)

    @patch("scout.hardware.subprocess.run", side_effect=Exception("wmic failed"))
    def test_returns_empty_on_error(self, mock_run):
        assert _detect_gpus_windows_wmi() == []


# ---------------------------------------------------------------------------
# _detect_cpu_linux
# ---------------------------------------------------------------------------

class TestDetectCpuLinux:
    def test_returns_real_values_on_linux(self):
        # Directly calling on Linux covers the actual code path
        name, cores, threads = _detect_cpu_linux()
        assert isinstance(name, str)
        assert isinstance(cores, int)
        assert isinstance(threads, int)
        assert cores >= 1
        assert threads >= 1

    @patch("builtins.open", side_effect=OSError("no /proc/cpuinfo"))
    def test_returns_defaults_when_proc_unavailable(self, mock_open_fn):
        import multiprocessing
        name, cores, threads = _detect_cpu_linux()
        assert name == "Unknown CPU"
        assert cores == 1
        assert threads == multiprocessing.cpu_count()

    @patch("builtins.open", mock_open(read_data=(
        "model name\t: Intel Core i7-12700K\ncpu cores\t: 12\n"
    )))
    def test_parses_proc_cpuinfo(self):
        name, cores, threads = _detect_cpu_linux()
        assert name == "Intel Core i7-12700K"
        assert cores == 12


# ---------------------------------------------------------------------------
# _detect_cpu_macos
# ---------------------------------------------------------------------------

class TestDetectCpuMacos:
    @patch("scout.hardware.subprocess.run")
    def test_parses_sysctl_output(self, mock_run):
        mock_run.side_effect = [
            MagicMock(stdout="Apple M2 Pro\n", returncode=0),
            MagicMock(stdout="10\n", returncode=0),
        ]
        name, cores, threads = _detect_cpu_macos()
        assert name == "Apple M2 Pro"
        assert cores == 10

    @patch("scout.hardware.subprocess.run", side_effect=Exception("no sysctl"))
    def test_returns_defaults_on_error(self, mock_run):
        name, cores, threads = _detect_cpu_macos()
        assert name == "Unknown CPU"


# ---------------------------------------------------------------------------
# _detect_cpu_windows
# ---------------------------------------------------------------------------

class TestDetectCpuWindows:
    @patch("scout.hardware.subprocess.run")
    @patch("scout.hardware.shutil.which", return_value="C:\\Windows\\wmic.exe")
    def test_uses_wmic_when_available(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(
            stdout=(
                "Node,Name,NumberOfCores,NumberOfLogicalProcessors\n"
                "DESKTOP,Intel Core i9-13900K,24,32\n"
            ),
            returncode=0,
        )
        name, cores, threads = _detect_cpu_windows()
        assert name == "Intel Core i9-13900K"
        assert cores == 24
        assert threads == 32

    @patch("scout.hardware._detect_cpu_windows_ps", return_value=("AMD Ryzen 9", 12, 24))
    @patch("scout.hardware.shutil.which", return_value=None)
    def test_falls_back_to_ps_when_no_wmic(self, mock_which, mock_ps):
        name, cores, threads = _detect_cpu_windows()
        assert name == "AMD Ryzen 9"
        mock_ps.assert_called_once()

    @patch("scout.hardware._detect_cpu_windows_ps", return_value=("Unknown CPU", 1, 1))
    @patch("scout.hardware.shutil.which", return_value=None)
    def test_returns_defaults_when_both_fail(self, mock_which, mock_ps):
        name, cores, threads = _detect_cpu_windows()
        assert name == "Unknown CPU"


# ---------------------------------------------------------------------------
# _detect_ram_gb fallback paths
# ---------------------------------------------------------------------------

class TestDetectRamGbFallback:
    @patch.dict("sys.modules", {"psutil": None})
    @patch("scout.hardware.platform.system", return_value="Linux")
    def test_linux_fallback_reads_proc_meminfo(self, mock_sys):
        fake_meminfo = "MemTotal:        33554432 kB\nMemFree: 1000 kB\n"
        with patch("builtins.open", mock_open(read_data=fake_meminfo)):
            ram = _detect_ram_gb()
        assert ram == 32.0  # 33554432 KB = 32 GB

    @patch.dict("sys.modules", {"psutil": None})
    @patch("scout.hardware.subprocess.run")
    @patch("scout.hardware.platform.system", return_value="Darwin")
    def test_darwin_fallback_uses_sysctl(self, mock_sys, mock_run):
        mock_run.return_value = MagicMock(stdout="34359738368\n", returncode=0)
        ram = _detect_ram_gb()
        assert ram == 32.0  # 34359738368 bytes = 32 GB

    @patch.dict("sys.modules", {"psutil": None})
    @patch("scout.hardware.subprocess.run")
    @patch("scout.hardware.shutil.which", return_value="C:\\Windows\\wmic.exe")
    @patch("scout.hardware.platform.system", return_value="Windows")
    def test_windows_fallback_uses_wmic(self, mock_sys, mock_which, mock_run):
        mock_run.return_value = MagicMock(
            stdout="Node,TotalPhysicalMemory\nDESKTOP,34359738368\n",
            returncode=0,
        )
        ram = _detect_ram_gb()
        assert ram == 32.0

    @patch.dict("sys.modules", {"psutil": None})
    @patch("scout.hardware._detect_ram_windows_ps", return_value=32.0)
    @patch("scout.hardware.shutil.which", return_value=None)
    @patch("scout.hardware.platform.system", return_value="Windows")
    def test_windows_fallback_uses_ps_when_no_wmic(self, mock_sys, mock_which, mock_ps):
        ram = _detect_ram_gb()
        assert ram == 32.0
        mock_ps.assert_called_once()

    @patch.dict("sys.modules", {"psutil": None})
    @patch("scout.hardware.platform.system", return_value="Linux")
    def test_returns_zero_on_error(self, mock_sys):
        with patch("builtins.open", side_effect=OSError("no meminfo")):
            ram = _detect_ram_gb()
        assert ram == 0.0


# ---------------------------------------------------------------------------
# detect_hardware — additional paths
# ---------------------------------------------------------------------------

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

    @patch("scout.hardware._detect_ram_gb", return_value=32.0)
    @patch("scout.hardware._detect_gpus_windows_wmi",
           return_value=[GPUInfo(name="RTX 4090", vram_mb=24576)])
    @patch("scout.hardware._detect_gpus_nvidia", return_value=[])
    @patch("scout.hardware._is_apple_silicon", return_value=False)
    @patch("scout.hardware._detect_cpu_windows", return_value=("Intel i9-13900K", 24, 32))
    @patch("scout.hardware.shutil.which", return_value="C:\\wmic.exe")
    @patch("scout.hardware.platform")
    def test_windows_with_wmic_gpu_detection(
        self, mock_platform, mock_which, mock_cpu, mock_apple, mock_nvidia, mock_wmi, mock_ram
    ):
        mock_platform.system.return_value = "Windows"
        hw = detect_hardware()
        assert hw.os == "Windows"
        assert len(hw.gpus) == 1
        assert hw.gpus[0].name == "RTX 4090"

    @patch("scout.hardware._detect_ram_gb", return_value=32.0)
    @patch("scout.hardware._detect_gpus_windows_ps",
           return_value=[GPUInfo(name="RTX 4090", vram_mb=24576)])
    @patch("scout.hardware._detect_gpus_nvidia", return_value=[])
    @patch("scout.hardware._is_apple_silicon", return_value=False)
    @patch("scout.hardware._detect_cpu_windows", return_value=("Intel i9", 16, 24))
    @patch("scout.hardware.shutil.which", return_value=None)
    @patch("scout.hardware.platform")
    def test_windows_fallback_to_ps_when_no_wmic(
        self, mock_platform, mock_which, mock_cpu, mock_apple, mock_nvidia, mock_ps, mock_ram
    ):
        mock_platform.system.return_value = "Windows"
        hw = detect_hardware()
        assert len(hw.gpus) == 1

    @patch("scout.hardware._detect_ram_gb", return_value=16.0)
    @patch("scout.hardware._detect_gpus_amd_linux",
           return_value=[GPUInfo(name="AMD RX 6800", vram_mb=16384)])
    @patch("scout.hardware._detect_gpus_nvidia", return_value=[])
    @patch("scout.hardware._is_apple_silicon", return_value=False)
    @patch("scout.hardware._detect_cpu_linux", return_value=("AMD Ryzen 7", 8, 16))
    @patch("scout.hardware.platform")
    def test_linux_amd_gpu_detection(
        self, mock_platform, mock_cpu, mock_apple, mock_nvidia, mock_amd, mock_ram
    ):
        mock_platform.system.return_value = "Linux"
        hw = detect_hardware()
        assert len(hw.gpus) == 1
        assert hw.gpus[0].name == "AMD RX 6800"

    @patch("scout.hardware._detect_ram_gb", return_value=16.0)
    @patch("scout.hardware._detect_gpus_macos",
           return_value=[GPUInfo(name="AMD Radeon", vram_mb=8192)])
    @patch("scout.hardware._detect_gpus_nvidia", return_value=[])
    @patch("scout.hardware._is_apple_silicon", return_value=False)
    @patch("scout.hardware._detect_cpu_macos", return_value=("Intel Core i7", 6, 12))
    @patch("scout.hardware.platform")
    def test_macos_non_apple_silicon(
        self, mock_platform, mock_cpu, mock_apple, mock_nvidia, mock_macos, mock_ram
    ):
        mock_platform.system.return_value = "Darwin"
        hw = detect_hardware()
        assert len(hw.gpus) == 1
        assert hw.is_unified_memory is False


# ---------------------------------------------------------------------------
# Windows PowerShell fallbacks (already existed, keeping them)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TestDetectRamGb (was already there)
# ---------------------------------------------------------------------------

class TestDetectRamGb:
    @patch("scout.hardware.platform")
    def test_returns_positive_float(self, mock_platform):
        """On the current system, _detect_ram_gb should return a real positive value."""
        mock_platform.system.return_value = "Linux"
        ram = _detect_ram_gb()
        assert isinstance(ram, float)
        assert ram > 0
