"""
hardware.py - Cross-platform hardware detection (GPU VRAM, CPU, RAM)
Supports: Windows, macOS (including Apple Silicon), Linux
"""
import platform
import shutil
import subprocess
from dataclasses import dataclass, field


@dataclass
class GPUInfo:
    name: str
    vram_mb: int

    @property
    def vram_gb(self) -> float:
        return round(self.vram_mb / 1024, 1)


@dataclass
class HardwareProfile:
    os: str
    cpu_name: str
    cpu_cores: int
    cpu_threads: int
    ram_gb: float
    gpus: list[GPUInfo] = field(default_factory=list)
    is_unified_memory: bool = False

    @property
    def total_vram_gb(self) -> float:
        if self.is_unified_memory:
            return self.ram_gb
        return round(sum(g.vram_mb for g in self.gpus) / 1024, 1)

    @property
    def best_vram_gb(self) -> float:
        if self.is_unified_memory:
            return self.ram_gb
        if not self.gpus:
            return 0.0
        return round(max(g.vram_mb for g in self.gpus) / 1024, 1)

    @property
    def combined_vram_gb(self) -> float:
        """Total VRAM across all GPUs."""
        return self.total_vram_gb

    @property
    def multi_gpu(self) -> bool:
        return len(self.gpus) > 1


def _is_apple_silicon() -> bool:
    """Detect if running on Apple Silicon (M1/M2/M3/M4)."""
    if platform.system() != "Darwin":
        return False
    # Check processor string
    proc = platform.processor()
    if proc == "arm" or "apple" in proc.lower():
        return True
    # Fallback: check sysctl
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() == "1"
    except Exception:
        return False


def _detect_gpus_nvidia() -> list[GPUInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                name = parts[0].strip()
                vram_mb = int(parts[1].strip())
                gpus.append(GPUInfo(name=name, vram_mb=vram_mb))
        return gpus
    except Exception:
        return []


def _detect_gpus_amd_linux() -> list[GPUInfo]:
    """Use rocm-smi or parse /sys for AMD GPUs on Linux."""
    gpus = []
    if shutil.which("rocm-smi"):
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--csv"],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.strip().splitlines():
                if "GPU" in line and "Total" in line:
                    parts = line.split(",")
                    if len(parts) >= 3:
                        try:
                            vram_mb = int(parts[2].strip()) // 1024
                            gpus.append(GPUInfo(name="AMD GPU (ROCm)", vram_mb=vram_mb))
                        except ValueError:
                            pass
        except Exception:
            pass
    return gpus


def _detect_gpus_macos() -> list[GPUInfo]:
    """Use system_profiler on macOS to detect GPU VRAM."""
    gpus = []
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType"],
            capture_output=True, text=True, timeout=15
        )
        current_gpu = None
        for line in result.stdout.splitlines():
            line = line.strip()
            if "Chipset Model:" in line:
                current_gpu = line.split(":", 1)[1].strip()
            elif "VRAM" in line and current_gpu:
                parts = line.split(":")
                if len(parts) == 2:
                    vram_str = parts[1].strip().lower()
                    try:
                        if "gb" in vram_str:
                            vram_mb = int(float(vram_str.replace("gb", "").strip()) * 1024)
                        elif "mb" in vram_str:
                            vram_mb = int(float(vram_str.replace("mb", "").strip()))
                        else:
                            continue
                        gpus.append(GPUInfo(name=current_gpu, vram_mb=vram_mb))
                        current_gpu = None
                    except ValueError:
                        pass
    except Exception:
        pass
    return gpus


def _detect_gpus_windows_wmi() -> list[GPUInfo]:
    """Use wmic on Windows to detect GPU VRAM."""
    gpus = []
    try:
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM", "/format:csv"],
            capture_output=True, text=True, timeout=10, shell=False
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split(",")
            if len(parts) >= 3 and parts[1].strip() and parts[1].strip() != "AdapterRAM":
                try:
                    vram_bytes = int(parts[1].strip())
                    name = parts[2].strip()
                    if vram_bytes > 0:
                        gpus.append(GPUInfo(name=name, vram_mb=vram_bytes // (1024 * 1024)))
                except ValueError:
                    pass
    except Exception:
        pass
    return gpus


def _detect_gpus_windows_ps() -> list[GPUInfo]:
    """Use PowerShell on Windows 11+ to detect GPU VRAM (fallback when wmic is absent)."""
    import json as _json
    gpus = []
    try:
        result = subprocess.run(
            [
                "powershell", "-Command",
                "Get-CimInstance Win32_VideoController"
                " | Select-Object Name, AdapterRAM"
                " | ConvertTo-Json",
            ],
            capture_output=True, text=True, timeout=15, shell=False,
        )
        data = _json.loads(result.stdout.strip())
        # PowerShell returns a single object if only one GPU, or a list
        if isinstance(data, dict):
            data = [data]
        for item in data:
            name = item.get("Name", "Unknown GPU")
            vram_bytes = item.get("AdapterRAM", 0) or 0
            if vram_bytes > 0:
                gpus.append(GPUInfo(name=name, vram_mb=int(vram_bytes) // (1024 * 1024)))
    except Exception:
        pass
    return gpus


def _detect_cpu_linux() -> tuple[str, int, int]:
    name, cores, threads = "Unknown CPU", 1, 1
    try:
        import multiprocessing
        threads = multiprocessing.cpu_count()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                if line.startswith("cpu cores"):
                    cores = int(line.split(":", 1)[1].strip())
    except Exception:
        pass
    return name, cores, threads


def _detect_cpu_macos() -> tuple[str, int, int]:
    import multiprocessing
    name, cores, threads = "Unknown CPU", 1, multiprocessing.cpu_count()
    try:
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True, timeout=5)
        name = result.stdout.strip() or name
        result2 = subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                                 capture_output=True, text=True, timeout=5)
        cores = int(result2.stdout.strip()) if result2.stdout.strip() else threads
    except Exception:
        pass
    return name, cores, threads


def _detect_cpu_windows() -> tuple[str, int, int]:
    import multiprocessing
    name, cores, threads = "Unknown CPU", 1, multiprocessing.cpu_count()

    if shutil.which("wmic"):
        try:
            result = subprocess.run(
                [
                    "wmic", "cpu", "get",
                    "Name,NumberOfCores,NumberOfLogicalProcessors",
                    "/format:csv",
                ],
                capture_output=True, text=True, timeout=10, shell=False
            )
            for line in result.stdout.strip().splitlines():
                parts = line.split(",")
                if len(parts) >= 4 and parts[1].strip() and parts[1].strip() != "Name":
                    name = parts[1].strip()
                    cores = int(parts[2].strip()) if parts[2].strip().isdigit() else cores
                    threads = int(parts[3].strip()) if parts[3].strip().isdigit() else threads
            return name, cores, threads
        except Exception:
            pass

    # PowerShell fallback for Windows 11+
    result = _detect_cpu_windows_ps()
    if result[0] != "Unknown CPU":
        return result

    return name, cores, threads


def _detect_cpu_windows_ps() -> tuple[str, int, int]:
    """Use PowerShell to detect CPU (fallback when wmic is absent)."""
    import json as _json
    import multiprocessing
    name, cores, threads = "Unknown CPU", 1, multiprocessing.cpu_count()
    try:
        result = subprocess.run(
            [
                "powershell", "-Command",
                "Get-CimInstance Win32_Processor"
                " | Select-Object Name, NumberOfCores,"
                " NumberOfLogicalProcessors"
                " | ConvertTo-Json",
            ],
            capture_output=True, text=True, timeout=15, shell=False,
        )
        data = _json.loads(result.stdout.strip())
        if isinstance(data, dict):
            data = [data]
        item = data[0]
        name = item.get("Name", name)
        cores = item.get("NumberOfCores", cores) or cores
        threads = item.get("NumberOfLogicalProcessors", threads) or threads
    except Exception:
        pass
    return name, cores, threads


def _detect_ram_gb() -> float:
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass
    os_name = platform.system()
    try:
        if os_name == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 1)
        elif os_name == "Darwin":
            result = subprocess.run(["sysctl", "-n", "hw.memsize"],
                                    capture_output=True, text=True, timeout=5)
            return round(int(result.stdout.strip()) / (1024 ** 3), 1)
        elif os_name == "Windows":
            if shutil.which("wmic"):
                result = subprocess.run(
                    ["wmic", "computersystem", "get", "TotalPhysicalMemory", "/format:csv"],
                    capture_output=True, text=True, timeout=10, shell=False
                )
                for line in result.stdout.strip().splitlines():
                    parts = line.split(",")
                    if len(parts) >= 2 and parts[1].strip().isdigit():
                        return round(int(parts[1].strip()) / (1024 ** 3), 1)
            else:
                return _detect_ram_windows_ps()
    except Exception:
        pass
    return 0.0


def _detect_ram_windows_ps() -> float:
    """Use PowerShell to detect RAM (fallback when wmic is absent)."""
    import json as _json
    try:
        result = subprocess.run(
            [
                "powershell", "-Command",
                "Get-CimInstance Win32_ComputerSystem"
                " | Select-Object TotalPhysicalMemory"
                " | ConvertTo-Json",
            ],
            capture_output=True, text=True, timeout=15, shell=False,
        )
        data = _json.loads(result.stdout.strip())
        if isinstance(data, dict):
            total = data.get("TotalPhysicalMemory", 0)
            if total:
                return round(int(total) / (1024 ** 3), 1)
    except Exception:
        pass
    return 0.0


def detect_hardware() -> HardwareProfile:
    os_name = platform.system()

    # CPU
    if os_name == "Linux":
        cpu_name, cpu_cores, cpu_threads = _detect_cpu_linux()
    elif os_name == "Darwin":
        cpu_name, cpu_cores, cpu_threads = _detect_cpu_macos()
    else:
        cpu_name, cpu_cores, cpu_threads = _detect_cpu_windows()

    # RAM
    ram_gb = _detect_ram_gb()

    # Apple Silicon unified memory detection
    unified = _is_apple_silicon()

    # GPUs
    gpus = _detect_gpus_nvidia()
    if not gpus:
        if os_name == "Darwin":
            if unified:
                # Apple Silicon: treat unified memory as GPU VRAM
                vram_mb = int(ram_gb * 1024)
                gpus = [GPUInfo(name=f"{cpu_name} (Unified Memory)", vram_mb=vram_mb)]
            else:
                gpus = _detect_gpus_macos()
        elif os_name == "Linux":
            gpus = _detect_gpus_amd_linux()
        elif os_name == "Windows":
            if shutil.which("wmic"):
                gpus = _detect_gpus_windows_wmi()
            else:
                gpus = _detect_gpus_windows_ps()

    return HardwareProfile(
        os=os_name,
        cpu_name=cpu_name,
        cpu_cores=cpu_cores,
        cpu_threads=cpu_threads,
        ram_gb=ram_gb,
        gpus=gpus,
        is_unified_memory=unified,
    )
