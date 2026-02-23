"""
doctor.py - System health check for ollama-scout.
"""
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

from rich import box
from rich.console import Console
from rich.table import Table

console = Console()


def _check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = v >= (3, 10)
    return ok, f"{v.major}.{v.minor}.{v.micro}"


def _check_ollama() -> tuple[bool, str]:
    import shutil
    if not shutil.which("ollama"):
        return False, "not found in PATH"
    try:
        result = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
        return False, "command returned non-zero"
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)


def _check_gpu() -> tuple[bool, str]:
    try:
        from .hardware import detect_hardware
        hw = detect_hardware()
        if hw.is_unified_memory:
            return True, f"Apple Silicon — {hw.ram_gb:.0f} GB unified memory"
        elif hw.gpus:
            names = ", ".join(g.name for g in hw.gpus)
            total_vram = sum(g.vram_mb for g in hw.gpus) / 1024
            return True, f"{names} ({total_vram:.1f} GB VRAM)"
        else:
            return False, "no GPU detected (CPU-only mode)"
    except Exception as e:
        return False, f"detection error: {e}"


def _check_ram() -> tuple[bool, str]:
    ram_gb: float | None = None
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        system = platform.system()
        if system == "Linux":
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            ram_gb = int(line.split()[1]) / (1024 ** 2)
                            break
            except OSError:
                pass
        if ram_gb is None:
            return False, "psutil not available"
    ok = (ram_gb or 0.0) >= 4.0
    return ok, f"{ram_gb:.1f} GB"


def _check_internet() -> tuple[bool, str]:
    try:
        import socket
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("8.8.8.8", 53))
        return True, "reachable"
    except OSError:
        return False, "no internet connection"


def _check_model_cache() -> tuple[bool, str]:
    try:
        from .ollama_api import _get_cache_path
        cache_path = _get_cache_path()
    except ImportError:
        return False, "cache module unavailable"

    if not os.path.exists(cache_path):
        return False, "no cache file (will fetch on first run)"

    try:
        mtime = os.path.getmtime(cache_path)
        age_hours = (datetime.now(timezone.utc).timestamp() - mtime) / 3600
        size = os.path.getsize(cache_path)
        if age_hours < 24:
            return True, f"fresh ({age_hours:.1f}h old, {size // 1024} KB)"
        else:
            return False, f"stale ({age_hours:.1f}h old) — run --update-models to refresh"
    except OSError as e:
        return False, str(e)


def _check_config() -> tuple[bool, str]:
    try:
        from .config import CONFIG_PATH, DEFAULT_CONFIG, load_config
        cfg = load_config()
        if not os.path.exists(CONFIG_PATH):
            return True, "using defaults (no config file yet)"
        unknown = [k for k in cfg if k not in DEFAULT_CONFIG]
        if unknown:
            return False, f"unknown keys: {', '.join(unknown)}"
        return True, f"valid ({CONFIG_PATH})"
    except Exception as e:
        return False, f"error reading config: {e}"


def _check_pulled_models() -> tuple[bool, str]:
    try:
        from .ollama_api import get_pulled_models
        pulled = get_pulled_models()
        if pulled:
            names = ", ".join(pulled[:5]) + ("..." if len(pulled) > 5 else "")
            return True, f"{len(pulled)} pulled: {names}"
        return False, "no models pulled yet"
    except Exception as e:
        return False, f"error: {e}"


_CHECKS = [
    ("Python ≥ 3.10", _check_python),
    ("Ollama binary", _check_ollama),
    ("GPU / VRAM", _check_gpu),
    ("RAM (≥ 4 GB)", _check_ram),
    ("Internet", _check_internet),
    ("Model cache", _check_model_cache),
    ("Config file", _check_config),
    ("Pulled models", _check_pulled_models),
]


def run_doctor() -> None:
    """Run all health checks and print a summary table."""
    table = Table(
        title="[bold cyan]ollama-scout Doctor[/bold cyan]",
        box=box.ROUNDED,
        border_style="cyan",
    )
    table.add_column("Check", style="bold white", min_width=20)
    table.add_column("Status", justify="center", min_width=6)
    table.add_column("Details", style="dim")

    all_ok = True
    for label, check_fn in _CHECKS:
        ok, detail = check_fn()
        if ok:
            status = "[bold green]✓  OK[/bold green]"
        else:
            status = "[bold yellow]⚠  WARN[/bold yellow]"
            all_ok = False
        table.add_row(label, status, detail)

    console.print()
    console.print(table)
    console.print()
    if all_ok:
        console.print("[bold green]All checks passed.[/bold green]")
    else:
        console.print(
            "[yellow]Some checks returned warnings. "
            "See details above.[/yellow]"
        )
    console.print()
