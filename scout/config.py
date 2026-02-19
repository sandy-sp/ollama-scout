"""
config.py - Persistent configuration with platform-aware config paths.
"""
import json
import os
import platform
import shutil

LEGACY_CONFIG_PATH = os.path.expanduser("~/.ollama-scout.json")


def _get_config_path() -> str:
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        return os.path.join(base, "ollama-scout", "config.json")
    elif system == "Darwin":
        return os.path.expanduser(
            "~/Library/Application Support/ollama-scout/config.json"
        )
    else:  # Linux and others
        xdg = os.environ.get(
            "XDG_CONFIG_HOME", os.path.expanduser("~/.config")
        )
        return os.path.join(xdg, "ollama-scout", "config.json")


CONFIG_PATH = _get_config_path()

DEFAULT_CONFIG: dict = {
    "default_use_case": "all",
    "default_top_n": 15,
    "auto_export": False,
    "export_dir": "",
    "offline_mode": False,
    "show_benchmark": False,
}


def _migrate_legacy_config() -> bool:
    """Migrate legacy ~/.ollama-scout.json to new XDG path if needed.

    Returns True if migration occurred.
    """
    if not os.path.exists(LEGACY_CONFIG_PATH):
        return False
    if os.path.exists(CONFIG_PATH):
        return False
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        shutil.copy2(LEGACY_CONFIG_PATH, CONFIG_PATH)
        os.remove(LEGACY_CONFIG_PATH)
        return True
    except OSError:
        return False


def load_config() -> dict:
    """Load config from disk, merging with defaults. Creates file on first run."""
    migrated = _migrate_legacy_config()
    if migrated:
        from .display import print_info
        print_info(f"Config migrated to [bold]{CONFIG_PATH}[/bold]")

    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            if isinstance(user_cfg, dict):
                for key in DEFAULT_CONFIG:
                    if key in user_cfg:
                        cfg[key] = user_cfg[key]
        except (json.JSONDecodeError, OSError):
            pass  # corrupted or unreadable, use defaults
    else:
        save_config(cfg)
    return cfg


def save_config(cfg: dict) -> None:
    """Write config to disk."""
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
    except OSError:
        pass  # can't write, silently skip


def print_config() -> None:
    """Print current config to stdout."""
    cfg = load_config()
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(
        title=f"[bold cyan]Config[/bold cyan]  [dim]({CONFIG_PATH})[/dim]",
        box=box.ROUNDED,
        border_style="cyan",
    )
    table.add_column("Key", style="bold white")
    table.add_column("Value", style="white")
    table.add_column("Default", style="dim")

    for key, default in DEFAULT_CONFIG.items():
        current = cfg.get(key, default)
        is_changed = current != default
        val_style = "bold yellow" if is_changed else "white"
        table.add_row(key, f"[{val_style}]{current!r}[/{val_style}]", repr(default))

    console.print(table)
