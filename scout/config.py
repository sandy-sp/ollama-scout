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


def _get_profiles_path() -> str:
    return os.path.join(os.path.dirname(CONFIG_PATH), "profiles.json")


PROFILES_PATH = _get_profiles_path()

_DEFAULT_PROFILES: dict = {"active": "default", "profiles": {"default": {}}}


def _load_profiles() -> dict:
    """Load profiles data from disk."""
    if os.path.exists(PROFILES_PATH):
        try:
            with open(PROFILES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "profiles" in data:
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"active": "default", "profiles": {"default": {}}}


def _save_profiles(data: dict) -> None:
    """Save profiles data to disk."""
    try:
        os.makedirs(os.path.dirname(PROFILES_PATH), exist_ok=True)
        with open(PROFILES_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
    except OSError:
        pass


def list_profiles() -> list[str]:
    """Return list of all profile names."""
    return list(_load_profiles().get("profiles", {}).keys())


def get_active_profile() -> str:
    """Return the name of the currently active profile."""
    return _load_profiles().get("active", "default")


def switch_profile(name: str) -> bool:
    """Set the active profile. Returns False if profile doesn't exist."""
    data = _load_profiles()
    if name not in data.get("profiles", {}):
        return False
    data["active"] = name
    _save_profiles(data)
    return True


def create_profile(name: str, overrides: dict | None = None) -> bool:
    """Create a new named profile. Returns False if name already exists."""
    data = _load_profiles()
    if name in data.get("profiles", {}):
        return False
    if "profiles" not in data:
        data["profiles"] = {}
    data["profiles"][name] = {
        k: v for k, v in (overrides or {}).items() if k in DEFAULT_CONFIG
    }
    _save_profiles(data)
    return True


def delete_profile(name: str) -> bool:
    """Delete a named profile. Returns False if not found or is 'default'."""
    if name == "default":
        return False
    data = _load_profiles()
    if name not in data.get("profiles", {}):
        return False
    del data["profiles"][name]
    if data.get("active") == name:
        data["active"] = "default"
    _save_profiles(data)
    return True


def get_profile_overrides(name: str) -> dict:
    """Return the overrides dict for a named profile."""
    data = _load_profiles()
    return dict(data.get("profiles", {}).get(name, {}))


def set_profile_value(profile_name: str, key: str, value) -> bool:
    """Set a single config key in a named profile. Returns False if not found."""
    if key not in DEFAULT_CONFIG:
        return False
    data = _load_profiles()
    if profile_name not in data.get("profiles", {}):
        return False
    data["profiles"][profile_name][key] = value
    _save_profiles(data)
    return True


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


def load_config(profile: str | None = None) -> dict:
    """Load config from disk, merging with defaults and active profile overrides.

    Args:
        profile: Profile name to apply overrides from. Defaults to active profile.
    """
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

    # Apply profile overrides on top of base config
    active = profile if profile is not None else get_active_profile()
    overrides = get_profile_overrides(active)
    for key in DEFAULT_CONFIG:
        if key in overrides:
            cfg[key] = overrides[key]

    return cfg


def save_config(cfg: dict) -> None:
    """Write base config to disk."""
    try:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
    except OSError:
        pass  # can't write, silently skip


def print_config() -> None:
    """Print current config to stdout."""
    from rich import box
    from rich.console import Console
    from rich.table import Table

    console = Console()
    active = get_active_profile()
    profiles = list_profiles()

    cfg = load_config()
    title = f"[bold cyan]Config[/bold cyan]  [dim]({CONFIG_PATH})[/dim]"
    if active != "default":
        title += f"  [yellow]profile: {active}[/yellow]"

    table = Table(title=title, box=box.ROUNDED, border_style="cyan")
    table.add_column("Key", style="bold white")
    table.add_column("Value", style="white")
    table.add_column("Default", style="dim")

    for key, default in DEFAULT_CONFIG.items():
        current = cfg.get(key, default)
        is_changed = current != default
        val_style = "bold yellow" if is_changed else "white"
        table.add_row(key, f"[{val_style}]{current!r}[/{val_style}]", repr(default))

    console.print(table)

    if profiles:
        profile_list = ", ".join(
            f"[bold]{p}[/bold]" if p == active else p
            for p in profiles
        )
        console.print(f"\n[dim]Profiles:[/dim] {profile_list}")
