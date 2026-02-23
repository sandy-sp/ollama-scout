"""Tests for config profiles feature in scout.config."""
import os
import tempfile
from unittest.mock import patch

from scout.config import (
    DEFAULT_CONFIG,
    create_profile,
    delete_profile,
    get_active_profile,
    get_profile_overrides,
    list_profiles,
    load_config,
    set_profile_value,
    switch_profile,
)


def _patch_paths(tmpdir):
    """Returns a context manager patching both CONFIG_PATH and PROFILES_PATH."""
    config_path = os.path.join(tmpdir, "ollama-scout", "config.json")
    profiles_path = os.path.join(tmpdir, "ollama-scout", "profiles.json")
    return (
        patch("scout.config.CONFIG_PATH", config_path),
        patch("scout.config.PROFILES_PATH", profiles_path),
        patch("scout.config.LEGACY_CONFIG_PATH", "/nonexistent"),
    )


class TestListProfiles:
    def test_default_profile_always_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                profiles = list_profiles()
                assert "default" in profiles

    def test_lists_created_profiles(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                create_profile("coding")
                create_profile("quick")
                profiles = list_profiles()
                assert "coding" in profiles
                assert "quick" in profiles
                assert "default" in profiles


class TestGetActiveProfile:
    def test_default_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                assert get_active_profile() == "default"

    def test_returns_switched_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                create_profile("work")
                switch_profile("work")
                assert get_active_profile() == "work"


class TestSwitchProfile:
    def test_switch_to_existing_profile_returns_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                create_profile("fast")
                assert switch_profile("fast") is True
                assert get_active_profile() == "fast"

    def test_switch_to_nonexistent_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                assert switch_profile("nonexistent") is False
                assert get_active_profile() == "default"


class TestCreateProfile:
    def test_creates_empty_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                result = create_profile("myprofile")
                assert result is True
                assert "myprofile" in list_profiles()
                assert get_profile_overrides("myprofile") == {}

    def test_creates_profile_with_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                result = create_profile("heavy", {"default_top_n": 20, "show_benchmark": True})
                assert result is True
                overrides = get_profile_overrides("heavy")
                assert overrides["default_top_n"] == 20
                assert overrides["show_benchmark"] is True

    def test_duplicate_create_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                create_profile("dup")
                assert create_profile("dup") is False

    def test_ignores_unknown_override_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                create_profile("clean", {"default_top_n": 5, "unknown_key": "value"})
                overrides = get_profile_overrides("clean")
                assert "unknown_key" not in overrides
                assert overrides.get("default_top_n") == 5


class TestDeleteProfile:
    def test_delete_existing_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                create_profile("temp")
                assert delete_profile("temp") is True
                assert "temp" not in list_profiles()

    def test_cannot_delete_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                assert delete_profile("default") is False
                assert "default" in list_profiles()

    def test_delete_nonexistent_returns_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                assert delete_profile("ghost") is False

    def test_delete_active_switches_to_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                create_profile("active_one")
                switch_profile("active_one")
                assert get_active_profile() == "active_one"
                delete_profile("active_one")
                assert get_active_profile() == "default"


class TestSetProfileValue:
    def test_sets_valid_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                create_profile("tweaked")
                assert set_profile_value("tweaked", "default_top_n", 20) is True
                assert get_profile_overrides("tweaked")["default_top_n"] == 20

    def test_rejects_unknown_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                create_profile("p")
                assert set_profile_value("p", "not_a_key", "x") is False

    def test_rejects_nonexistent_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                assert set_profile_value("ghost", "default_top_n", 5) is False


class TestLoadConfigWithProfile:
    def test_profile_overrides_base_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                create_profile("custom", {"default_top_n": 5, "offline_mode": True})
                cfg = load_config(profile="custom")
                assert cfg["default_top_n"] == 5
                assert cfg["offline_mode"] is True
                # Non-overridden keys remain as defaults
                assert cfg["auto_export"] is False

    def test_default_profile_no_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                cfg = load_config(profile="default")
                assert cfg == DEFAULT_CONFIG

    def test_active_profile_applied_when_no_profile_arg(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                create_profile("fast", {"default_top_n": 5})
                switch_profile("fast")
                cfg = load_config()  # no profile arg — should use active
                assert cfg["default_top_n"] == 5

    def test_explicit_profile_overrides_active(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                create_profile("a", {"default_top_n": 5})
                create_profile("b", {"default_top_n": 20})
                switch_profile("a")
                # Explicitly request "b" profile
                cfg = load_config(profile="b")
                assert cfg["default_top_n"] == 20

    def test_unknown_profile_falls_back_to_base(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p1, p2, p3 = _patch_paths(tmpdir)
            with p1, p2, p3:
                cfg = load_config(profile="nonexistent")
                assert cfg == DEFAULT_CONFIG
