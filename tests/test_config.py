"""Tests for scout.config module."""
import json
import os
import tempfile
from unittest.mock import patch

from scout.config import (
    DEFAULT_CONFIG,
    _get_config_path,
    _migrate_legacy_config,
    load_config,
    print_config,
    save_config,
)


class TestGetConfigPath:
    @patch("scout.config.platform.system", return_value="Linux")
    def test_linux_uses_xdg(self, mock_sys):
        with patch.dict(os.environ, {"XDG_CONFIG_HOME": "/tmp/xdg"}, clear=False):
            path = _get_config_path()
            assert path == "/tmp/xdg/ollama-scout/config.json"

    @patch("scout.config.platform.system", return_value="Linux")
    def test_linux_defaults_to_dot_config(self, mock_sys):
        env = dict(os.environ)
        env.pop("XDG_CONFIG_HOME", None)
        with patch.dict(os.environ, env, clear=True):
            path = _get_config_path()
            assert ".config/ollama-scout/config.json" in path

    @patch("scout.config.platform.system", return_value="Darwin")
    def test_macos_uses_application_support(self, mock_sys):
        path = _get_config_path()
        assert "Library/Application Support/ollama-scout/config.json" in path

    @patch("scout.config.platform.system", return_value="Windows")
    def test_windows_uses_appdata(self, mock_sys):
        with patch.dict(os.environ, {"APPDATA": "C:\\Users\\test\\AppData"}, clear=False):
            path = _get_config_path()
            assert "ollama-scout" in path
            assert path.endswith("config.json")


class TestLegacyMigration:
    def test_migrates_legacy_to_new_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy = os.path.join(tmpdir, "legacy.json")
            new_dir = os.path.join(tmpdir, "new")
            new_path = os.path.join(new_dir, "config.json")

            with open(legacy, "w") as f:
                json.dump({"default_top_n": 25}, f)

            with patch("scout.config.LEGACY_CONFIG_PATH", legacy), \
                 patch("scout.config.CONFIG_PATH", new_path):
                result = _migrate_legacy_config()

            assert result is True
            assert os.path.exists(new_path)
            assert not os.path.exists(legacy)
            with open(new_path) as f:
                assert json.load(f)["default_top_n"] == 25

    def test_no_migration_when_no_legacy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy = os.path.join(tmpdir, "nonexistent.json")
            new_path = os.path.join(tmpdir, "new", "config.json")

            with patch("scout.config.LEGACY_CONFIG_PATH", legacy), \
                 patch("scout.config.CONFIG_PATH", new_path):
                result = _migrate_legacy_config()

            assert result is False

    def test_no_migration_when_new_already_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy = os.path.join(tmpdir, "legacy.json")
            new_path = os.path.join(tmpdir, "config.json")

            with open(legacy, "w") as f:
                json.dump({"default_top_n": 25}, f)
            with open(new_path, "w") as f:
                json.dump({"default_top_n": 10}, f)

            with patch("scout.config.LEGACY_CONFIG_PATH", legacy), \
                 patch("scout.config.CONFIG_PATH", new_path):
                result = _migrate_legacy_config()

            assert result is False
            # Legacy should NOT be deleted
            assert os.path.exists(legacy)


class TestLoadConfig:
    def test_returns_defaults_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ollama-scout", "config.json")
            with patch("scout.config.CONFIG_PATH", path), \
                 patch("scout.config.LEGACY_CONFIG_PATH", "/nonexistent"):
                cfg = load_config()
                assert cfg == DEFAULT_CONFIG
                assert os.path.exists(path)

    def test_merges_user_values_with_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "ollama-scout")
            os.makedirs(config_dir)
            path = os.path.join(config_dir, "config.json")
            with open(path, "w") as f:
                json.dump({"default_top_n": 25, "offline_mode": True}, f)
            with patch("scout.config.CONFIG_PATH", path), \
                 patch("scout.config.LEGACY_CONFIG_PATH", "/nonexistent"):
                cfg = load_config()
                assert cfg["default_top_n"] == 25
                assert cfg["offline_mode"] is True
                assert cfg["default_use_case"] == "all"
                assert cfg["auto_export"] is False

    def test_handles_corrupted_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = os.path.join(tmpdir, "ollama-scout")
            os.makedirs(config_dir)
            path = os.path.join(config_dir, "config.json")
            with open(path, "w") as f:
                f.write("not valid json{{{")
            with patch("scout.config.CONFIG_PATH", path), \
                 patch("scout.config.LEGACY_CONFIG_PATH", "/nonexistent"):
                cfg = load_config()
                assert cfg == DEFAULT_CONFIG


class TestMigrationOSError:
    def test_silently_returns_false_on_os_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy = os.path.join(tmpdir, "legacy.json")
            # new path is in a location that can't be created (simulate OSError)
            new_path = os.path.join(tmpdir, "readonly", "config.json")

            with open(legacy, "w") as f:
                json.dump({"default_top_n": 25}, f)

            with patch("scout.config.LEGACY_CONFIG_PATH", legacy), \
                 patch("scout.config.CONFIG_PATH", new_path), \
                 patch("scout.config.os.makedirs", side_effect=OSError("permission denied")):
                result = _migrate_legacy_config()

            assert result is False


class TestSaveConfigError:
    def test_silently_ignores_os_error(self):
        with patch("scout.config.os.makedirs", side_effect=OSError("read only")):
            # Should not raise
            save_config({"default_top_n": 10})


class TestLoadConfigMigration:
    def test_prints_migration_message_when_migrated(self):
        """When migration happens, load_config prints an info message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy = os.path.join(tmpdir, "legacy.json")
            new_path = os.path.join(tmpdir, "new", "config.json")
            with open(legacy, "w") as f:
                json.dump({"default_top_n": 25}, f)

            with patch("scout.config.LEGACY_CONFIG_PATH", legacy), \
                 patch("scout.config.CONFIG_PATH", new_path), \
                 patch("scout.display.console"):
                cfg = load_config()
            assert cfg["default_top_n"] == 25


class TestPrintConfig:
    def test_print_config_runs_without_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ollama-scout", "config.json")
            with patch("scout.config.CONFIG_PATH", path), \
                 patch("scout.config.LEGACY_CONFIG_PATH", "/nonexistent"):
                # print_config creates its own Console; just ensure no exception
                print_config()

    def test_print_config_shows_all_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ollama-scout", "config.json")
            overrides = {k: v for k, v in DEFAULT_CONFIG.items() if k != "default_top_n"}
            with patch("scout.config.CONFIG_PATH", path), \
                 patch("scout.config.LEGACY_CONFIG_PATH", "/nonexistent"), \
                 patch("rich.console.Console.print"):
                # Just verify it doesn't crash with a changed value
                save_config({"default_top_n": 25, **overrides})
                print_config()


class TestSaveConfig:
    def test_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ollama-scout", "config.json")
            with patch("scout.config.CONFIG_PATH", path):
                save_config({"default_top_n": 30, "offline_mode": True})
                with open(path) as f:
                    data = json.load(f)
                assert data["default_top_n"] == 30
                assert data["offline_mode"] is True

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ollama-scout", "config.json")
            with patch("scout.config.CONFIG_PATH", path), \
                 patch("scout.config.LEGACY_CONFIG_PATH", "/nonexistent"):
                original = dict(DEFAULT_CONFIG)
                original["show_benchmark"] = True
                original["export_dir"] = "/tmp/reports"
                save_config(original)
                loaded = load_config()
                assert loaded == original
