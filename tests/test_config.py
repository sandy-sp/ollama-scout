"""Tests for scout.config module."""
import json
import os
import tempfile
from unittest.mock import patch

from scout.config import DEFAULT_CONFIG, load_config, save_config


class TestLoadConfig:
    def test_returns_defaults_when_no_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            with patch("scout.config.CONFIG_PATH", path):
                cfg = load_config()
                assert cfg == DEFAULT_CONFIG
                # Should have created the file
                assert os.path.exists(path)

    def test_merges_user_values_with_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            with open(path, "w") as f:
                json.dump({"default_top_n": 25, "offline_mode": True}, f)
            with patch("scout.config.CONFIG_PATH", path):
                cfg = load_config()
                assert cfg["default_top_n"] == 25
                assert cfg["offline_mode"] is True
                # Defaults preserved for unset keys
                assert cfg["default_use_case"] == "all"
                assert cfg["auto_export"] is False

    def test_handles_corrupted_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            with open(path, "w") as f:
                f.write("not valid json{{{")
            with patch("scout.config.CONFIG_PATH", path):
                cfg = load_config()
                assert cfg == DEFAULT_CONFIG


class TestSaveConfig:
    def test_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            with patch("scout.config.CONFIG_PATH", path):
                save_config({"default_top_n": 30, "offline_mode": True})
                with open(path) as f:
                    data = json.load(f)
                assert data["default_top_n"] == 30
                assert data["offline_mode"] is True

    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "config.json")
            with patch("scout.config.CONFIG_PATH", path):
                original = dict(DEFAULT_CONFIG)
                original["show_benchmark"] = True
                original["export_dir"] = "/tmp/reports"
                save_config(original)
                loaded = load_config()
                assert loaded == original
