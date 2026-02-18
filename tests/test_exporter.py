"""Tests for scout.exporter module."""
import os
import tempfile

from scout.exporter import export_markdown
from scout.hardware import GPUInfo, HardwareProfile
from scout.ollama_api import ModelVariant, OllamaModel
from scout.recommender import Recommendation


def _make_test_data():
    hw = HardwareProfile(
        os="Linux",
        cpu_name="Test CPU",
        cpu_cores=8,
        cpu_threads=16,
        ram_gb=32.0,
        gpus=[GPUInfo(name="Test GPU", vram_mb=10240)],
    )
    model = OllamaModel(
        name="test-model",
        description="A test model",
        tags=[ModelVariant(tag="7b", size_gb=4.0, quantization="Q4_K_M", param_size="7B")],
        use_cases=["chat"],
    )
    rec = Recommendation(
        model=model,
        variant=model.tags[0],
        score=90,
        run_mode="GPU",
        fit_label="Excellent",
        note="Fits fully in VRAM",
    )
    grouped = {"coding": [], "reasoning": [], "chat": [rec]}
    return hw, grouped


class TestExportMarkdown:
    def test_creates_file_at_given_path(self):
        hw, grouped = _make_test_data()
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            result_path = export_markdown(hw, grouped, output_path=tmp_path)
            assert os.path.exists(result_path)
            assert result_path == os.path.abspath(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_file_contains_expected_headers(self):
        hw, grouped = _make_test_data()
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            export_markdown(hw, grouped, output_path=tmp_path)
            with open(tmp_path, encoding="utf-8") as f:
                content = f.read()

            assert "# " in content
            assert "ollama-scout Report" in content
            assert "## " in content
            assert "System Hardware" in content
            assert "Chat Models" in content
            assert "test-model" in content
        finally:
            os.unlink(tmp_path)

    def test_generates_default_filename_when_no_path(self):
        hw, grouped = _make_test_data()
        result_path = export_markdown(hw, grouped)
        try:
            assert os.path.exists(result_path)
            assert "ollama_scout_" in os.path.basename(result_path)
            assert result_path.endswith(".md")
        finally:
            os.unlink(result_path)
