"""Tests for scout.interactive module."""
from unittest.mock import patch

from rich.console import Console

from scout.hardware import GPUInfo, HardwareProfile
from scout.interactive import TOP_N_MENU, InteractiveSession
from scout.ollama_api import ModelVariant, OllamaModel
from scout.recommender import Recommendation


def _make_hw(vram_gb=10.0, ram_gb=32.0, unified=False, gpus=None):
    if gpus is None:
        gpus = [GPUInfo(name="Test GPU", vram_mb=int(vram_gb * 1024))] if vram_gb > 0 else []
    return HardwareProfile(
        os="Linux", cpu_name="Test CPU", cpu_cores=8, cpu_threads=16,
        ram_gb=ram_gb, gpus=gpus, is_unified_memory=unified,
    )


def _make_rec(name="test-model", size_gb=4.0):
    variant = ModelVariant(tag="7b", size_gb=size_gb, quantization="Q4_K_M", param_size="7B")
    model = OllamaModel(name=name, description="Test", tags=[variant], use_cases=["chat"])
    return Recommendation(
        model=model, variant=variant, score=90, run_mode="GPU",
        fit_label="Excellent", note="test",
    )


class TestInteractiveSessionInstantiation:
    def test_can_instantiate(self):
        session = InteractiveSession()
        assert session is not None
        assert hasattr(session, "run")


class TestUseCaseMenu:
    def test_valid_choice_coding(self):
        with patch.object(
            Console, "input", side_effect=["2"],
        ), patch.object(Console, "print"):
            result = InteractiveSession._ask_use_case()
            assert result == "coding"

    def test_valid_choice_reasoning(self):
        with patch.object(
            Console, "input", side_effect=["3"],
        ), patch.object(Console, "print"):
            result = InteractiveSession._ask_use_case()
            assert result == "reasoning"

    def test_valid_choice_chat(self):
        with patch.object(
            Console, "input", side_effect=["4"],
        ), patch.object(Console, "print"):
            result = InteractiveSession._ask_use_case()
            assert result == "chat"

    def test_empty_defaults_to_all(self):
        with patch.object(
            Console, "input", side_effect=[""],
        ), patch.object(Console, "print"):
            result = InteractiveSession._ask_use_case()
            assert result == "all"

    def test_invalid_then_valid(self):
        with patch.object(
            Console, "input", side_effect=["5", "2"],
        ), patch.object(Console, "print"):
            result = InteractiveSession._ask_use_case()
            assert result == "coding"


class TestResultsCount:
    def test_valid_choice_2_gives_10(self):
        with patch.object(Console, "input", side_effect=["2"]), \
             patch.object(Console, "print"):
            result = InteractiveSession._ask_results_count()
            assert result == 10

    def test_empty_defaults_to_10(self):
        with patch.object(Console, "input", side_effect=[""]), \
             patch.object(Console, "print"):
            result = InteractiveSession._ask_results_count()
            assert result == 10

    def test_invalid_then_valid(self):
        with patch.object(
            Console, "input", side_effect=["5", "3"],
        ), patch.object(Console, "print"):
            result = InteractiveSession._ask_results_count()
            assert result == 15

    def test_all_valid_options(self):
        expected = {"1": 5, "2": 10, "3": 15, "4": 20}
        for key, val in expected.items():
            with patch.object(
                Console, "input", side_effect=[key],
            ), patch.object(Console, "print"):
                result = InteractiveSession._ask_results_count()
                assert result == val

    def test_top_n_menu_has_four_options(self):
        assert len(TOP_N_MENU) == 4
        assert TOP_N_MENU["1"] == 5
        assert TOP_N_MENU["2"] == 10
        assert TOP_N_MENU["3"] == 15
        assert TOP_N_MENU["4"] == 20


class TestCtrlCHandling:
    def test_keyboard_interrupt_exits_cleanly(self):
        session = InteractiveSession()
        with patch.object(
            session, "_run_steps", side_effect=KeyboardInterrupt,
        ), patch.object(Console, "print"):
            try:
                session.run()
                assert False, "Should have called sys.exit"
            except SystemExit as e:
                assert e.code == 0


class TestOfflineFallback:
    @patch("scout.interactive.check_ollama_installed", return_value=(True, "ollama version 0.5.0"))
    @patch("scout.interactive.get_pulled_models", return_value=[])
    @patch("scout.interactive.get_fallback_models")
    @patch("scout.interactive.detect_hardware")
    def test_uses_fallback_when_user_says_no(
        self, mock_hw, mock_fallback, mock_pulled, mock_ollama,
    ):
        from scout.hardware import HardwareProfile
        from scout.ollama_api import ModelVariant, OllamaModel

        mock_hw.return_value = HardwareProfile(
            os="Linux", cpu_name="Test", cpu_cores=4,
            cpu_threads=8, ram_gb=16.0, gpus=[],
        )
        variant = ModelVariant(
            tag="3b", size_gb=2.0,
            quantization="Q4_K_M", param_size="3B",
        )
        mock_fallback.return_value = [
            OllamaModel(
                name="test-model", description="Test",
                tags=[variant], use_cases=["chat"],
            ),
        ]

        session = InteractiveSession()

        # Mock console.input for all interactive steps:
        # Step 1: Enter (welcome)
        # Step 3: "n" (offline)
        # Step 4: "1" (all categories)
        # Step 5: "" (default 10)
        # Step 6b: "n" (no compare)
        # Step 7: "n" (no benchmark)
        # Step 8: "n" (no export)
        # Step 9: "0" (skip pull)
        inputs = ["", "n", "1", "", "n", "n", "n", "0"]
        with patch.object(Console, "input", side_effect=inputs), \
             patch.object(Console, "print"):
            session.run()

        mock_fallback.assert_called_once()


# ---------------------------------------------------------------------------
# Step-level unit tests
# ---------------------------------------------------------------------------

class TestStepWelcome:
    @patch("scout.interactive.check_ollama_installed", return_value=(True, "ollama version 0.5.0"))
    @patch("scout.interactive.get_pulled_models", return_value=[])
    @patch("scout.interactive.get_fallback_models")
    @patch("scout.interactive.detect_hardware")
    def test_welcome_with_ollama_installed(self, mock_hw, mock_fallback, mock_pulled, mock_ollama):
        mock_hw.return_value = _make_hw()
        mock_fallback.return_value = [
            OllamaModel(
                name="llama3.2", description="Test",
                tags=[ModelVariant(tag="3b", size_gb=2.0, quantization="Q4_K_M", param_size="3B")],
                use_cases=["chat"],
            )
        ]
        session = InteractiveSession()
        inputs = ["", "n", "1", "", "n", "n", "n", "0"]
        with patch.object(Console, "input", side_effect=inputs), \
             patch.object(Console, "print") as mock_print:
            session.run()
        # Verify it ran without error
        assert mock_print.called

    @patch("scout.interactive.check_ollama_installed", return_value=(False, ""))
    @patch("scout.interactive.get_pulled_models", return_value=[])
    @patch("scout.interactive.get_fallback_models")
    @patch("scout.interactive.detect_hardware")
    def test_welcome_without_ollama_skips_pull(
        self, mock_hw, mock_fallback, mock_pulled, mock_ollama,
    ):
        mock_hw.return_value = _make_hw()
        mock_fallback.return_value = [
            OllamaModel(
                name="llama3.2", description="Test",
                tags=[ModelVariant(tag="3b", size_gb=2.0, quantization="Q4_K_M", param_size="3B")],
                use_cases=["chat"],
            )
        ]
        session = InteractiveSession()
        # Without ollama, step 9 (pull) is skipped — only 7 inputs needed
        inputs = ["", "n", "1", "", "n", "n", "n"]
        with patch.object(Console, "input", side_effect=inputs), \
             patch.object(Console, "print"):
            session.run()


class TestHardwareScanContextMessages:
    def test_gpu_found_message(self):
        hw = _make_hw(vram_gb=10.0)
        # Just test that detect_hardware returns gpu context
        with patch("scout.interactive.detect_hardware", return_value=hw), \
             patch.object(Console, "print"), \
             patch.object(Console, "rule"):
            result = InteractiveSession().__class__._step_hardware_scan(
                InteractiveSession()
            )
        assert result.gpus

    def test_no_gpu_message_branches(self):
        """Directly test _step_hardware_scan with no-GPU hardware."""
        hw_no_gpu = _make_hw(vram_gb=0)
        with patch("scout.interactive.detect_hardware", return_value=hw_no_gpu), \
             patch.object(Console, "print"), \
             patch.object(Console, "rule"):
            result = InteractiveSession()._step_hardware_scan()
        assert result.gpus == []

    def test_apple_silicon_message(self):
        """Test Apple Silicon branch in hardware scan step."""
        hw_apple = HardwareProfile(
            os="Darwin", cpu_name="Apple M2", cpu_cores=8, cpu_threads=8,
            ram_gb=16.0,
            gpus=[GPUInfo(name="Apple M2 (Unified Memory)", vram_mb=16384)],
            is_unified_memory=True,
        )
        with patch("scout.interactive.detect_hardware", return_value=hw_apple), \
             patch.object(Console, "print"), \
             patch.object(Console, "rule"):
            result = InteractiveSession()._step_hardware_scan()
        assert result.is_unified_memory is True


class TestStepCompare:
    def test_compare_skipped_on_no(self):
        hw = _make_hw()
        with patch.object(Console, "input", side_effect=["n"]), \
             patch.object(Console, "print"):
            InteractiveSession._step_compare([], hw, [])

    @patch("scout.interactive.print_model_comparison")
    def test_compare_runs_with_valid_models(self, mock_compare):
        variant = ModelVariant(tag="7b", size_gb=4.0, quantization="Q4_K_M", param_size="7B")
        model1 = OllamaModel(
            name="llama3.2", description="Test", tags=[variant], use_cases=["chat"],
        )
        model2 = OllamaModel(
            name="mistral", description="Test", tags=[variant], use_cases=["chat"],
        )
        hw = _make_hw()
        inputs = ["y", "llama3.2", "mistral"]
        with patch.object(Console, "input", side_effect=inputs), \
             patch.object(Console, "print"):
            InteractiveSession._step_compare([model1, model2], hw, [])
        mock_compare.assert_called_once()

    def test_compare_handles_missing_model(self):
        hw = _make_hw()
        inputs = ["y", "nonexistent", "alsonotfound"]
        with patch.object(Console, "input", side_effect=inputs), \
             patch.object(Console, "print"):
            InteractiveSession._step_compare([], hw, [])

    def test_compare_skips_when_empty_names(self):
        hw = _make_hw()
        inputs = ["y", "", ""]
        with patch.object(Console, "input", side_effect=inputs), \
             patch.object(Console, "print"):
            InteractiveSession._step_compare([], hw, [])


class TestStepBenchmark:
    def test_benchmark_skipped_on_no(self):
        hw = _make_hw()
        recs = [_make_rec()]
        with patch.object(Console, "input", side_effect=["n"]), \
             patch.object(Console, "print"):
            InteractiveSession._step_benchmark(recs, hw, [])

    def test_benchmark_message_when_no_pulled_models(self):
        hw = _make_hw()
        recs = [_make_rec()]
        with patch.object(Console, "input", side_effect=["y"]), \
             patch.object(Console, "print") as mock_print:
            InteractiveSession._step_benchmark(recs, hw, [])
        # Should print "no models" message
        assert mock_print.called

    @patch("scout.interactive.print_benchmark")
    @patch("scout.interactive.benchmark_pulled_models")
    def test_benchmark_runs_with_pulled_models(self, mock_bench, mock_print):
        from scout.benchmark import BenchmarkEstimate
        mock_bench.return_value = [
            BenchmarkEstimate(model_name="llama3.2:3b", run_mode="GPU",
                              tokens_per_sec=80.0, rating="Fast")
        ]
        hw = _make_hw()
        recs = [_make_rec()]
        with patch.object(Console, "input", side_effect=["y"]), \
             patch.object(Console, "print"):
            InteractiveSession._step_benchmark(recs, hw, ["llama3.2"])
        mock_print.assert_called_once()

    @patch("scout.interactive.benchmark_pulled_models", return_value=[])
    def test_benchmark_handles_empty_results(self, mock_bench):
        hw = _make_hw()
        recs = [_make_rec()]
        with patch.object(Console, "input", side_effect=["y"]), \
             patch.object(Console, "print"):
            InteractiveSession._step_benchmark(recs, hw, ["llama3.2"])


class TestStepExport:
    @patch("scout.interactive.export_markdown", return_value="/tmp/report.md")
    def test_export_runs_on_yes_with_blank_path(self, mock_export):
        hw = _make_hw()
        recs = [_make_rec()]
        inputs = ["y", ""]
        with patch.object(Console, "input", side_effect=inputs), \
             patch.object(Console, "print"):
            InteractiveSession._step_export(hw, recs)
        mock_export.assert_called_once()

    @patch("scout.interactive.export_markdown", return_value="/tmp/report.md")
    def test_export_uses_custom_path(self, mock_export):
        hw = _make_hw()
        recs = [_make_rec()]
        inputs = ["y", "/tmp/myreport.md"]
        with patch.object(Console, "input", side_effect=inputs), \
             patch.object(Console, "print"):
            InteractiveSession._step_export(hw, recs)
        _, kwargs = mock_export.call_args
        assert kwargs.get("output_path") or mock_export.call_args[0]

    def test_export_skipped_on_no(self):
        hw = _make_hw()
        recs = [_make_rec()]
        with patch.object(Console, "input", side_effect=["n"]), \
             patch.object(Console, "print"):
            InteractiveSession._step_export(hw, recs)


class TestStepPull:
    @patch("scout.interactive.pull_model")
    def test_pull_success_returns_model_name(self, mock_pull):
        rec = _make_rec()
        with patch.object(Console, "input", side_effect=["1"]), \
             patch.object(Console, "print"):
            result = InteractiveSession._step_pull([rec])
        assert result == "test-model:7b"
        mock_pull.assert_called_once_with("test-model:7b")

    def test_pull_skip_on_zero(self):
        rec = _make_rec()
        with patch.object(Console, "input", side_effect=["0"]), \
             patch.object(Console, "print"):
            result = InteractiveSession._step_pull([rec])
        assert result is None

    def test_pull_skip_on_empty(self):
        rec = _make_rec()
        with patch.object(Console, "input", side_effect=[""]), \
             patch.object(Console, "print"):
            result = InteractiveSession._step_pull([rec])
        assert result is None

    @patch("scout.interactive.pull_model", side_effect=FileNotFoundError("not found"))
    def test_pull_handles_file_not_found(self, mock_pull):
        rec = _make_rec()
        with patch.object(Console, "input", side_effect=["1"]), \
             patch.object(Console, "print"):
            result = InteractiveSession._step_pull([rec])
        assert result is None

    @patch("scout.interactive.pull_model", side_effect=Exception("pull failed"))
    def test_pull_handles_generic_error(self, mock_pull):
        rec = _make_rec()
        with patch.object(Console, "input", side_effect=["1"]), \
             patch.object(Console, "print"):
            result = InteractiveSession._step_pull([rec])
        assert result is None
