"""Tests for scout.interactive module."""
from unittest.mock import patch

from rich.console import Console

from scout.interactive import TOP_N_MENU, InteractiveSession


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
