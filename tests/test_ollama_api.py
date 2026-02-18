"""Tests for scout.ollama_api module."""
from unittest.mock import MagicMock, patch

from scout.ollama_api import (
    FALLBACK_MODELS,
    _generate_description,
    _infer_use_cases,
    _parse_param_size,
    _parse_param_size_from_name_and_tag,
    _parse_quantization,
    fetch_ollama_models,
    get_fallback_models,
    get_pulled_models,
)


class TestInferUseCases:
    def test_coding_model(self):
        cases = _infer_use_cases("deepseek-coder")
        assert "coding" in cases

    def test_reasoning_model(self):
        cases = _infer_use_cases("deepseek-r1")
        assert "reasoning" in cases

    def test_chat_model(self):
        cases = _infer_use_cases("llama3.2")
        assert "chat" in cases

    def test_unknown_defaults_to_chat(self):
        cases = _infer_use_cases("totally-unknown-model")
        assert cases == ["chat"]

    def test_multi_use_case(self):
        # phi4 appears in both reasoning and chat
        cases = _infer_use_cases("phi4")
        assert "reasoning" in cases
        assert "chat" in cases


class TestParseParamSize:
    def test_parses_7b(self):
        assert _parse_param_size("7b") == "7B"

    def test_parses_from_tag(self):
        assert _parse_param_size("13b-q4_0") == "13B"

    def test_parses_decimal(self):
        assert _parse_param_size("6.7b") == "6.7B"

    def test_returns_unknown_for_no_match(self):
        assert _parse_param_size("latest") == "?"

    def test_from_name_and_tag_prefers_tag(self):
        result = _parse_param_size_from_name_and_tag("llama3", "7b")
        assert result == "7B"

    def test_from_name_and_tag_falls_back_to_name(self):
        result = _parse_param_size_from_name_and_tag("model7b", "latest")
        assert result == "7B"


class TestParseQuantization:
    def test_detects_q4_0(self):
        assert _parse_quantization("7b-q4_0") == "Q4_0"

    def test_detects_f16(self):
        assert _parse_quantization("7b-f16") == "F16"

    def test_detects_q4_k_m(self):
        assert _parse_quantization("7b-q4_k_m") == "Q4_K_M"

    def test_instruct_defaults_to_q4_k_m(self):
        assert _parse_quantization("7b-instruct") == "Q4_K_M"

    def test_unknown_defaults_to_q4_0(self):
        assert _parse_quantization("latest") == "Q4_0"


class TestGenerateDescription:
    def test_known_model(self):
        desc = _generate_description("llama3.2", ["chat"])
        assert "Llama 3.2" in desc

    def test_unknown_model_uses_use_case(self):
        desc = _generate_description("my-custom-model", ["coding"])
        assert "code" in desc.lower()

    def test_completely_unknown(self):
        desc = _generate_description("xyz123", [])
        assert "xyz123" in desc


class TestGetFallbackModels:
    def test_returns_correct_count(self):
        models = get_fallback_models()
        assert len(models) == len(FALLBACK_MODELS)

    def test_models_have_variants(self):
        models = get_fallback_models()
        for m in models:
            assert len(m.tags) >= 1
            assert m.tags[0].size_gb > 0

    def test_models_have_use_cases(self):
        models = get_fallback_models()
        for m in models:
            assert len(m.use_cases) >= 1


class TestFetchOllamaModels:
    @patch("scout.ollama_api.requests.get")
    def test_parses_api_response(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {
                    "name": "llama3.2:3b",
                    "size": 2147483648,
                    "details": {},
                },
                {
                    "name": "mistral:7b",
                    "size": 4402341478,
                    "details": {
                        "parameter_size": "7B",
                        "quantization_level": "Q4_K_M",
                    },
                },
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        models = fetch_ollama_models()
        assert len(models) == 2
        assert models[0].name == "llama3.2"
        assert models[0].tags[0].tag == "3b"
        assert models[1].name == "mistral"
        assert models[1].tags[0].param_size == "7B"

    @patch("scout.ollama_api.requests.get")
    def test_fills_gaps_when_details_empty(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "deepseek-coder:6.7b", "size": 0, "details": {}},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        models = fetch_ollama_models()
        m = models[0]
        assert m.description  # gap-filled
        assert m.tags[0].param_size == "6.7B"  # parsed from tag
        assert m.tags[0].quantization  # inferred

    @patch("scout.ollama_api.requests.get")
    def test_raises_on_empty_response(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        try:
            fetch_ollama_models()
            assert False, "Should have raised ConnectionError"
        except ConnectionError:
            pass


class TestGetPulledModels:
    @patch("scout.ollama_api.shutil.which", return_value=None)
    def test_returns_empty_when_ollama_not_installed(self, mock_which):
        assert get_pulled_models() == []

    @patch("scout.ollama_api.subprocess.run")
    @patch("scout.ollama_api.shutil.which", return_value="/usr/bin/ollama")
    def test_parses_ollama_list_output(self, mock_which, mock_run):
        mock_run.return_value = MagicMock(
            stdout="NAME           ID          SIZE     MODIFIED\n"
                   "llama3.2:3b    abc123      2.0 GB   2 days ago\n"
                   "mistral:7b     def456      4.1 GB   5 days ago\n",
            returncode=0,
        )
        pulled = get_pulled_models()
        assert "llama3.2" in pulled
        assert "mistral" in pulled
