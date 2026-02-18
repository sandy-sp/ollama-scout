"""
ollama_api.py - Fetch available models from Ollama library + detect locally pulled models.
"""
import re
import shutil
import subprocess
from dataclasses import dataclass, field

import requests

OLLAMA_API_URL = "https://ollama.com/api/tags"

REQUEST_TIMEOUT = 15


@dataclass
class ModelVariant:
    tag: str           # e.g. "7b-q4_0"
    size_gb: float
    quantization: str  # e.g. "Q4_0", "Q8_0", "F16"
    param_size: str    # e.g. "7B", "13B"


@dataclass
class OllamaModel:
    name: str
    description: str
    tags: list[ModelVariant] = field(default_factory=list)
    use_cases: list[str] = field(default_factory=list)  # ["coding", "chat", "reasoning"]
    pulled: bool = False


# Static use-case mapping since Ollama API doesn't categorize by use case
USE_CASE_MAP: dict[str, list[str]] = {
    "coding": [
        "codellama", "deepseek-coder", "codegemma", "starcoder", "starcoder2",
        "codestral", "qwen2.5-coder", "qwen3-coder", "granite-code", "magicoder",
    ],
    "reasoning": [
        "deepseek-r1", "qwq", "phi4", "phi3", "llama3.3", "llama3.1",
        "mistral-large", "command-r-plus", "mixtral", "deepseek-v3",
    ],
    "chat": [
        "llama3.2", "llama3.1", "llama3", "mistral", "gemma2", "gemma3", "gemma",
        "qwen2.5", "qwen3", "phi3", "phi4", "smollm2", "hermes3", "openhermes",
        "neural-chat", "dolphin-mixtral", "llava", "bakllava",
    ],
}

# Descriptions for known model families when the API returns none
_MODEL_DESCRIPTIONS: dict[str, str] = {
    "llama": "Meta's open-weight large language model",
    "llama3": "Meta's Llama 3 general-purpose model",
    "llama3.1": "Meta's Llama 3.1 with extended context support",
    "llama3.2": "Meta's compact and efficient Llama 3.2 model",
    "llama3.3": "Meta's Llama 3.3 flagship model",
    "mistral": "Mistral AI's efficient base model",
    "mixtral": "Mistral AI's sparse mixture-of-experts model",
    "mistral-large": "Mistral AI's largest and most capable model",
    "codellama": "Meta's code-specialized Llama model",
    "deepseek-coder": "DeepSeek's model optimized for code generation",
    "deepseek-r1": "DeepSeek's reasoning-focused model",
    "deepseek-v3": "DeepSeek V3 large language model",
    "deepseek-v3.1": "DeepSeek V3.1 improved large language model",
    "deepseek-v3.2": "DeepSeek V3.2 latest large language model",
    "phi3": "Microsoft's compact and efficient Phi-3 model",
    "phi4": "Microsoft's Phi-4 reasoning model",
    "gemma": "Google's lightweight open model",
    "gemma2": "Google's Gemma 2 open model",
    "gemma3": "Google's latest Gemma 3 multimodal model",
    "qwen2.5": "Alibaba's multilingual Qwen 2.5 model",
    "qwen2.5-coder": "Alibaba's code-specialized Qwen model",
    "qwen3": "Alibaba's Qwen 3 next-generation model",
    "qwen3-coder": "Alibaba's Qwen 3 code-specialized model",
    "smollm2": "HuggingFace's ultra-compact language model",
    "starcoder": "BigCode's code generation model",
    "starcoder2": "BigCode's improved code generation model",
    "codegemma": "Google's code-specialized Gemma model",
    "codestral": "Mistral's code-specialized model",
    "command-r-plus": "Cohere's enterprise command model",
    "llava": "Large Language and Vision Assistant multimodal model",
    "bakllava": "BakLLaVA multimodal vision-language model",
    "hermes3": "Nous Research's Hermes 3 instruction-tuned model",
    "openhermes": "Nous Research's OpenHermes chat model",
    "neural-chat": "Intel's neural chat optimized model",
    "dolphin-mixtral": "Dolphin fine-tuned Mixtral model",
    "granite-code": "IBM's Granite code model",
    "magicoder": "Code generation model trained on synthetic data",
}

# Use-case descriptions for generating model descriptions
_USE_CASE_DESCRIPTIONS: dict[str, str] = {
    "coding": "Optimized for code generation and completion",
    "reasoning": "Designed for complex reasoning and analysis",
    "chat": "General-purpose conversational model",
}

# Hardcoded fallback models used when the live API is unreachable (--offline mode)
FALLBACK_MODELS: list[dict] = [
    {
        "name": "llama3.2", "param_size": "3B",
        "size_gb": 2.0, "quant": "Q4_K_M",
        "desc": "Meta's compact and efficient Llama 3.2 model",
    },
    {
        "name": "llama3.2", "param_size": "1B",
        "size_gb": 0.7, "quant": "Q4_K_M",
        "desc": "Meta's compact and efficient Llama 3.2 model",
    },
    {
        "name": "llama3.1", "param_size": "8B",
        "size_gb": 4.7, "quant": "Q4_K_M",
        "desc": "Meta's Llama 3.1 with extended context support",
    },
    {
        "name": "llama3.3", "param_size": "70B",
        "size_gb": 40.0, "quant": "Q4_K_M",
        "desc": "Meta's Llama 3.3 flagship model",
    },
    {
        "name": "mistral", "param_size": "7B",
        "size_gb": 4.1, "quant": "Q4_K_M",
        "desc": "Mistral AI's efficient base model",
    },
    {
        "name": "deepseek-r1", "param_size": "7B",
        "size_gb": 4.7, "quant": "Q4_K_M",
        "desc": "DeepSeek's reasoning-focused model",
    },
    {
        "name": "deepseek-coder", "param_size": "6.7B",
        "size_gb": 3.8, "quant": "Q4_K_M",
        "desc": "DeepSeek's model optimized for code generation",
    },
    {
        "name": "codellama", "param_size": "7B",
        "size_gb": 3.8, "quant": "Q4_K_M",
        "desc": "Meta's code-specialized Llama model",
    },
    {
        "name": "phi4", "param_size": "14B",
        "size_gb": 8.4, "quant": "Q4_K_M",
        "desc": "Microsoft's Phi-4 reasoning model",
    },
    {
        "name": "phi3", "param_size": "3.8B",
        "size_gb": 2.3, "quant": "Q4_K_M",
        "desc": "Microsoft's compact and efficient Phi-3 model",
    },
    {
        "name": "gemma2", "param_size": "9B",
        "size_gb": 5.4, "quant": "Q4_K_M",
        "desc": "Google's Gemma 2 open model",
    },
    {
        "name": "gemma3", "param_size": "12B",
        "size_gb": 7.2, "quant": "Q4_K_M",
        "desc": "Google's latest Gemma 3 multimodal model",
    },
    {
        "name": "qwen2.5", "param_size": "7B",
        "size_gb": 4.4, "quant": "Q4_K_M",
        "desc": "Alibaba's multilingual Qwen 2.5 model",
    },
    {
        "name": "qwen2.5-coder", "param_size": "7B",
        "size_gb": 4.4, "quant": "Q4_K_M",
        "desc": "Alibaba's code-specialized Qwen model",
    },
    {
        "name": "smollm2", "param_size": "1.7B",
        "size_gb": 1.0, "quant": "Q4_K_M",
        "desc": "HuggingFace's ultra-compact language model",
    },
]


def _infer_use_cases(model_name: str) -> list[str]:
    name_lower = model_name.lower()
    cases = []
    for use_case, patterns in USE_CASE_MAP.items():
        if any(p in name_lower for p in patterns):
            cases.append(use_case)
    return cases if cases else ["chat"]


def _generate_description(model_name: str, use_cases: list[str]) -> str:
    """Generate a description from known model families or use cases."""
    # Try exact match first, then prefix match
    name_lower = model_name.lower()
    if name_lower in _MODEL_DESCRIPTIONS:
        return _MODEL_DESCRIPTIONS[name_lower]
    for key, desc in _MODEL_DESCRIPTIONS.items():
        if name_lower.startswith(key):
            return desc

    # Fall back to use-case description
    if use_cases:
        parts = [_USE_CASE_DESCRIPTIONS.get(uc, "") for uc in use_cases]
        parts = [p for p in parts if p]
        if parts:
            return ". ".join(parts)

    return f"{model_name} model from Ollama library"


def _parse_quantization(tag: str) -> str:
    tag_lower = tag.lower()
    for q in ["q2_k", "q3_k", "q4_0", "q4_k_m", "q4_k_s", "q5_0", "q5_k_m",
              "q6_k", "q8_0", "f16", "fp16", "f32"]:
        if q in tag_lower:
            return q.upper()
    if "instruct" in tag_lower or "chat" in tag_lower:
        return "Q4_K_M"
    return "Q4_0"


def _parse_param_size(text: str) -> str:
    """Extract parameter size from a tag or name string like 'llama3.2:7b' -> '7B'."""
    match = re.search(r"(\d+\.?\d*)[bB]", text)
    if match:
        return f"{match.group(1)}B"
    return "?"


def _parse_param_size_from_name_and_tag(name: str, tag: str) -> str:
    """Try tag first, then name, to extract parameter size."""
    result = _parse_param_size(tag)
    if result != "?":
        return result
    result = _parse_param_size(name)
    if result != "?":
        return result
    return "?"


def _estimate_size(tag: str) -> float:
    """Rough size estimate based on tag string."""
    match = re.search(r"(\d+\.?\d*)[bB]", tag)
    if match:
        params = float(match.group(1))
        # Q4 ≈ params * 0.55 GB roughly
        return round(params * 0.55, 1)
    return 4.0


def _generate_default_variants(model_name: str) -> list[ModelVariant]:
    """Generate a default variant for a model when tags aren't detailed."""
    size = _estimate_size(model_name)
    param = _parse_param_size(model_name)
    return [ModelVariant(tag="latest", size_gb=size, quantization="Q4_0", param_size=param)]


def get_fallback_models() -> list[OllamaModel]:
    """Return hardcoded fallback models for offline mode."""
    models = []
    for entry in FALLBACK_MODELS:
        tag = entry["param_size"].lower().rstrip("b") + "b"
        variant = ModelVariant(
            tag=tag,
            size_gb=entry["size_gb"],
            quantization=entry["quant"],
            param_size=entry["param_size"],
        )
        models.append(OllamaModel(
            name=entry["name"],
            description=entry["desc"],
            tags=[variant],
            use_cases=_infer_use_cases(entry["name"]),
        ))
    return models


def fetch_ollama_models(limit: int = 50) -> list[OllamaModel]:
    """Fetch models from the Ollama library API with robust gap-filling."""
    try:
        response = requests.get(
            OLLAMA_API_URL,
            timeout=REQUEST_TIMEOUT,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        raise ConnectionError(
            f"Failed to fetch Ollama model list: {e}\n"
            "  Hint: use --offline to skip the live fetch and use built-in fallback models."
        )

    models = []

    # The /api/tags endpoint returns {"models": [...]}
    items = data.get("models", []) if isinstance(data, dict) else data

    if not items:
        raise ConnectionError(
            "Ollama API returned an empty model list. "
            "Use --offline to use built-in fallback models."
        )

    for item in items:
        name_raw = item.get("name", "")
        if not name_raw:
            continue

        # The API returns names like "llama3.2:3b" — split into base name and tag
        if ":" in name_raw:
            base_name, tag_str = name_raw.split(":", 1)
        else:
            base_name, tag_str = name_raw, "latest"

        use_cases = _infer_use_cases(base_name)

        # Description: API returns none, so generate from known families
        description = item.get("description", "") or _generate_description(base_name, use_cases)

        # Size: parse from bytes in response
        size_bytes = item.get("size", 0)
        size_gb = round(size_bytes / (1024 ** 3), 1) if size_bytes else _estimate_size(tag_str)

        # Parameter size: API returns empty, parse from name:tag
        details = item.get("details", {}) or {}
        param_size = (
            details.get("parameter_size", "")
            or _parse_param_size_from_name_and_tag(base_name, tag_str)
        )

        # Quantization: API returns empty, infer from tag or default
        quantization = details.get("quantization_level", "") or _parse_quantization(tag_str)

        variant = ModelVariant(
            tag=tag_str,
            size_gb=size_gb,
            quantization=quantization,
            param_size=param_size,
        )

        models.append(OllamaModel(
            name=base_name,
            description=description,
            tags=[variant],
            use_cases=use_cases,
        ))

    return models[:limit]


def get_pulled_models() -> list[str]:
    """Return list of already-pulled model names via `ollama list`."""
    if not shutil.which("ollama"):
        return []
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        pulled = []
        for line in result.stdout.strip().splitlines()[1:]:  # skip header
            parts = line.split()
            if parts:
                model_tag = parts[0]
                name = model_tag.split(":")[0]
                pulled.append(name)
        return list(set(pulled))
    except Exception:
        return []


def pull_model(model_name: str) -> None:
    """Run `ollama pull <model>` as a live streaming subprocess."""
    if not shutil.which("ollama"):
        raise FileNotFoundError("ollama binary not found. Is Ollama installed?")
    subprocess.run(["ollama", "pull", model_name], check=True)
