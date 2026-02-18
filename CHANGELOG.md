# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-17

### Added

- Cross-platform hardware detection (GPU VRAM, CPU, RAM)
- NVIDIA GPU detection via `nvidia-smi`
- AMD ROCm GPU detection via `rocm-smi`
- Apple Silicon unified memory support (M1/M2/M3/M4)
- Windows PowerShell fallback when `wmic` is unavailable
- Live model fetching from Ollama library API (`/api/tags`)
- Built-in fallback list of 15 popular models for offline use
- Smart recommendation scoring: Excellent (GPU) > Good (CPU+GPU) > Possible (CPU)
- Use-case grouping: Coding, Reasoning, Chat
- `--use-case` flag to filter by category
- `--flat` flag for flat list view
- `--top N` flag to limit results
- `--offline` flag to skip API fetch
- `--benchmark` flag for inference speed estimates
- `--model NAME` flag for detailed single-model view
- `--export` and `--output` flags for Markdown report export
- `--pull MODEL` flag to pull models directly
- `--no-pull-prompt` flag to skip interactive prompts
- `--config` and `--config-set` flags for persistent configuration
- `--version` flag
- Config file support (`~/.ollama-scout.json`)
- Rich terminal UI with tables, panels, and spinners
- Legend panel explaining fit labels and run modes
- Already-pulled model detection via `ollama list`
- Gap-filling for missing API metadata (descriptions, param sizes, quantization)
- GitHub Actions CI workflow (Python 3.10-3.12 matrix, ruff, pytest)
