# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-02-21

### Added

- Graceful "Ollama not installed" detection: `check_ollama_installed()` checks binary + version; warning panel shown at startup; pull steps skipped automatically
- Interactive mode shows Ollama version in welcome step when installed
- Interactive mode shows Apple Silicon context message in hardware step
- Use-case menu now shows one-line descriptions for each option
- Results count prompt replaced with numbered menu (1→5, 2→10, 3→15, 4→20)
- Summary line shown before recommendation tables ("Found N compatible models")
- Pull step shows context note and run command after successful pull
- Exit step shows `ollama run MODEL` command if a model was pulled

### Changed

- `group_by_use_case()` no longer hard-caps groups at 5 models (bounded only by `top_n`)
- Benchmark results show only real measured timings (formula estimates removed)
- `--benchmark` shows "pull a model first" panel when no models are pulled
- Model comparison no longer shows estimated speed (Est. Speed row shows N/A)
- Separator lines (`console.rule`) added between interactive steps

### Removed

- `estimate_speed()` and `get_benchmark_estimates()` (formula-based benchmark helpers)
- `is_real` field from `BenchmarkEstimate` (all benchmarks are real now)
- Source column ("⚡ Real" / "~ Est.") from benchmark table

## [0.2.0] - 2026-02-18

### Added

- Multi-GPU support: models scored across combined VRAM with new "Multi-GPU" run mode
- Real benchmark timing via `ollama run` for pulled models with "Source" column (⚡ Real / ~ Est.)
- Model comparison mode: `--compare model1 model2` for side-by-side analysis with verdict
- Interactive comparison prompt after recommendations
- Smarter model deduplication: same-name models merged with all variants
- Model list caching (24h TTL) with `--update-models` to force-refresh
- XDG-compliant config paths (Linux: `~/.config/`, macOS: `~/Library/Application Support/`, Windows: `%APPDATA%`)
- Automatic migration of legacy `~/.ollama-scout.json` to new config location

### Fixed

- EOFError when stdout is piped or stdin is closed (`prompt_export()`, `prompt_pull()`)

## [0.1.1] - 2026-02-18

### Added

- Interactive guided mode (`ollama-scout` with no args, or `-i` flag)
- 10-step session: hardware scan → use case → results → benchmark → export → pull

### Fixed

- Table columns no longer truncate (Fit, Mode, Note columns)
- CPU-only notes now show time estimates for 200-token response
- Graceful Ctrl+C handling in interactive mode

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
