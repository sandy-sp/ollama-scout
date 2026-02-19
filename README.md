# ollama-scout

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/ollama-scout)](https://pypi.org/project/ollama-scout/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)](#)
[![CI](https://github.com/sandy-sp/ollama-scout/actions/workflows/ci.yml/badge.svg)](https://github.com/sandy-sp/ollama-scout/actions/workflows/ci.yml)

> One command to find the right LLMs for your hardware.

**ollama-scout** scans your GPU VRAM, CPU, and RAM, then recommends compatible [Ollama](https://ollama.com/) models grouped by use case — with an interactive guided mode for new users and full CLI flags for power users.

---

## Demo

<!-- TODO: Add demo.gif once recorded. Recommended tool: vhs (https://github.com/charmbracelet/vhs) -->
<!-- ![demo](docs/demo.gif) -->

```
$ ollama-scout

  ollama-scout v0.2.0  |  LLM Hardware Advisor  (Interactive Mode)

  Welcome! Let's find the best LLMs for your hardware.
  Press Enter to scan your system, or Ctrl+C to exit.

                   System Hardware
  ┌─────────────────┬────────────────────────────────────────┐
  │ OS              │ Linux                                  │
  │ CPU             │ AMD Ryzen 9 5900X                      │
  │ Cores / Threads │ 12 cores / 24 threads                  │
  │ RAM             │ 32.0 GB                                │
  │ GPU             │ NVIDIA RTX 3080 (10.0 GB VRAM)         │
  └─────────────────┴────────────────────────────────────────┘

  What are you mainly using this for?
    1. All categories    2. Coding    3. Reasoning    4. Chat

  Coding Models
  Model             Tag    Quant    Size    Fit        Mode     Status
  deepseek-coder    6.7b   Q4_K_M   3.8GB   Excellent  GPU      Available
  codellama         7b     Q4_K_M   3.8GB   Excellent  GPU      Pulled

  Would you like to compare two models? [y/N]:
  Save results as a Markdown report? [y/N]:
```

---

## Quick Start

```bash
# Recommended
pipx install ollama-scout

# Or with pip
pip install ollama-scout

# Or from source
git clone https://github.com/sandy-sp/ollama-scout.git
cd ollama-scout && pip install -e .
```

Then run:

```bash
ollama-scout          # interactive guided mode — no flags needed
```

Requires **Python 3.10+** and [Ollama](https://ollama.com/) installed.

---

## Features

| Feature | Description |
|---------|-------------|
| Interactive guided mode | Step-by-step session with no flags needed (`ollama-scout` or `-i`) |
| Hardware detection | NVIDIA, AMD ROCm, Apple Silicon unified memory, multi-GPU |
| Live + offline models | Fetches from Ollama API with 24hr cache; `--offline` uses built-in list |
| Smart scoring | GPU > Multi-GPU > CPU+GPU offload > CPU-only, with time estimates |
| Use-case grouping | Coding, Reasoning, Chat with per-category tables |
| Real benchmark timing | Measures actual tokens/sec on pulled models via `ollama run` |
| Model comparison | `--compare model1 model2` for side-by-side analysis with verdict |
| Model detail view | `--model NAME` shows all variants scored against your hardware |
| Markdown export | `--export` saves a formatted report |
| Persistent config | XDG-compliant paths with `--config` and `--config-set` |
| Auto-pull | Pull recommended models interactively or via `--pull` |

---

## CLI Reference

| Flag | Description | Example |
|------|-------------|---------|
| `-i`, `--interactive` | Launch guided mode (default with no args) | `ollama-scout -i` |
| `--use-case` | Filter by category | `--use-case coding` |
| `--flat` | Flat list instead of grouped tables | `--flat` |
| `--top N` | Limit number of results | `--top 20` |
| `--offline` | Use built-in fallback model list | `--offline` |
| `--benchmark` | Show inference speed estimates | `--benchmark` |
| `--model NAME` | Detail view for a specific model | `--model deepseek-coder` |
| `--compare M1 M2` | Side-by-side model comparison | `--compare llama3.2 mistral` |
| `--export` | Auto-export Markdown report | `--export` |
| `--output PATH` | Export to a specific file | `--output ~/report.md` |
| `--pull MODEL` | Pull a model via ollama | `--pull llama3.2:3b` |
| `--update-models` | Force-refresh model list cache | `--update-models` |
| `--config` | Show current configuration | `--config` |
| `--config-set K=V` | Set a config value | `--config-set offline_mode=true` |
| `--no-pull-prompt` | Skip interactive pull prompt | `--no-pull-prompt` |
| `--version` | Show version | `--version` |

Run `ollama-scout --help` for the full list. See [docs/USAGE.md](docs/USAGE.md) for detailed examples.

---

## How Scoring Works

| Fit | Mode | Meaning |
|-----|------|---------|
| Excellent | GPU | Model fits fully in single GPU VRAM |
| Excellent | Multi-GPU | Model distributed across multiple GPUs |
| Good | CPU+GPU | Partially offloaded to RAM — usable but slower |
| Possible | CPU | CPU-only inference with time estimate |
| *(excluded)* | — | Model too large for available memory |

Apple Silicon: unified memory is treated as VRAM, so M1/M2/M3/M4 Macs get GPU-tier scoring with a 4GB system reserve.

---

## Roadmap

- [x] Interactive guided mode
- [x] Real benchmark timing (`ollama run`)
- [x] Model comparison mode (`--compare`)
- [x] Multi-GPU support
- [x] XDG-compliant config paths
- [x] Model list caching (24hr TTL)
- [x] pip installable on PyPI
- [ ] Live streaming benchmarks with progress bar
- [ ] Model search and filter by keyword
- [ ] Web UI version
- [ ] Config profiles (work / gaming / minimal)

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and guidelines.

## License

MIT — see [LICENSE](LICENSE).

---

*Generated by [ollama-scout](https://github.com/sandy-sp/ollama-scout)*
