# ollama-scout v0.1.0 — Initial Release

**ollama-scout** is a cross-platform CLI tool that scans your GPU VRAM, CPU, and RAM, then recommends compatible Ollama models grouped by use case. It helps you find the right LLMs for your hardware and pull them instantly.

## Highlights

- **Smart hardware-aware scoring** — Automatically detects your GPU, CPU, and RAM across Windows, macOS, and Linux, then ranks every Ollama model as Excellent (full GPU), Good (CPU+GPU offload), or Possible (CPU-only)
- **Apple Silicon unified memory** — Correctly treats M1/M2/M3/M4 shared memory as GPU VRAM with a 4GB macOS reserve, so recommendations are accurate on Apple hardware
- **Use-case grouping** — Models are categorized into Coding, Reasoning, and Chat so you can find the right tool for your task at a glance
- **Offline mode with fallback** — Works without internet using a built-in list of 15 popular models; auto-falls back if the Ollama API is unreachable
- **Rich terminal UI** — Beautiful tables, color-coded fit labels, inference speed estimates, model detail views, and Markdown report export — all from a single command

## Installation

```bash
# From source
git clone https://github.com/sandy-sp/ollama-scout.git
cd ollama-scout
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
python main.py
```

Requires **Python 3.10+** and [Ollama](https://ollama.com/) installed.

## Quick Start

```bash
# Scan your hardware and see recommendations
ollama-scout

# See what runs best for coding tasks
ollama-scout --use-case coding

# Get speed estimates for top models
ollama-scout --benchmark
```

## Supported Platforms

| Platform | GPU Detection | Notes |
|----------|--------------|-------|
| **Linux** | NVIDIA (`nvidia-smi`), AMD (`rocm-smi`) | Reads `/proc/cpuinfo` and `/proc/meminfo` |
| **macOS** | Apple Silicon (unified memory), Intel (`system_profiler`) | M1/M2/M3/M4 unified memory treated as VRAM |
| **Windows** | NVIDIA (`nvidia-smi`), others (`wmic` or PowerShell) | PowerShell fallback for Windows 11+ where `wmic` is deprecated |

## Known Limitations

- **Benchmark estimates are formula-based**, not real measurements — actual tokens/sec depends on model architecture, context length, and system load
- **AMD ROCm GPU detection** has not been validated on real ROCm hardware — community testing welcome
- **Apple Silicon unified memory scoring** has not been validated on real M-series hardware — the 4GB macOS reserve is a conservative estimate
- **Multi-GPU setups** use only the best single GPU for scoring — multi-GPU model splitting is not yet supported
- **Ollama API metadata gaps** — the `/api/tags` endpoint returns limited metadata (no descriptions, sometimes missing param sizes); ollama-scout fills gaps heuristically from known model families
- **The `ollama-scout` CLI entry point** requires `pip install -e .` — not yet published on PyPI

## What's Next (v0.2.0)

- Real GPU benchmark timing via `ollama run` instead of formula estimates
- Model comparison mode (`--compare model1 model2`)
- XDG config path support (`~/.config/ollama-scout/`)
- Published PyPI package (`pip install ollama-scout`)
- Web UI version
