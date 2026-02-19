# ollama-scout

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

> Scan your hardware. Find the right LLMs. Pull them instantly.

**ollama-scout** is a cross-platform CLI tool that detects your GPU VRAM, CPU, and RAM, then recommends compatible [Ollama](https://ollama.com/) models grouped by use case.

<!-- TODO: Add demo.gif once recorded. Recommended tool: vhs (https://github.com/charmbracelet/vhs) -->
<!-- ![demo](docs/demo.gif) -->

## Demo

```
╭──────────────────────────────────────────────────────────╮
│  ollama-scout  |  LLM Hardware Advisor                   │
╰──────────────────────────────────────────────────────────╯

                   System Hardware
╭─────────────────┬────────────────────────────────────────╮
│ OS              │ Linux                                  │
│ CPU             │ AMD Ryzen 9 5900X                      │
│ Cores / Threads │ 12 cores / 24 threads                  │
│ RAM             │ 32.0 GB                                │
│ GPU             │ NVIDIA RTX 3080 (10.0 GB VRAM)         │
╰─────────────────┴────────────────────────────────────────╯

  Coding Models
  Model             Tag      Quant    Size    Fit        Mode     Status
  deepseek-coder    6.7b     Q4_K_M   3.8GB   Excellent  GPU      Available
  codellama         7b       Q4_K_M   3.8GB   Excellent  GPU      Pulled
  qwen2.5-coder    7b       Q4_K_M   4.4GB   Excellent  GPU      Available

  Reasoning Models
  deepseek-r1       7b       Q4_K_M   4.7GB   Excellent  GPU      Available
  phi4              14b      Q4_K_M   8.4GB   Good       CPU+GPU  Available

  Chat Models
  llama3.2          3b       Q4_K_M   2.0GB   Excellent  GPU      Pulled
  mistral           7b       Q4_K_M   4.1GB   Excellent  GPU      Available
```

## Interactive Mode

Just run `ollama-scout` with no arguments for a guided experience:

```bash
ollama-scout
```

The interactive mode walks you through hardware scanning, use case selection, and model recommendations step by step — no flags needed. You can also launch it explicitly with `ollama-scout -i`.

## How It Works

```
1. Scan       Detects GPU VRAM, CPU cores/threads, RAM
              Supports NVIDIA, AMD (ROCm), Apple Silicon unified memory
                                    |
2. Fetch      Pulls latest models from Ollama library API
              Falls back to built-in list if offline
                                    |
3. Score      Matches each model variant to your hardware
              GPU fit > CPU+GPU offload > CPU-only > excluded
                                    |
4. Recommend  Groups results by use case: Coding, Reasoning, Chat
              Shows fit label, run mode, and pull status
```

## Installation

```bash
git clone https://github.com/sandy-sp/ollama-scout.git
cd ollama-scout
pip install -r requirements.txt
```

Requires **Python 3.10+** and [Ollama](https://ollama.com/) installed.

## Usage

```bash
ollama-scout                                # Interactive guided mode (default)
ollama-scout -i                             # Explicit interactive mode
python main.py                              # Full scan, grouped by use case
python main.py --use-case coding            # Filter by use case
python main.py --flat                       # Flat list instead of grouped
python main.py --top 20                     # Show top 20 results
python main.py --offline                    # Use built-in model list (no network)
python main.py --benchmark                  # Show inference speed estimates
python main.py --model deepseek-coder       # Detail view for a specific model
python main.py --export                     # Auto-export to Markdown report
python main.py --output ~/report.md         # Export to specific path
python main.py --pull llama3.2:latest       # Pull a model directly
python main.py --no-pull-prompt             # Skip interactive pull prompt
python main.py --config                     # Show current config
python main.py --config-set offline_mode=true  # Set a config value
```

See [docs/USAGE.md](docs/USAGE.md) for the full guide with platform-specific notes and FAQ.

## Features

- **Interactive mode** — Guided step-by-step session when run with no arguments
- **Hardware detection** — GPU VRAM, CPU, RAM on Windows, macOS, Linux
- **Apple Silicon support** — Treats unified memory as VRAM for accurate scoring
- **Live + offline modes** — Fetches from Ollama API or uses built-in fallback list
- **Smart recommendations** — Full GPU / partial CPU+GPU offload / CPU-only scoring
- **Use-case grouping** — Coding, Reasoning, Chat
- **Benchmark estimates** — Rough tokens/sec estimation per model
- **Model detail view** — Deep dive into a specific model's variants and compatibility
- **Already-pulled detection** — Highlights models you've downloaded
- **Auto-pull** — Pull a recommended model interactively
- **Markdown export** — Save results as a formatted report
- **Config file** — Persistent defaults via `~/.ollama-scout.json`

## Requirements

| Package | Purpose |
|---------|---------|
| `rich` | Terminal UI (tables, panels, spinners) |
| `requests` | Fetch Ollama library API |
| `psutil` | Cross-platform RAM detection |

## Roadmap

- [x] Config file support (~/.ollama-scout.json)
- [x] pip installable package (`ollama-scout` CLI command)
- [ ] GPU benchmark integration (real `ollama run` timing)
- [ ] Model comparison mode (side-by-side two models)
- [ ] XDG config path support (~/.config/ollama-scout/)
- [ ] Web UI version

## Contributing

PRs welcome! Especially for:
- Better use-case mapping for new models
- Multi-GPU scoring improvements
- Additional platform testing (Windows ARM, Linux ARM)

## License

MIT
