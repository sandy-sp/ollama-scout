# Usage Guide

## Installation

```bash
pip install ollama-scout
```

Or from source:

```bash
git clone https://github.com/sandy-sp/ollama-scout.git
cd ollama-scout
pip install -e .
```

**Requirements:**
- Python 3.10+
- [Ollama](https://ollama.com/) installed (for model pulling and benchmark; recommendations work without it)

---

## Interactive Mode

When you run `ollama-scout` with no arguments (or with `-i`), it launches an interactive guided session:

```bash
ollama-scout        # launches interactive mode
ollama-scout -i     # same thing, explicit flag
```

The session walks you through these steps:

1. **Welcome** — Press Enter to begin scanning
2. **Hardware scan** — Detects GPU, CPU, and RAM automatically
3. **Connection check** — Choose to fetch live models or use the built-in offline list
4. **Use case selection** — Pick from All, Coding, Reasoning, or Chat
5. **Results count** — Choose how many recommendations to show (5/10/15/20)
6. **Recommendations** — View the results table with fit labels and run modes
7. **Compare** — Optionally compare two models side by side
8. **Benchmark** — Optionally run real inference speed tests on pulled models
9. **Export** — Optionally save results as a Markdown report
10. **Pull** — Optionally pull a recommended model directly
11. **Exit** — Done! The `ollama run MODEL` command is shown if you pulled a model

Interactive mode is recommended for new users. Power users can use CLI flags to skip the prompts.

---

## CLI Flags

### Basic scan

```bash
ollama-scout --use-case coding
ollama-scout --use-case reasoning
ollama-scout --use-case chat
ollama-scout --use-case all      # default
```

### Flat list view

```bash
ollama-scout --flat
```

Displays all recommendations in a single flat table instead of grouped by use case.

### Limit results

```bash
ollama-scout --top 20
```

Show the top N recommendations (default: 15).

### Export to Markdown

```bash
ollama-scout --export                    # Auto-export with timestamp filename
ollama-scout --output ~/report.md        # Export to a specific path
```

### Pull a model

```bash
ollama-scout --pull llama3.2:latest
```

Directly pull a specific model via `ollama pull` without running a full scan.

### Offline mode

```bash
ollama-scout --offline
```

Skips the live API fetch and uses a built-in list of ~15 popular models. Useful when you have no internet or the Ollama API is unreachable.

### Single model detail view

```bash
ollama-scout --model deepseek-coder
```

Shows all variants of a specific model with individual fit scores and run modes.

### Compare two models

```bash
ollama-scout --compare llama3.2 mistral
```

Side-by-side comparison with a verdict on which model is a better fit for your hardware.

### Benchmark pulled models

```bash
ollama-scout --benchmark
```

Runs real `ollama run` timing tests on all currently-pulled models and shows tokens/sec with a rating.

### Skip interactive prompts

```bash
ollama-scout --no-pull-prompt            # Skip the "pull a model?" prompt
ollama-scout --export --no-pull-prompt   # Non-interactive: export and exit
```

### Update model cache

```bash
ollama-scout --update-models
```

Force-refreshes the model list from the Ollama API and updates the local 24-hour cache.

---

## System Health Check (`--doctor`)

```bash
ollama-scout --doctor
```

Runs a comprehensive health check and prints a summary table:

| Check | What it verifies |
|-------|-----------------|
| Python ≥ 3.10 | Python version compatibility |
| Ollama binary | `ollama` found in PATH, returns version |
| GPU / VRAM | GPU(s) detected with total VRAM |
| RAM (≥ 4 GB) | System RAM meets minimum requirement |
| Internet | Can reach the internet (for live model fetch) |
| Model cache | Cache file exists and freshness (24h TTL) |
| Config file | Config file is valid JSON with known keys |
| Pulled models | Number of currently-pulled Ollama models |

Warnings are shown for items that need attention; all checks passing prints "All checks passed."

---

## Config Profiles

Config profiles let you save named sets of settings for different use cases (e.g., a "quick" profile for demos or a "coding" profile for your daily workflow).

### List profiles

```bash
ollama-scout --profile-list
```

Shows all profiles with their active status and any override values.

### Create a profile

```bash
ollama-scout --profile-create coding
```

Creates a new empty profile named `coding`.

### Set values in a profile

Use `--profile` with `--config-set` to set values in a specific profile:

```bash
ollama-scout --profile coding --config-set default_use_case=coding
ollama-scout --profile coding --config-set default_top_n=20
```

### Switch the active profile

```bash
ollama-scout --profile-switch coding
```

Makes `coding` the active profile for all future runs.

### Use a profile for a single run

```bash
ollama-scout --profile coding
```

Applies the `coding` profile overrides for this run only, without changing the active profile.

### Delete a profile

```bash
ollama-scout --profile-delete coding
```

Deletes the named profile. The `default` profile cannot be deleted.

---

## Persistent Configuration

```bash
ollama-scout --config                       # Show current config
ollama-scout --config-set key=value         # Set a value
```

Available config keys:

| Key | Default | Description |
|-----|---------|-------------|
| `default_use_case` | `"all"` | Use case filter applied automatically |
| `default_top_n` | `15` | Default number of recommendations |
| `auto_export` | `false` | Automatically export results to Markdown |
| `export_dir` | `""` | Directory for auto-exported reports |
| `offline_mode` | `false` | Always use built-in fallback model list |
| `show_benchmark` | `false` | Always run benchmarks |

Config files are stored at:
- **Linux:** `~/.config/ollama-scout/config.json` (respects `$XDG_CONFIG_HOME`)
- **macOS:** `~/Library/Application Support/ollama-scout/config.json`
- **Windows:** `%APPDATA%\ollama-scout\config.json`

---

## How Scoring Works

ollama-scout matches each model variant against your hardware:

1. **Model size** — from the download size (reflects parameter count + quantization).

2. **Fit scoring:**

| Condition | Fit Label | Run Mode | Meaning |
|-----------|-----------|----------|---------|
| VRAM ≥ model size | **Excellent** | GPU | Model fits entirely in GPU memory |
| VRAM + RAM ≥ model size | **Good** | CPU+GPU | Split between GPU and system RAM |
| RAM ≥ model size (no GPU) | **Possible** | CPU | CPU-only inference |
| Neither sufficient | *Excluded* | N/A | Too large, not shown |

3. **Multi-GPU:** Combined VRAM from all GPUs is used for scoring ("Multi-GPU" run mode).

4. **Apple Silicon:** Total RAM is treated as VRAM (unified memory), minus a 4 GB OS reserve.

5. **Score tiebreakers:** Smaller models score higher (faster inference). Already-pulled models get a bonus.

---

## Platform-Specific Notes

### Linux
- **GPU detection:** `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD ROCm)
- **CPU detection:** Reads `/proc/cpuinfo`
- **RAM detection:** `psutil` if available, otherwise `/proc/meminfo`

### macOS
- **Apple Silicon (M1–M4):** Detected via `platform.processor()` and `sysctl hw.optional.arm64`
- **Intel Macs:** GPU VRAM via `system_profiler SPDisplaysDataType`
- **CPU:** `sysctl -n machdep.cpu.brand_string`
- **RAM:** `sysctl -n hw.memsize`

### Windows
- **GPU detection:** `nvidia-smi` (NVIDIA) or `wmic` / PowerShell fallback
- **CPU detection:** `wmic cpu get` commands
- **RAM detection:** `psutil` if available, otherwise `wmic computersystem get TotalPhysicalMemory`

---

## FAQ

### "Why is my model marked Possible?"

The model fits in RAM but not GPU VRAM. It will run via CPU inference — functional but slow. Consider a smaller variant or a more quantized version (e.g., Q4_K_M instead of Q8_0).

### "Ollama not detected?"

ollama-scout needs the `ollama` binary on your PATH to detect pulled models and pull new ones. The scan and recommendations still work without Ollama. Install from [ollama.com](https://ollama.com/).

### "Live fetch failed?"

The Ollama library API may be temporarily unreachable. ollama-scout automatically falls back to a built-in list of ~15 popular models. Force offline mode with `--offline`.

### "How fresh is the model cache?"

The model list is cached for 24 hours. Run `--update-models` to force a refresh. `--doctor` shows the cache age.
