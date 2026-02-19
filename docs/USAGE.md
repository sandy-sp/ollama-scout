# Usage Guide

## Installation

```bash
git clone https://github.com/sandy-sp/ollama-scout.git
cd ollama-scout
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- [Ollama](https://ollama.com/) installed (for model pulling and local model detection)

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
7. **Benchmark** — Optionally view estimated inference speeds
8. **Export** — Optionally save results as a Markdown report
9. **Pull** — Optionally pull a recommended model directly
10. **Exit** — Done! Tips for CLI flags are shown

Interactive mode is recommended for new users. Power users can use CLI flags to skip the prompts.

## CLI Flags

### Basic scan

```bash
python main.py
```

Scans your hardware and displays model recommendations grouped by use case (Coding, Reasoning, Chat).

### Filter by use case

```bash
python main.py --use-case coding
python main.py --use-case reasoning
python main.py --use-case chat
```

Only shows models tagged for the specified use case.

### Flat list view

```bash
python main.py --flat
```

Displays all recommendations in a single flat table instead of grouped by use case.

### Limit results

```bash
python main.py --top 20
```

Show the top N recommendations (default: 15).

### Export to Markdown

```bash
python main.py --export                    # Auto-export with timestamp filename
python main.py --output ~/report.md        # Export to a specific path
```

Saves a formatted Markdown report of your hardware profile and recommendations.

### Pull a model

```bash
python main.py --pull llama3.2:latest
```

Directly pull a specific model via `ollama pull` without running a full scan.

### Offline mode

```bash
python main.py --offline
```

Skips the live API fetch and uses a built-in list of ~15 popular models with known sizes. Useful when you have no internet or the Ollama API is unreachable.

### Skip interactive prompts

```bash
python main.py --no-pull-prompt            # Skip the "pull a model?" prompt
python main.py --export --no-pull-prompt   # Non-interactive: export and exit
```

### Combine flags

```bash
python main.py --offline --use-case coding --top 10 --export --no-pull-prompt
```

## How Scoring Works

ollama-scout matches each model variant against your hardware using this logic:

1. **Model size** is determined by the download size (from the API or fallback list), which reflects both parameter count and quantization level.

2. **Fit scoring** follows this priority:

| Condition | Fit Label | Run Mode | Meaning |
|-----------|-----------|----------|---------|
| VRAM >= model size | **Excellent** | GPU | Model fits entirely in GPU memory. Best performance. |
| VRAM + RAM >= model size | **Good** | CPU+GPU | Model is split between GPU and system RAM. Moderate speed. |
| RAM >= model size (no GPU) | **Possible** | CPU | Runs entirely on CPU. Functional but slow. |
| Neither sufficient | *Excluded* | N/A | Model is too large and is not shown. |

3. **Score tiebreakers:** smaller models score higher (faster inference). Already-pulled models get a bonus.

4. **Apple Silicon:** On M1/M2/M3/M4 Macs, RAM and VRAM are the same unified memory pool. ollama-scout detects this and treats your total RAM as available VRAM, minus a 4GB reserve for macOS overhead.

## Platform-Specific Notes

### Linux
- **GPU detection:** Uses `nvidia-smi` (NVIDIA) or `rocm-smi` (AMD ROCm)
- **CPU detection:** Reads `/proc/cpuinfo`
- **RAM detection:** Uses `psutil` if available, otherwise reads `/proc/meminfo`

### macOS
- **Apple Silicon (M1/M2/M3/M4):** Detected via `platform.processor()` and `sysctl hw.optional.arm64`. Unified memory is treated as VRAM for scoring.
- **Intel Macs:** GPU VRAM detected via `system_profiler SPDisplaysDataType`
- **CPU detection:** Uses `sysctl -n machdep.cpu.brand_string`
- **RAM detection:** Uses `sysctl -n hw.memsize`

### Windows
- **GPU detection:** Uses `nvidia-smi` (NVIDIA) or `wmic` for other GPUs
- **CPU detection:** Uses `wmic cpu get` commands
- **RAM detection:** Uses `psutil` if available, otherwise `wmic computersystem get TotalPhysicalMemory`

## FAQ

### "Why is my model marked Possible?"

The model fits in your system RAM but not in your GPU VRAM. It will run via CPU inference, which works but is significantly slower than GPU inference. Consider a smaller variant or a quantized version (e.g., Q4 instead of F16).

### "Ollama not detected?"

ollama-scout needs the `ollama` binary on your PATH to detect already-pulled models and to pull new ones. Install Ollama from [ollama.com](https://ollama.com/). The scan and recommendations still work without Ollama installed — you just won't see pulled-model detection or be able to auto-pull.

### "Live fetch failed?"

The Ollama library API (`https://ollama.com/api/tags`) may be temporarily unreachable. When this happens, ollama-scout automatically falls back to a built-in list of ~15 popular models. You can also force this with `--offline`:

```bash
python main.py --offline
```

### "Why are the sizes different from what I see on ollama.com?"

The sizes shown are the on-disk download sizes reported by the API, which reflect the quantized model weights. The actual RAM/VRAM needed at runtime is similar but may vary slightly depending on context window size and batch settings.
