# ollama-scout v0.2.0

A major feature release bringing real benchmarks, multi-GPU support, model comparison, smarter deduplication, and model list caching.

## What's New

- **Multi-GPU support** — Models too large for a single GPU are now scored across all GPUs combined, with a new "Multi-GPU" run mode
- **Real benchmark timing** — `--benchmark` now runs `ollama run` on your pulled models and reports measured tokens/sec alongside formula estimates
- **Model comparison mode** — `--compare model1 model2` shows a side-by-side table with a winner verdict
- **Smarter model deduplication** — Models with multiple variants (e.g. llama3.2 with 1b and 3b) are now properly merged into a single entry
- **Model list caching** — API results are cached for 24 hours; use `--update-models` to force-refresh
- **XDG-compliant config paths** — Config now lives in `~/.config/ollama-scout/` (Linux), `~/Library/Application Support/ollama-scout/` (macOS), or `%APPDATA%\ollama-scout\` (Windows), with automatic migration from the legacy `~/.ollama-scout.json` path
- **Interactive comparison step** — The guided interactive mode now offers to compare two models after showing recommendations

## Fixes

- **EOFError when piped** — `prompt_export()` and `prompt_pull()` no longer crash when stdout is piped or stdin is closed
- **Table column truncation** — Fit, Mode, and Note columns now have minimum widths and graceful wrapping (v0.1.1 backport)
- **CPU-only time estimates** — Recommendation notes show estimated seconds for 200 tokens instead of generic "may be slow" (v0.1.1 backport)

## Upgrade

```bash
pipx upgrade ollama-scout
# or
pip install --upgrade ollama-scout
```

## Breaking Changes

None — fully backwards compatible.

## What's Next (v0.3.0)

- **Live streaming benchmarks** — Show real-time token output during benchmark runs with a progress bar
- **Model search and filter** — `--search "code python"` to find models by keyword across names and descriptions
- **Custom model registry** — Support for private Ollama registries and custom model sources beyond the public library
- **Profile presets** — Save and switch between hardware profiles (e.g. "laptop" vs "workstation") for quick comparisons
- **JSON/CSV export** — Machine-readable output formats alongside the existing Markdown export
