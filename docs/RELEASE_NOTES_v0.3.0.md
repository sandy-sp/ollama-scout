# Release Notes — ollama-scout v0.3.0

**Release date:** 2026-02-23

## Overview

v0.3.0 adds two major new features (`--doctor` health check and config profiles), pushes test coverage from 69% to 88%, and fully rewrites the usage documentation.

---

## New Features

### `--doctor` — System Health Check

```bash
ollama-scout --doctor
```

Runs 8 health checks and prints a formatted summary:

- **Python ≥ 3.10** — version compatibility
- **Ollama binary** — detects `ollama` in PATH and reads version
- **GPU / VRAM** — lists GPUs with total VRAM (Apple Silicon shown as unified memory)
- **RAM (≥ 4 GB)** — system RAM adequacy
- **Internet** — connectivity check for live model fetch
- **Model cache** — existence and freshness (24h TTL)
- **Config file** — valid JSON with known keys
- **Pulled models** — count of currently-pulled Ollama models

All passing prints "All checks passed." Failing checks show yellow warnings with details.

---

### Config Profiles

Named sets of config overrides for different workflows.

```bash
# Manage profiles
ollama-scout --profile-list
ollama-scout --profile-create coding
ollama-scout --profile-delete coding
ollama-scout --profile-switch coding

# Set values in a profile
ollama-scout --profile coding --config-set default_use_case=coding
ollama-scout --profile coding --config-set default_top_n=20

# Use a profile for a single run (without switching the active profile)
ollama-scout --profile coding
```

Profiles stack on top of the base config: only the overridden keys differ from the base. The `default` profile always exists and cannot be deleted.

Profile data is stored alongside `config.json` as `profiles.json`.

---

## Improvements

### Test coverage: 69% → 88%

Added 107 new tests across:

- `tests/test_hardware.py` — full coverage of platform-specific GPU/CPU/RAM detection, Apple Silicon detection, multi-GPU profiles, `_is_apple_silicon()` branches
- `tests/test_config.py` — migration error handling, OSError on save, print_config output
- `tests/test_interactive.py` — `_step_welcome`, `_step_hardware_scan`, `_step_compare`, `_step_benchmark`, `_step_export`, `_step_pull` step-level tests
- `tests/test_profiles.py` — 30 new tests for profile CRUD, override loading, active profile switching
- `tests/test_doctor.py` — 17 new tests for every doctor check function and `run_doctor()`

Total: 219 tests (was 112 at v0.2.0).

### Documentation rewrite

`docs/USAGE.md` fully rewritten to cover:

- All CLI flags (including `--doctor`, `--profile-*`, `--compare`, `--benchmark`, `--update-models`)
- Config profiles with usage examples
- `--doctor` health check reference table
- Persistent config key reference
- Platform-specific detection notes
- Updated FAQ

---

## Upgrade Notes

- No breaking changes — all existing CLI flags and config keys remain unchanged
- Profiles are optional — if you don't create any, behaviour is identical to v0.2.x
- `profiles.json` is created automatically the first time a profile command is used; existing `config.json` is untouched

---

## Stats

| Metric | v0.2.1 | v0.3.0 |
|--------|--------|--------|
| Tests | 178 | 219 |
| Coverage | 69% | 88% |
| CLI flags | 14 | 20 |
| Source lines (scout/) | ~1,150 | ~1,500 |
