# Contributing to ollama-scout

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
git clone https://github.com/sandy-sp/ollama-scout.git
cd ollama-scout
pip install -e ".[dev]"
```

This installs the project in editable mode with dev dependencies (pytest, pytest-cov, ruff).

## Running Tests

```bash
pytest                              # Run all tests
pytest --cov=scout                  # Run with coverage report
pytest tests/test_hardware.py -v    # Run a specific test file
```

## Linting

We use [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check .          # Check for issues
ruff check --fix .    # Auto-fix what's possible
```

Configuration is in `pyproject.toml`. Rules enforced: E (pycodestyle errors), F (pyflakes), W (pycodestyle warnings), I (isort).

## Branch Naming

Use descriptive branch names:

- `feat/add-multi-gpu-scoring` — new features
- `fix/windows-ram-detection` — bug fixes
- `docs/update-usage-guide` — documentation
- `refactor/simplify-scoring` — code improvements

## Pull Request Checklist

Before submitting a PR:

- [ ] All tests pass (`pytest`)
- [ ] No lint errors (`ruff check .`)
- [ ] New features include tests
- [ ] Commit messages are clear and descriptive

## Areas Where Help is Wanted

- Better use-case mapping for new Ollama models
- Multi-GPU scoring improvements
- Additional platform testing (Windows ARM, Linux ARM)
- Integration tests with real Ollama installations
- Performance profiling for large model lists

## Code Style

- Max line length: 100 characters
- Target Python version: 3.10+
- Use type hints where practical
- Keep functions focused and small
