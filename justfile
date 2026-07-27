# Project command menu. `just` lists the recipes; `just <name>` runs one.

# List the available recipes (the default when no recipe is given).
default:
    @just --list

# End-to-end pipeline check: trains the pipeline on synthetic data and logs a number.
smoke:
    uv run python scripts/run_pipeline_check.py

# Run the test suite.
test:
    uv run pytest

# Lint the code with ruff.
lint:
    uv run ruff check .

# Auto-format the code with ruff.
format:
    uv run ruff format .

# Run what CI runs: lint, then tests.
check: lint test
