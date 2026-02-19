# Contributing Guide

## Development Setup

```bash
git clone https://github.com/jgonsior/olympic-games-of-active-learning.git
cd olympic-games-of-active-learning

# Create conda environment (choose your platform)
conda create --name al_olympics_dev --file conda-linux-64.lock   # Linux
# conda create --name al_olympics_dev --file conda-osx-64.lock   # macOS Intel
# conda create --name al_olympics_dev --file conda-osx-arm64.lock # macOS Apple Silicon
# conda create --name al_olympics_dev --file conda-win-64.lock    # Windows

conda activate al_olympics_dev
poetry install
```

---

## Code Quality

```bash
ruff check .          # Lint (auto-fix: ruff check --fix .)
black --check .       # Format check (apply: black .)
mypy .                # Type check
pycln .               # Remove unused imports
```

---

## Testing

```bash
pytest                # Run tests
pytest -v             # Verbose
./test.sh             # Full evaluation pipeline (slow)
```

---

## Project Structure

```
├── 01_create_workload.py        # Workload generation
├── 02_run_experiment.py         # Experiment execution
├── 03_calculate_dataset_categorizations.py
├── 04_calculate_advanced_metrics.py
├── datasets/                    # Dataset loaders
├── framework_runners/           # AL framework adapters
├── metrics/                     # Metric computation
├── misc/                        # Config, logging, helpers
├── resources/                   # Config files, enums
├── eva_scripts/                 # Evaluation & plotting
└── docs/                        # Documentation
```

---

## Coding Style

- PEP 8, type hints, lines ≤ 100 chars
- Google-style docstrings
- Import order: stdlib → third-party → local

---

## Pull Request Checklist

- [ ] All tests pass (`pytest`)
- [ ] Linting passes (`ruff check .`)
- [ ] Type checking passes (`mypy .`)
- [ ] New functions have type hints and docstrings
- [ ] Documentation updated if needed

---

## Debugging Tips

```bash
# Single experiment with verbose output
python 02_run_experiment.py --EXP_TITLE test_debug --WORKER_INDEX 0 --verbose

# Validate config
python -c "from misc.config import Config; Config()"

# Smoke test with new config
python 01_create_workload.py --EXP_TITLE config_test
python 02_run_experiment.py --EXP_TITLE config_test --WORKER_INDEX 0
```

---

## Reporting Bugs

Include: OS, Python version, exact command, full traceback, and relevant config (sanitized).
