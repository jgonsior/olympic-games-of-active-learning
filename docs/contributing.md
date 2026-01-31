# Contributing Guide

This document explains how to set up a development environment and contribute to OGAL.

## Development Setup

### Prerequisites

- [Anaconda](https://docs.anaconda.com/anaconda/install/) or Miniconda
- Git
- Python 3.11

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/jgonsior/olympic-games-of-active-learning.git
cd olympic-games-of-active-learning

# Create conda environment from lock file
conda create --name al_olympics_dev --file conda-linux-64.lock

# Activate environment
conda activate al_olympics_dev

# Install Python dependencies with Poetry
poetry install
```

### Platform-Specific Lock Files

| Platform | Lock File |
|----------|-----------|
| Linux | `conda-linux-64.lock` |
| macOS (Intel) | `conda-osx-64.lock` |
| macOS (Apple Silicon) | `conda-osx-arm64.lock` |
| Windows | `conda-win-64.lock` |

### Updating Dependencies

```bash
# Update Poetry dependencies
poetry update

# Regenerate conda lock files (requires conda-lock)
conda-lock --file environment.yml --platform linux-64 --platform osx-64 --platform osx-arm64 --platform win-64
```

---

## Code Quality Tools

### Linting with Ruff

```bash
# Run ruff linter
ruff check .

# Auto-fix issues
ruff check --fix .
```

### Type Checking with mypy

```bash
# Run mypy
mypy .
```

Configuration in `mypy.ini`:

```ini
[mypy]
plugins = numpy.typing.mypy_plugin
ignore_missing_imports = True
```

### Code Formatting with Black

```bash
# Format code
black .

# Check formatting without changes
black --check .
```

### Import Sorting with pycln

```bash
# Clean unused imports
pycln .
```

---

## Testing

### Running Tests

```bash
# Run pytest
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_specific.py
```

### Test Script

The [`test.sh`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/test.sh) script runs the full evaluation pipeline:

```bash
./test.sh
```

**Note:** This runs actual experiments and may take significant time.

---

## Project Structure

```
olympic-games-of-active-learning/
├── 00_download_datasets.py      # Dataset acquisition
├── 01_create_workload.py        # Workload generation
├── 02_run_experiment.py         # Experiment execution
├── 03_calculate_dataset_categorizations.py
├── 04_calculate_advanced_metrics.py
├── 05_analyze_partially_run_workload.py
├── 07b_create_results_without_flask.py
├── datasets/                    # Dataset loading utilities
│   ├── __init__.py             # DATASET enum and loaders
│   ├── base.py                 # Base loader class
│   ├── openml_loader.py        # OpenML dataset loader
│   ├── kaggle_loader.py        # Kaggle dataset loader
│   └── local_loader.py         # Local file loader
├── framework_runners/           # AL framework adapters
│   ├── base_runner.py          # Abstract base class
│   ├── alipy_runner.py         # ALiPy adapter
│   ├── libact_runner.py        # libact adapter
│   ├── smalltext_runner.py     # small-text adapter
│   ├── skactiveml_runner.py    # scikit-activeml adapter
│   ├── playground_runner.py    # playground adapter
│   └── optimal_runner.py       # Oracle strategies
├── optimal_query_strategies/    # Oracle/optimal implementations
├── metrics/                     # Metric computation
│   ├── base_metric.py          # Abstract metric class
│   ├── Standard_ML_Metrics.py  # Accuracy, F1, etc.
│   ├── Selected_Indices.py     # Track selected samples
│   ├── Timing_Metrics.py       # Query/training timing
│   └── computed/               # Post-hoc metrics
├── misc/                        # Shared utilities
│   ├── config.py               # Configuration management
│   ├── logging.py              # Logging utilities
│   ├── helpers.py              # Helper functions
│   └── plotting.py             # Plotting utilities
├── resources/                   # Configuration files
│   ├── exp_config.yaml         # Experiment definitions
│   ├── openml_datasets.yaml    # OpenML dataset configs
│   ├── kaggle_datasets.yaml    # Kaggle dataset configs
│   ├── data_types.py           # Enums and mappings
│   └── slurm_templates/        # SLURM job templates
├── scripts/                     # Utility scripts
├── eva_scripts/                 # Evaluation scripts
├── analyse_results/             # DEPRECATED
├── docs/                        # Documentation
└── tests/                       # Test files
```

---

## Coding Guidelines

### Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Keep lines under 100 characters
- Use descriptive variable names

### Docstrings

Use Google-style docstrings:

```bash
# Update Poetry dependencies
poetry update

# Regenerate conda lock files (requires conda-lock)
conda-lock --file environment.yml --platform linux-64 --platform osx-64 --platform osx-arm64 --platform win-64
```0

### Imports

Order imports as:
1. Standard library
2. Third-party packages
3. Local modules

```bash
# Update Poetry dependencies
poetry update

# Regenerate conda lock files (requires conda-lock)
conda-lock --file environment.yml --platform linux-64 --platform osx-64 --platform osx-arm64 --platform win-64
```1

---

## Pull Request Checklist

Before submitting a PR:

- [ ] Code follows project style guidelines
- [ ] All existing tests pass (`pytest`)
- [ ] New tests added for new functionality
- [ ] Type hints added for new functions
- [ ] Docstrings added for public functions
- [ ] Linting passes (`ruff check .`)
- [ ] Type checking passes (`mypy .`)
- [ ] Documentation updated if needed
- [ ] Commit messages are descriptive

### PR Template

```bash
# Update Poetry dependencies
poetry update

# Regenerate conda lock files (requires conda-lock)
conda-lock --file environment.yml --platform linux-64 --platform osx-64 --platform osx-arm64 --platform win-64
```2

---

## Common Development Tasks

### Adding a New Script

1. Create the script file (e.g., `06_new_script.py`)
2. Import and use the `Config` class for configuration
3. Follow the existing patterns from other numbered scripts
4. Add documentation to [`docs/pipeline.md`](https://github.com/jgonsior/olympic-games-of-active-learning/blob/main/docs/pipeline.md)

### Debugging Experiments

```bash
# Update Poetry dependencies
poetry update

# Regenerate conda lock files (requires conda-lock)
conda-lock --file environment.yml --platform linux-64 --platform osx-64 --platform osx-arm64 --platform win-64
```3

### Testing Configuration Changes

```bash
# Update Poetry dependencies
poetry update

# Regenerate conda lock files (requires conda-lock)
conda-lock --file environment.yml --platform linux-64 --platform osx-64 --platform osx-arm64 --platform win-64
```4

### Profiling Performance

```bash
# Update Poetry dependencies
poetry update

# Regenerate conda lock files (requires conda-lock)
conda-lock --file environment.yml --platform linux-64 --platform osx-64 --platform osx-arm64 --platform win-64
```5

---

## Reporting Issues

When reporting bugs, include:

1. **Environment**: OS, Python version, conda environment
2. **Command**: Exact command that failed
3. **Error**: Full error message and traceback
4. **Configuration**: Relevant config files (sanitized)
5. **Expected behavior**: What should have happened

### Bug Report Template

```bash
# Update Poetry dependencies
poetry update

# Regenerate conda lock files (requires conda-lock)
conda-lock --file environment.yml --platform linux-64 --platform osx-64 --platform osx-arm64 --platform win-64
```6
Paste full traceback here
```bash
# Update Poetry dependencies
poetry update

# Regenerate conda lock files (requires conda-lock)
conda-lock --file environment.yml --platform linux-64 --platform osx-64 --platform osx-arm64 --platform win-64
```7

---

## Contact

- **Repository**: [jgonsior/olympic-games-of-active-learning](https://github.com/jgonsior/olympic-games-of-active-learning)
- **Issues**: [GitHub Issues](https://github.com/jgonsior/olympic-games-of-active-learning/issues)
