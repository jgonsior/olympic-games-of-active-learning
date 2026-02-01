# Extend the Benchmark

**Add your experiments (new strategies/datasets/hyperparameters) and integrate them with the shared benchmark.**

---

## What You Can Add

| Extension | Difficulty | Time |
|-----------|------------|------|
| New hyperparameter values | Easy | Minutes |
| New dataset | Easy | 30 min |
| New learner model | Medium | 1 hour |
| New AL strategy | Medium | 1-2 hours |

---

## Quick Wins

### Add a New Batch Size

```yaml
# resources/exp_config.yaml
my_experiment:
  EXP_GRID_BATCH_SIZE: [1, 5, 10, 20, 50, 100, 200]  # Added 200
```

```bash
python 01_create_workload.py --EXP_TITLE my_experiment
python 02_run_experiment.py --EXP_TITLE my_experiment --WORKER_INDEX 0
```

### Add a New Dataset (OpenML)

```yaml
# resources/openml_datasets.yaml
my_new_dataset:
  data_id: 12345  # OpenML ID
```

```yaml
# resources/exp_config.yaml
my_experiment:
  EXP_GRID_DATASET: [my_new_dataset, Iris]
```

---

## Add a New AL Strategy

**Step 1:** Add to enum (`resources/data_types.py`):

```python
class AL_STRATEGY(IntEnum):
    MY_CUSTOM_STRATEGY = 100  # Choose unused ID
```

**Step 2:** Add mapping:

```python
al_strategy_to_python_classes_mapping[AL_STRATEGY.MY_CUSTOM_STRATEGY] = (
    MyStrategyClass, {"param": "value"}
)
```

**Step 3:** Use in experiment:

```yaml
EXP_GRID_STRATEGY: [ALIPY_RANDOM, MY_CUSTOM_STRATEGY]
```

---

## Validate and Post-Process

```bash
# Validate schema
python scripts/validate_results_schema.py --results_path ${OGAL_OUTPUT}/my_experiment

# Post-process
python 03_calculate_dataset_categorizations.py --EXP_TITLE my_experiment --SAMPLES_CATEGORIZER _ALL --EVA_MODE local
python 04_calculate_advanced_metrics.py --EXP_TITLE my_experiment --COMPUTED_METRICS _ALL --EVA_MODE local

# Generate leaderboard
python -m eva_scripts.learning_curve --EXP_TITLE my_experiment
python -m eva_scripts.final_leaderboard --EXP_TITLE my_experiment
```

---

## Keep Results Separate (Recommended)

Run evaluation scripts separately, then compare:

```bash
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
python -m eva_scripts.final_leaderboard --EXP_TITLE my_experiment
```

---

## Next Steps

| Goal | Page |
|------|------|
| Run at HPC scale | [Run from Scratch](run_from_scratch.md) |
| Understand the architecture | [Architecture & Design](understand_codebase.md) |
| Development guidelines | [Contributing](../contributing.md) |
