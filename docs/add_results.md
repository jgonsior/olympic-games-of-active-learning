# Add results

Contribute new experiments to OGAL and regenerate artifacts.

---

## Essentials (where, what, how)

- **Run identity:** `(EXP_DATASET, EXP_STRATEGY, EXP_LEARNER_MODEL, EXP_BATCH_SIZE, EXP_START_POINT, EXP_TRAIN_TEST_BUCKET_SIZE, EXP_RANDOM_SEED)` + `EXP_UNIQUE_ID`.
- **Place results at:** `${OGAL_OUTPUT}/<EXP_TITLE>/`
  - `05_done_workload.csv`, `05_failed_workloads.csv` (if any)
  - `<STRATEGY>/<DATASET>/accuracy.csv.xz`, `weighted_f1-score.csv.xz`, optional `selected_indices.csv.xz`, `query_selection_time.csv.xz`
- **Validate:** `python scripts/validate_results_schema.py --results_path ${OGAL_OUTPUT}/<EXP_TITLE> [--compare_with ${OGAL_OUTPUT}/full_exp_jan]`
- **Regenerate artifacts:** metrics + plots via steps below.

---

## Minimal flow

```bash
# 1) Create workload
python 01_create_workload.py --EXP_TITLE my_new_experiment

# 2) Run experiments (local or HPC)
python 02_run_experiment.py --EXP_TITLE my_new_experiment --WORKER_INDEX 0

# 3) Validate schema + duplicates
export OGAL_OUTPUT=/path/to/results
python scripts/validate_results_schema.py --results_path ${OGAL_OUTPUT}/my_new_experiment
python scripts/validate_results_schema.py --results_path ${OGAL_OUTPUT}/my_new_experiment --compare_with ${OGAL_OUTPUT}/full_exp_jan

# 4) Post-process + regenerate artifacts
python 03_calculate_dataset_categorizations.py --EXP_TITLE my_new_experiment --SAMPLES_CATEGORIZER _ALL --EVA_MODE local
python 04_calculate_advanced_metrics.py --EXP_TITLE my_new_experiment --COMPUTED_METRICS _ALL --EVA_MODE local
python -m eva_scripts.final_leaderboard --EXP_TITLE my_new_experiment
```

Worked example: add a new strategy → follow enum + mapping steps in code, then rerun the flow above. See [Runbook](reference/runbook.md) for HPC tips.
