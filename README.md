# AL Survey
```bash
python 00_dowload_datasets.py
python 01_create_workload.py --EXP_TITLE test_experiment --IGNORE_CONFIG_FILE --EXP_DATASETS 1 2 3 --EXP_STRATEGIES 5 2 --EXP_RANDOM_SEEDS_END 100
python 02_run_experiment.py --EXP_TITLE test_if_it_works --WORKER_INDEX 100
```
