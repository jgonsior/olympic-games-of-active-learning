# AL Survey
```bash
python 00_dowload_datasets.py
python 01_create_workload.py --EXP_TITLE test_experiment --IGNORE_CONFIG_FILE --EXP_DATASETS 1 2 3 --EXP_STRATEGIES 5 2 --EXP_RANDOM_SEEDS_END 100
# or alternatively using the yaml file:
python 01_create_workload.py --USE_EXP_YAML test_exp_2 python 01_create_workload.py --USE_EXP_YAML test_exp_2
python 02_run_experiment.py --EXP_TITLE test_experiment --WORKER_INDEX 100
```


# Example `.server_access_credentials.cfg` (part of `.gitignore` for obvious reasons)

```ini
[HPC]
SSH_LOGIN=user@hpc_server
WS_PATH=/some/path/to/a/workspace
DATASETS_PATH =/some/path/to/the/datasets
OUTPUT_PATH =/some/path/where/to/store/the/results
SLURM_MAIL=your.name@example.org
SLURM_PROJECT="project title"

[LOCAL]
DATASETS_PATH = /home/your_name/Projects/al_survey/datasets
LOCAL_CODE_PATH=/home/your/name/Projects/al_survey/code
OUTPUT_PATH = /home/your_name/Projects/al_survey/exp_results

```
