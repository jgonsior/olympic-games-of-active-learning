# OGAL - Olympic Games of Active Learning

## Installation
Requierements:

[anaconda](https://docs.anaconda.com/anaconda/install/index.html) installed.

```bash
# to be run in the root directory of this project
conda init
conda create --name al_olympics_conda python=3.10
conda activate al_olympics_conda
conda install -c anaconda cython
conda install -c conda-forge cvxpy pipenv liblapacke libsvm modin-ray
#conda install -c conda-forge modin-ray
conda env export
pipenv --python=$(conda run which python) --site-packages install --dev
# OR
python -m pipenv --python=$(conda run which python) --site-packages install --dev
python -m pipenv shell
#pip install modin[ray] # has to be run everytime anything is being changed by pipenv
pip install git+https://github.com/ntucllab/libact.git "modin[ray]"
```


HPC:
```bash
module load Anaconda3
sh $EBROOTANACONDA3/etc/profile.d/conda.sh
conda create --prefix $WS_URL/al_olympics/conda-env python=3.10
conda activate $WS_URL/-al_olympics/conda-env
conda install -c anaconda cython -y
conda install -y -c conda-forge cvxpy pipenv liblapacke
python -m pipenv --python=$(conda run which python) --site-packages install --dev
python -m pipenv --python=$(conda run which python) --site-packages install --dev
pip install git+https://github.com/jgonsior/libact.git
```


## Usage

```bash
python 00_download_datasets.py
python 01_create_workload.py --EXP_TITLE test_experiment --IGNORE_CONFIG_FILE --EXP_DATASETS 1 2 3 --EXP_STRATEGIES 5 2 --EXP_RANDOM_SEEDS_END 100
# or alternatively using the yaml file:
python 01_create_workload.py --EXP_TITLE test_exp_2
python 02_run_experiment.py --EXP_TITLE test_experiment --WORKER_INDEX 100
```


## Example `.server_access_credentials.cfg` (part of `.gitignore` for obvious reasons)

```ini
[HPC]
SSH_LOGIN=user@hpc_server
WS_PATH=/some/path/to/a/workspace
DATASETS_PATH =/some/path/to/the/datasets
OUTPUT_PATH =/some/path/where/to/store/the/results
SLURM_MAIL=your.name@example.org
SLURM_PROJECT="project title"
CODE_PATH=/some/path/where/the/code/should/be
PYTHON_PATH=/the/path/of/the/python/interpreter

[LOCAL]
DATASETS_PATH = /home/your_name/Projects/al_survey/datasets
LOCAL_CODE_PATH=/home/your/name/Projects/al_survey/code
OUTPUT_PATH = /home/your_name/Projects/al_survey/exp_results

```
