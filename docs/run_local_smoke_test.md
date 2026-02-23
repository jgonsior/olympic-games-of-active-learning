# Run a Local Smoke Test

Run a tiny experiment end-to-end on your laptop to understand the pipeline.

---

## 1. Set up the environment

```bash
git clone https://github.com/jgonsior/olympic-games-of-active-learning.git
cd olympic-games-of-active-learning
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install
cp .server_access_credentials.cfg.example .server_access_credentials.cfg
# Edit .server_access_credentials.cfg → set OUTPUT_PATH and DATASETS_PATH under [LOCAL]
```

## 2. Download datasets

```bash
python 00_download_datasets.py
```

## 3. Create a minimal workload

Add a small block to `resources/exp_config.yaml`:

```yaml
smoke_test:
  EXP_GRID_DATASET: [Iris]
  EXP_GRID_STRATEGY: [ALIPY_RANDOM]
  EXP_GRID_RANDOM_SEED: [0]
  EXP_GRID_NUM_QUERIES: [10]
  EXP_GRID_BATCH_SIZE: [5]
  EXP_GRID_LEARNER_MODEL: [RF]
  EXP_GRID_TRAIN_TEST_BUCKET_SIZE: [0]
  EXP_GRID_START_POINT: [0]
  METRICS: [Standard_ML_Metrics]
```

## 4. Run it

```bash
python 01_create_workload.py --EXP_TITLE smoke_test
python 02_run_experiment.py  --EXP_TITLE smoke_test --WORKER_INDEX 0
```

See [Pipeline](pipeline.md) for the full step-by-step and [Configuration](configuration.md) for all config keys.

---

**Next:** [Use OPARA archive](use_opara_archive.md) · [Run at HPC scale](run_hpc.md) · [Configuration](configuration.md)
