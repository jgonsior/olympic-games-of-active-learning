# Run a Local Smoke Test

A quick sanity check — not a paper reproduction. Uses the built-in `test` config
(2 datasets, 4 strategies — finishes in minutes).
To reproduce the paper results, download the [OPARA archive](use_opara_archive.md) instead.

---

## 1. Install

```bash
git clone https://github.com/jgonsior/olympic-games-of-active-learning.git
cd olympic-games-of-active-learning
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install
```

See the [README](https://github.com/jgonsior/olympic-games-of-active-learning#quickstart)
for other platform lock files (macOS, Windows).

## 2. Create local config

```bash
cp .server_access_credentials.cfg.example .server_access_credentials.cfg
```

Edit `.server_access_credentials.cfg` and set two paths under `[LOCAL]`:

| Key | Example |
|-----|---------|
| `OUTPUT_PATH` | `/home/you/ogal_results` |
| `DATASETS_PATH` | `/home/you/ogal_datasets` |

Create both directories before continuing.
See [Configuration](configuration.md) for all keys.

## 3. Download datasets

```bash
python 00_download_datasets.py
```

!!! note
    **OpenML** datasets download without credentials.
    **Kaggle** datasets require an API token — see
    [Datasets & provenance → Kaggle token setup](datasets_and_provenance.md#kaggle-token-setup).
    For a quick test you can skip Kaggle; the `test` config only uses OpenML datasets.

## 4. Generate the workload

```bash
python 01_create_workload.py --EXP_TITLE test
```

This reads the `test` block in `resources/exp_config.yaml` and writes
`01_workload.csv` to your `OUTPUT_PATH/test/` directory.

## 5. Run one experiment

```bash
python 02_run_experiment.py --EXP_TITLE test --WORKER_INDEX 0
```

## 6. Verify output

```bash
ls "$(grep OUTPUT_PATH .server_access_credentials.cfg | head -1 | cut -d= -f2 | tr -d ' ')/test/"
```

You should see files like:

```
01_workload.csv
05_done_workload.csv
ALIPY_RANDOM/
```

Inside each strategy folder there are per-dataset CSV result files.
See [Results format](results_format.md) for the full schema.

## 7. (Optional) Run evaluation

If you have run **all** workers (not just index 0), you can generate a
leaderboard:

```bash
python -m eva_scripts.final_leaderboard --EXP_TITLE test
```

For more evaluation scripts see [Evaluation scripts](evaluation_scripts.md).

---

**Next:** [Use OPARA archive](use_opara_archive.md) · [Run at HPC scale](run_hpc.md) · [Pipeline](pipeline.md)
