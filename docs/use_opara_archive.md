# Use the Released OPARA Archive

The fastest way to work with OGAL — no experiments needed.

---

## 1. Download the archive

The full results are archived at [DOI:10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862) (~320 GB).

```bash
wget -c -O full_exp_jan.zip \
  "https://opara.zih.tu-dresden.de/bitstreams/38951489-5076-4544-a99b-c20dddfc2c6b/download"
unzip full_exp_jan.zip -d /path/to/results/full_exp_jan
```

## 2. Set up the environment

```bash
git clone https://github.com/jgonsior/olympic-games-of-active-learning.git
cd olympic-games-of-active-learning
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install
cp .server_access_credentials.cfg.example .server_access_credentials.cfg
# Edit .server_access_credentials.cfg → set OUTPUT_PATH under [LOCAL]
```

## 3. Generate the leaderboard

```bash
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

See [Results format](results_format.md) for the output schema and [Evaluation scripts](evaluation_scripts.md) for the full list of analysis scripts.

---

**Next:** [Run a local smoke test](run_local_smoke_test.md) · [Datasets & provenance](datasets_and_provenance.md) · [Evaluation scripts](evaluation_scripts.md)
