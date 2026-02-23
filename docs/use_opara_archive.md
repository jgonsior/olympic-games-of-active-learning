# Use the Released OPARA Archive

The fastest way to work with OGAL — download the 4.6M pre-computed experiment
results and start analysing immediately. No experiments needed.

**DOI:** [10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)

---

## 1. Download the artifacts

The archive contains two files.  Both commands use `-c` so you can resume an
interrupted download.

### File manifest (~184 MB)

=== "wget"

    ```bash
    wget -c -O archive_listing.txt \
      "https://opara.zih.tu-dresden.de/bitstreams/0f4dcc0e-4ba7-4b51-b3ed-778bbbd0c945/download"
    ```

=== "aria2c"

    ```bash
    aria2c -c -o archive_listing.txt \
      "https://opara.zih.tu-dresden.de/bitstreams/0f4dcc0e-4ba7-4b51-b3ed-778bbbd0c945/download"
    ```

### Main results archive (~320 GB)

!!! warning "Disk space"
    You need **~320 GB** of free space for the zip and roughly the same again
    for the extracted files.

=== "wget"

    ```bash
    wget -c -O full_exp_jan.zip \
      "https://opara.zih.tu-dresden.de/bitstreams/38951489-5076-4544-a99b-c20dddfc2c6b/download"
    ```

=== "aria2c"

    ```bash
    aria2c -c -o full_exp_jan.zip \
      "https://opara.zih.tu-dresden.de/bitstreams/38951489-5076-4544-a99b-c20dddfc2c6b/download"
    ```

---

## 2. Extract

```bash
export RESULTS_DIR=/path/to/results   # use the same value as OUTPUT_PATH in .server_access_credentials.cfg
unzip full_exp_jan.zip -d "${RESULTS_DIR}/full_exp_jan"
```

---

## 3. Verify

```bash
# archive_listing.txt must be present
ls archive_listing.txt

# (Optional) Compare the manifest against the extracted tree
# diff <(sort archive_listing.txt) <(find "${RESULTS_DIR}/full_exp_jan" -type f | sort)
```

---

## 4. Set up the environment

```bash
git clone https://github.com/jgonsior/olympic-games-of-active-learning.git
cd olympic-games-of-active-learning
conda create --name ogal --file conda-linux-64.lock && conda activate ogal && poetry install
cp .server_access_credentials.cfg.example .server_access_credentials.cfg
# Edit .server_access_credentials.cfg → set OUTPUT_PATH under [LOCAL] to your RESULTS_DIR
```

## 5. Generate the leaderboard

```bash
python -m eva_scripts.final_leaderboard --EXP_TITLE full_exp_jan
```

The paper uses config **`full_exp_jan`** — see [Paper subset](paper_subset.md) for
the full experiment grid and evaluation scripts.

See [Results format](results_format.md) for the output schema and [Evaluation scripts](evaluation_scripts.md) for the full list of analysis scripts.

!!! tip "If OPARA migrates"
    The DOI landing page at <https://doi.org/10.25532/OPARA-862> is canonical.
    If bitstream URLs change, retrieve the updated ones from that page.

---

**Next:** [Run a local smoke test](run_local_smoke_test.md) · [Datasets & provenance](datasets_and_provenance.md) · [Evaluation scripts](evaluation_scripts.md)
