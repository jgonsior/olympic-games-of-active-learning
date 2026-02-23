# Datasets & Provenance

OGAL experiments cover **92 tabular classification datasets** sourced from OpenML and Kaggle.

---

## Dataset sources

| Source | Config file | Notes |
|--------|-------------|-------|
| OpenML | `resources/openml_datasets.yaml` | Downloaded automatically by `00_download_datasets.py` |
| Kaggle | `resources/kaggle_datasets.yaml` | Requires a Kaggle API token (see below) |

The full list of datasets used in the paper is in `dataset_list.txt`.

---

## Kaggle token setup

Kaggle datasets require an API token. Without it, `00_download_datasets.py`
will fail for any Kaggle-sourced dataset.

### Create and install the token

1. Create a free account at <https://www.kaggle.com>.
2. Go to **Account → API → Create New Token** — this downloads a file called `kaggle.json`.
3. Place it in the expected location and lock down permissions:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

`00_download_datasets.py` picks up the token automatically.

!!! danger "Never commit credentials"
    `kaggle.json` contains your personal API key.
    **Do not** add it to version control, copy it into the repo directory, or
    share it in logs.  The file belongs only in `~/.kaggle/`.

### Common failures and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `OSError: Could not find kaggle.json` | Token file missing or wrong path | Place it at `~/.kaggle/kaggle.json` |
| `403 Forbidden` | Token expired or revoked | Generate a new token on kaggle.com |
| `Permission denied` on `kaggle.json` | File permissions too open | Run `chmod 600 ~/.kaggle/kaggle.json` |
| Download succeeds but dataset is empty/corrupt | Kaggle competition requires accepting rules | Accept the competition rules on kaggle.com first |

---

## OpenML notes

OpenML datasets download without credentials.  A few things to keep in mind
for reproducibility:

- Datasets are referenced by **numeric IDs** (e.g., `data_id: 31` for `credit-g`)
  listed in `resources/openml_datasets.yaml`.
- OpenML datasets can have **multiple versions**.  The version fetched depends on
  the OpenML API default at download time unless a specific version is pinned in
  the YAML file.
- **Recommendation:** record the dataset IDs, the OpenML dataset versions
  actually downloaded, and your repository commit hash.  This makes it possible
  to reproduce the exact dataset snapshot later.

The OPARA archive already contains all pre-computed splits, so you only need
to re-download from OpenML if you are regenerating datasets from scratch.

---

## Dataset licensing & terms

Datasets are hosted by third parties (OpenML, Kaggle) under their own licenses.
OGAL does not redistribute the raw data — `00_download_datasets.py` fetches
each dataset directly from its source.

Before using a dataset in published work, check its license on the source
platform.  Some Kaggle datasets require accepting competition rules or have
non-commercial restrictions.

---

## Dataset version stability

- **OpenML** may update a dataset version silently.  Pin the version in
  `resources/openml_datasets.yaml` if exact reproducibility matters.
- **Kaggle** datasets can be updated by their owners at any time.  The OPARA
  archive contains the exact dataset snapshots used in the paper — use those
  for full reproducibility rather than re-downloading.
- When adding a new dataset, note the version/date in your commit message so
  future users can identify which snapshot was used.

---

## Dataset IDs

!!! warning "Dataset IDs are calculated at runtime"
    Unlike strategies, **dataset IDs are assigned dynamically** based on the order
    datasets appear in the YAML files. See [Extend OGAL](personas/extend_benchmark.md)
    for details.

---

**See also:** [Splits & start sets](splits_and_start_sets.md) · [Configuration](configuration.md) · [Pipeline](pipeline.md)
