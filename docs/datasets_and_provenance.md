# Datasets & Provenance

OGAL experiments cover **92 tabular classification datasets** sourced from OpenML and Kaggle.

---

## Dataset sources

| Source | Config file | Notes |
|--------|-------------|-------|
| OpenML | `resources/openml_datasets.yaml` | Downloaded automatically by `00_download_datasets.py` |
| Kaggle | `resources/kaggle_datasets.yaml` | Requires a Kaggle API token (see below) |

The full list of datasets used in the paper is in `dataset_list.txt`.

## Kaggle token setup

1. Create a Kaggle account at <https://www.kaggle.com>.
2. Go to **Account → API → Create New Token** — this downloads `kaggle.json`.
3. Place it at `~/.kaggle/kaggle.json` and set permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

`00_download_datasets.py` picks up the token automatically.

## Dataset IDs

!!! warning "Dataset IDs are calculated at runtime"
    Unlike strategies, **dataset IDs are assigned dynamically** based on the order
    datasets appear in the YAML files. See [Extend OGAL](personas/extend_benchmark.md)
    for details.

---

**See also:** [Splits & start sets](splits_and_start_sets.md) · [Configuration](configuration.md) · [Pipeline](pipeline.md)
