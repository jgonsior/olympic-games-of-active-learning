# OGAL — Start here

OGAL (Olympic Games of Active Learning) is a large-scale benchmark that
systematically evaluates Active Learning query strategies across hundreds of
datasets and hyperparameter configurations. The archive contains **4.6 million
pre-computed experiments** so you can analyze results without running anything
yourself. DOI: [10.25532/OPARA-862](https://doi.org/10.25532/OPARA-862)

---

## In 10 minutes: analyze the released results

### 1 — Download the archive

Both `wget -c` and `aria2c -c` support resuming if the connection drops.

**File manifest** (~184 MB):

```bash
# wget
wget -c -O archive_listing.txt \
  "https://opara.zih.tu-dresden.de/bitstreams/0f4dcc0e-4ba7-4b51-b3ed-778bbbd0c945/download"

# or aria2c
aria2c -c -o archive_listing.txt \
  "https://opara.zih.tu-dresden.de/bitstreams/0f4dcc0e-4ba7-4b51-b3ed-778bbbd0c945/download"
```

**Full results archive** (~320 GB):

```bash
# wget
wget -c -O full_exp_jan.zip \
  "https://opara.zih.tu-dresden.de/bitstreams/38951489-5076-4544-a99b-c20dddfc2c6b/download"

# or aria2c
aria2c -c -o full_exp_jan.zip \
  "https://opara.zih.tu-dresden.de/bitstreams/38951489-5076-4544-a99b-c20dddfc2c6b/download"
```

!!! note "Disk space"
    You need **~320 GB** for the zip and roughly the same again after extraction.

### 2 — Extract

```bash
export RESULTS_DIR=/path/to/results
unzip full_exp_jan.zip -d "${RESULTS_DIR}/full_exp_jan"
```

### 3 — Verify

- [ ] `archive_listing.txt` exists in the current directory.
- [ ] `${RESULTS_DIR}/full_exp_jan/` directory exists and contains result files.

---

## What to read next

| Page | When to read |
|------|-------------|
| [Run & reproduce](run.md) | Re-run experiments yourself (locally or on HPC) |
| [Results & evaluation](results.md) | Understand the output format and evaluation scripts |
| [Strategies & extending](extend.md) | Add a new query strategy or dataset |

📄 [Paper](https://arxiv.org/abs/2506.03817) ・ 📦 [Dataset (DOI)](https://doi.org/10.25532/OPARA-862) ・ 💻 [GitHub](https://github.com/jgonsior/olympic-games-of-active-learning)
