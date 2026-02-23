# Documentation and Reproducibility Audit of olympic-games-of-active-learning

## Executive summary

The project provides a substantial codebase and a fairly extensive MkDocs documentation set, but there are several **critical reproducibility and consistency gaps** that would prevent an independent researcher from reliably reproducing the paper’s results end-to-end without additional tribal knowledge.

The most important findings are:

- The associated dataset release on entity["organization","OPARA","saxon universities data repo"] includes **`full_exp_jan.zip` (~320 GB)** and an **`archive_listing.txt` (~184 MB)**, but the README/docs currently use placeholders rather than providing the exact download links and verification guidance. citeturn7view0
- The documentation claims a simple local workflow based on an `OGAL_OUTPUT` environment variable, but **no code path actually reads `OGAL_OUTPUT`** (the code instead reads a local, gitignored config file). This is a high-friction “paper cut” that turns into a hard blocker for first-time users.
- The “download datasets” entrypoint **`00_download_datasets.py` is effectively disabled** (large sections commented out / `continue` inside key loops). This directly conflicts with the docs that present it as a working step in the pipeline.
- The paper describes a **4.6M configuration grid**, about **3.6M CPU hours**, and a **5-minute per-cycle runtime limit**, which are reflected in parts of the code and docs, but key operational details needed for reproduction (exact seed handling, full configuration-to-result provenance, minimal smoke-test subset) remain incomplete or ambiguous. citeturn7view0
- Several internal references are broken/missing (docstrings point to non-existent docs like `docs/pipeline.md`, `docs/configuration.md`, `docs/results_format.md`), and core “researcher needs” like **versioning/release discipline, CI, tests, and pinned git dependencies** are missing or under-specified.

## Scope and sources reviewed

Primary sources examined:

- Repository code, README, and MkDocs sources retrieved via the GitHub connector (limited to `jgonsior/olympic-games-of-active-learning`, as requested).
- The attached paper (PDF provided by you). fileciteturn0file0
- The corresponding public dataset landing page on entity["organization","OPARA","saxon universities data repo"], which lists the released artifacts and their sizes. citeturn7view0
- General service context for entity["organization","OPARA","saxon universities data repo"] (noting platform migration caveats that can affect reproducibility if links or old items move). citeturn6search1

Limitations to call out explicitly:

- I reviewed the **MkDocs source files in-repo** rather than the deployed GitHub Pages HTML, because the web fetch layer could not reliably retrieve the `jgonsior.github.io` pages in this environment. The audit therefore treats the repo’s `docs/` directory + `mkdocs.yml` as the authoritative docs source.

## Consistency review between paper, repository, and documentation

The paper positions the work as a large-scale empirical study of active learning hyperparameters, built from a 4,636,800-combination experimental grid, run at high computational cost, with runtime constraints per active-learning cycle. fileciteturn0file0 The dataset release supports this by publishing the large raw result archive. citeturn7view0

The table below focuses on **high-load-bearing** factual statements that should match across paper ↔ docs ↔ code.

| Topic | What the paper says | What the repo/docs/code say | Issues found |
|---|---|---|---|
| Scale of grid | Hyperparameter grid of ~4.6M combinations (explicitly 4,636,800). fileciteturn0file0 | README and docs repeatedly call out “4.6M” precomputed experiments; YAML experiment configs contain the large combinatorial grid. | Generally consistent. |
| Compute budget | ~3.6 million CPU hours. fileciteturn0file0 | Not prominently stated in README quickstart; mentioned in docs only indirectly via “large scale / HPC”. | **Omission:** compute-budget context is important for setting expectations and motivating runtime limits. |
| Runtime limit | Enforced 5 minutes per AL cycle; configurations exceeding are excluded. fileciteturn0file0 | Code uses a 300-second timeout setting (naming suggests “query selection limit”), and docs mention increasing it for timeouts. | Mostly consistent, but **wording mismatch**: code applies timeout to the entire cycle (not only query selection). Docs should reflect that precisely. |
| Datasets and repetitions | 92 datasets from UCI/Kaggle/OpenML; 5 train-test splits and 20 start sets per split (100 repeats per dataset). fileciteturn0file0 | Repo contains `dataset_list.txt`, OpenML/Kaggle YAMLs, and experiment configs reflecting 5 buckets and 20 start points for the main run. | Docs mention dataset sources, but **do not clearly document dataset versioning, licensing constraints (Kaggle), and the exact split/start-set generation procedure** needed to reproduce. |
| Frameworks used | Paper explicitly mentions combining multiple frameworks (ALiPy, libact, Google Playground, scikit-activeml, Small-Text). fileciteturn0file0 | Repo includes strategy enums spanning several frameworks (and docs sometimes claim 5 frameworks, elsewhere 6). | **Inconsistency:** “5 vs 6 frameworks” and “50+ vs 76 strategies” messaging differs across README/docs. Clarification needed: “paper subset” vs “full benchmark support”. |
| Strategies used in paper | Ultimately 28 strategies used after exclusions; reasons include errors, binary-only design, or high runtime. fileciteturn0file0 | Repo’s code enumerates many more strategies; main config for the paper run includes a subset. | Docs should explicitly identify “paper strategy set” by name (not just “full_exp_jan”). |
| Data release | Raw experiment results published on OPARA. fileciteturn0file0 | OPARA item lists `full_exp_jan.zip` and `archive_listing.txt` with sizes; docs/README refer to “download from DOI” but use placeholders. citeturn7view0 | **Critical docs issue:** placeholders make reproduction brittle, especially given OPARA platform migration caveats. citeturn6search1 |
| Citation metadata | Paper title and arXiv metadata should agree with repository citation instructions. fileciteturn0file0 | Repo includes BibTeX and `CITATION.cff`, but observed title/metadata do not appear consistently aligned with the attached PDF’s title. | **Potential factual inconsistency:** citation title/arXiv metadata should be verified and unified. |

### High-impact internal inconsistencies or factual errors

These are issues where **the docs make a concrete operational claim**, but the repo code/config does not support it as written.

- **`OGAL_OUTPUT` environment variable is documented, but not implemented.** The docs recommend `export OGAL_OUTPUT=/path/to/results`, yet the configuration loading in code relies on a gitignored `.server_access_credentials.cfg` (and no repository code references `OGAL_OUTPUT`). This makes the quickstart misleading.
- **The dataset download step is presented as working, but `00_download_datasets.py` is effectively inert.** Large portions are commented out and key loops short-circuit, so a user following the docs will likely not end up with datasets + split files in the expected structure.
- **Broken documentation references in code docstrings.** Multiple modules point to docs pages that do not exist in the `docs/` tree (e.g., `docs/pipeline.md`, `docs/configuration.md`, `docs/results_format.md`). This is both a usability issue and a documentation maintenance smell.
- **License metadata mismatch.** The repo includes an AGPL license text and README claims AGPL, but packaging metadata uses a different license string. This is significant for citation and downstream reuse.

## Documentation completeness for reproducibility and extension

The table below evaluates whether the documentation supplies the minimum information a researcher typically needs.

Status legend: **Present**, **Partial**, **Unspecified**, **Missing**.

| Reproducibility element | Status | Notes on what is missing or unclear |
|---|---|---|
| Dependencies list | Partial | Conda lock + Poetry setup exists, but the docs do not clearly explain the relationship between conda-lock vs Poetry dependencies, nor do they highlight git-based dependencies as a reproducibility risk. |
| Environment setup | Partial | Platform-specific lock files exist; docs do not state required OS packages (if any) nor provide a standard “one command smoke test”. |
| Data sources | Partial | OpenML and Kaggle are sources, and OPARA is the results archive. Kaggle API authentication requirements and dataset license constraints are not documented clearly. |
| Exact dataset versions | Unspecified | Dataset sources are named, but version pinning / snapshotting strategy is not clearly described (especially key for Kaggle datasets). |
| Dataset preprocessing | Partial | Output formats are documented, but the precise preprocessing stages (missing value handling, encoding, scaling, class filtering, etc.) are not described in a researcher-facing way. |
| Split / start-set generation | Partial | The paper describes 5×20 repeats. The code generates train-test splits and start sets, but the docs do not document the exact algorithm, seed, and persistence format clearly enough to reproduce. |
| Training and evaluation entrypoints | Partial | Script names exist, but the full “from scratch” workflow is brittle due to the disabled download script and under-specified configuration file requirements. |
| Hyperparameters for paper run | Partial | Configs exist, but docs do not provide a single canonical “paper reproduction config” with a stable identifier, command lines, and expected outputs. |
| Random seeds and determinism | Partial | Docs claim determinism, but seed values and coverage across libraries/frameworks are not documented; deterministic behavior under multithreading is not discussed. |
| Expected runtimes and hardware | Partial | Paper gives macro compute cost fileciteturn0file0, but docs do not give runtimes for minimal reproductions (e.g., “tiny subset takes X minutes on laptop”). |
| Expected outputs | Partial | There is a good results-format page, but it would benefit from concrete “golden” examples and a minimal dataset run output tree. |
| Checkpoints / cached artifacts | Unspecified | No published checkpoints (not always applicable for AL on tabular) and no artifact manifest for partial replication. |
| Tests | Missing | No real unit/integration test suite, only scripts. A researcher cannot validate correctness beyond manual runs. |
| CI | Partial | Docs build workflow exists; no test/lint pipeline that enforces reproducibility and prevents doc rot. |
| Versioning / releases | Missing | No clear release process, changelog, or stable version tags for citation. |
| Citation guidance | Partial | There is BibTeX and `CITATION.cff`, but metadata consistency needs verification against the attached paper and OPARA entry. citeturn7view0 |
| License | Partial | License file exists; packaging metadata inconsistent; contribution docs do not explicitly call out licensing expectations for PRs. |
| Contribution guide | Present | Contributing page exists; should be aligned with actual repo state (tests/CI). |

### Reproducibility pipeline diagram

The documentation implicitly describes a pipeline, but it is spread across multiple pages. Consolidating it into a single canonical diagram (and aligning code/doc entrypoints) would reduce misunderstandings.

```mermaid
flowchart TD
  A[Install env: conda-lock + poetry] --> B[Configure local paths: .server_access_credentials.cfg or env vars]
  B --> C[Download & preprocess datasets (OpenML/Kaggle)]
  C --> D[Generate train/test splits & start sets]
  D --> E[Create workload CSV + (optional) SLURM scripts]
  E --> F[Run experiments: 02_run_experiment.py]
  F --> G[Raw metric CSV files per run]
  G --> H[Compress/merge; convert metrics to Parquet]
  H --> I[Evaluation scripts: leaderboard/plots/tables]
  I --> J[Paper figures / aggregated metrics]
```

## Prioritized actionable fixes for an autonomous LLM agent

The fixes below are ordered by “what most blocks an external researcher today”. Each fix includes concrete file targets and suggested snippets.

### Fix backlog table

| Priority | Effort | Fix | Files/URLs to modify | Suggested change |
|---|---|---|---|---|
| Critical | Low | Replace OPARA download placeholders with exact commands and add artifact verification guidance | `README.md`, `docs/personas/reproduce_and_run.md`, `docs/personas/analyze_dataset.md` | Add a “Data download” block using the OPARA bitstream URLs and advise `wget -c`/`aria2c` + checksum/manifest verification. Source of file names/sizes: OPARA item page lists `full_exp_jan.zip` and `archive_listing.txt`. citeturn7view0 |
| Critical | Medium | Make local runs work without a hidden config file, or document the config file as mandatory and provide an example template | `docs/personas/reproduce_and_run.md`, `README.md`, add new `.server_access_credentials.cfg.example`, update `misc/config.py` (optional) | Either (A) implement `OGAL_OUTPUT` override in `Config` or (B) remove env-var claims and standardize on `.server_access_credentials.cfg` with a committed example file. |
| Critical | Medium | Restore or replace the “download datasets” entrypoint so the documented workflow is executable | `00_download_datasets.py`, `docs/personas/reproduce_and_run.md` | Un-comment/repair dataset download logic; or split into explicit scripts: `00a_download_openml.py`, `00b_download_kaggle.py`, `00c_generate_splits.py`, and update docs accordingly. |
| High | Low | Fix config-key naming mismatch in docs (e.g., `EXP_GRID_NR_QUERIES` vs actual) | `docs/personas/extend_benchmark.md` | Replace incorrect key names with those used in `resources/exp_config.yaml` and explain where keys are defined. |
| High | Low | Add a canonical “paper reproduction config” section that maps paper nomenclature to config names and exact CLI commands | `README.md`, `docs/personas/reproduce_and_run.md` | Add a dedicated sub-section: “Reproducing the paper run: `full_exp_jan`” including commands, expected output tree, and known runtime expectations. |
| High | Medium | Clarify and harden seed handling and determinism claims | `docs/personas/reproduce_and_run.md`, `docs/personas/analyze_dataset.md`, `misc/config.py`, runners | Document which seeds are set (Python, NumPy, and any framework-specific seeds), where they originate, and what remains non-deterministic. |
| High | Low | Unify project citation metadata (title/authors/arXiv/DOI) across README, `CITATION.cff`, and the attached paper/OPARA entry | `README.md`, `CITATION.cff` | Verify the canonical title and identifiers and update inconsistencies. Use OPARA DOI for dataset citation. citeturn7view0 |
| High | Low | Fix license metadata mismatch in packaging | `pyproject.toml`, `README.md` | Ensure `pyproject.toml` license field matches AGPL-3.0 (license file is AGPL). Add SPDX identifier where possible. |
| Medium | Medium | Add tests + CI so documentation cannot drift from code | Add `tests/`, add `.github/workflows/ci.yml`, update `docs/contributing.md` | Implement minimal unit tests (config parsing, dataset enum generation, result filename format) and CI (pytest + ruff). |
| Medium | Low | Add missing docs pages referenced by docstrings, or update docstrings to point to actual pages | Add `docs/pipeline.md`, `docs/configuration.md`, `docs/results_format.md` or adjust docstrings | Prefer adding pages so code references remain useful; link them in `mkdocs.yml` nav and cross-link from personas. |
| Medium | Medium | Document dataset preprocessing precisely (including scaling, encoding, missing values) and tie it to the paper’s dataset statistics | `docs/personas/analyze_dataset.md`, add new `docs/datasets.md` | Add a “Preprocessing contract” section and explain potential impacts on results. |
| Low | Medium | Improve project versioning and archival for citation | Add `CHANGELOG.md`, add release guidance in docs, optionally add GitHub release workflow | Establish a minimal semantic versioning policy and a “Cite this version” section. |

### Concrete snippets and file contents

#### Replace OPARA placeholders with exact download instructions

Add to `README.md` (and mirrored in `docs/personas/reproduce_and_run.md`) a block like:

```bash
# Download the released raw experiment archive from OPARA (DOI: 10.25532/OPARA-862)

# Manifest/listing (helps verify contents and supports resumable workflows)
wget -c -O archive_listing.txt \
  "https://opara.zih.tu-dresden.de/bitstreams/0f4dcc0e-4ba7-4b51-b3ed-778bbbd0c945/download"

# Main archive (~320+ GB). Use -c for resume.
wget -c -O full_exp_jan.zip \
  "https://opara.zih.tu-dresden.de/bitstreams/38951489-5076-4544-a99b-c20dddfc2c6b/download"

# Unpack into your results directory
unzip full_exp_jan.zip -d "${RESULTS_DIR}/full_exp_jan"
```

Then add guidance:

- “This archive is very large; prefer a machine with sufficient disk space (Unspecified in current docs).”
- “Verify expected files against `archive_listing.txt`.”  
- “If OPARA migrates again, use the DOI landing page to locate the newest bitstream URLs.” citeturn6search1turn7view0

#### Make configuration discoverable and runnable

Create a committed template file: `.server_access_credentials.cfg.example`

```ini
[LOCAL]
# Required for local runs:
OUTPUT_PATH=/absolute/path/to/results
DATASETS_PATH=/absolute/path/to/datasets
CACHE_DIR=/absolute/path/to/cache

# Optional convenience:
CODE_PATH=/absolute/path/to/repo

[HPC]
# Only needed for the SLURM pipeline:
SSH_LOGIN=your_login@your_cluster
WS_PATH=/absolute/path/to/workspace
PYTHON_PATH=/absolute/path/to/python
OUTPUT_PATH=/absolute/path/to/results
DATASETS_PATH=/absolute/path/to/datasets
SLURM_PROJECT=your_project
SLURM_MAIL=your_email@example.com
```

Then update `docs/personas/reproduce_and_run.md` to instruct:

- copy example to `.server_access_credentials.cfg`
- replace paths
- explain which keys are required for local-only usage vs HPC usage

Also update `README.md` quickstart so the first local run does not rely on undocumented implicit config.

Optional code quality improvement (if you prefer keeping the `OGAL_OUTPUT` UX): implement an override in `misc/config.py` such as:

```python
# pseudo-snippet to add inside Config._pathes_magic():
ogal_output = os.getenv("OGAL_OUTPUT")
if ogal_output:
    self.LOCAL_OUTPUT_PATH = ogal_output
```

If implementing that, the docs can keep `export OGAL_OUTPUT=...` and remain truthful.

#### Repair the dataset download pipeline

Given the current state of `00_download_datasets.py` (disabled logic), either:

- **Option A (preferred):** Refactor into explicit, testable scripts:
  - `00a_download_openml.py`  
  - `00b_download_kaggle.py`  
  - `00c_generate_splits.py`  
  Each with `--help` and a `--datasets` filter so researchers can run a small subset.

- **Option B (fastest):** Restore `00_download_datasets.py` by removing dead code paths, enabling calls to the OpenML/Kaggle loaders, and ensuring it writes the same dataset/split artifacts that downstream scripts expect.

In both cases, update docs to include:

- Kaggle API prerequisites (where to place kaggle.json, how to set env vars; avoid committing credentials).
- A “single dataset smoke run” section (“download one OpenML dataset by ID and generate splits”).

#### Fix mismatched config key names in docs

In `docs/personas/extend_benchmark.md`, replace all occurrences of incorrect keys (example: `EXP_GRID_NR_QUERIES`) with the actual keys used by the experiment config YAML (example: `EXP_GRID_NUM_QUERIES`).

Also add a small “Config schema” section that points users to:

- `resources/exp_config.yaml` (experiment grids)
- `misc/config.py` (runtime config defaults and how `Config` materializes derived values)

#### Add minimal tests and CI

Add:

- `tests/test_config_smoke.py` verifying `Config()` can be created with a temporary `.server_access_credentials.cfg` and that it resolves output paths.
- `tests/test_dataset_enum.py` verifying dataset enum extension order is stable given fixed YAML order.
- A GitHub Actions workflow `.github/workflows/ci.yml` running:
  - `ruff check .`
  - `pytest -q`

Update `docs/contributing.md` to match reality (what tests exist and how to run them).

#### Add missing referenced docs pages

Either add pages or fix references. The least disruptive approach is to add:

- `docs/pipeline.md` (canonical end-to-end workflow)
- `docs/configuration.md` (full config reference, including required keys)
- `docs/results_format.md` (formal results contract with examples)

Then update `mkdocs.yml` navigation to include them.

## Post-change verification checklist

The following items should be verified after implementing the changes above. Where expected outputs are unclear today, they are explicitly marked **Unspecified**.

Environment and static checks:

- `conda create -n ogal --file conda-linux-64.lock`  
  Expected: environment resolves without solver errors (exact packages are pinned).
- `conda activate ogal && poetry install`  
  Expected: installs without fetching moving targets beyond what `poetry.lock` specifies (verify git dependencies are pinned in the lock; otherwise outcome is **Unspecified**).
- `ruff check .`  
  Expected: passes (or known violations documented).
- `pytest -q`  
  Expected: passes (after introducing tests).

Docs build checks:

- `mkdocs build -s`  
  Expected: no missing pages, no broken internal nav, Mermaid/static rendering script runs if required by workflow.
- Link checks (manual or automated): OPARA links, arXiv/DOI, internal cross-references.

Local smoke test (minimal reproducibility):

- Create `.server_access_credentials.cfg` from example and set only LOCAL paths.
- Download exactly one dataset (OpenML-only path) and generate splits/start sets.
  - Expected: dataset CSV appears in `${DATASETS_PATH}` (exact filename currently **Unspecified** in docs; should be specified after fixes).
  - Expected: train/test split file(s) appear with consistent naming.
- Run a tiny workload:
  - `python 01_create_workload.py --config <tiny_config_name>` (or the repo’s actual CLI, if none exists today)  
  - `python 02_run_experiment.py --workload <path>`  
  Expected: a metrics CSV file is produced for at least one run; contents include the expected columns.

Result processing:

- Run CSV → Parquet conversion on the tiny run.
  Expected: Parquet file(s) created; schema matches what evaluation scripts assume.

Evaluation:

- `python -m eva_scripts.final_leaderboard` on the minimal output directory  
  Expected: produces a leaderboard artifact (exact file names/locations currently **Unspecified**; should be documented explicitly).

## Suggested commit messages and PR template

### Commit message suggestions

- `docs: replace OPARA download placeholders with durable links + verification guidance`
- `docs: add .server_access_credentials.cfg.example and clarify local vs HPC config`
- `fix: restore dataset download/split generation entrypoint and update docs workflow`
- `docs: align experiment config key names with resources/exp_config.yaml`
- `chore: add minimal pytest suite and CI workflow`
- `docs: add missing referenced pages (pipeline/configuration/results format)`
- `meta: unify citation metadata and license fields`

### Pull request description template

**Title:**  
`Docs + reproducibility: make paper workflow executable from scratch`

**Summary**  
Describe the user-facing outcome in 2–4 sentences (e.g., “new users can now reproduce a tiny run locally and can download the OPARA archive without guessing URLs”).

**Changes**  
- Documentation updates:
  - (list files edited)
- Reproducibility improvements:
  - (config template, seed notes, smoke test)
- Tooling:
  - (tests, CI, doc build)

**Verification**  
Paste exact commands run and outcomes:

- `conda create ...`
- `poetry install`
- `pytest -q`
- `mkdocs build -s`
- Minimal experiment run commands + artifacts produced (attach output tree)

**Notes / Follow-ups**  
List any known gaps that remain (e.g., “Kaggle dataset version pinning still Unspecified; needs a future design decision”).

**Checklist**  
- [ ] OPARA download instructions validated against DOI landing page citeturn7view0  
- [ ] Local config template works for minimal run  
- [ ] Docs build passes  
- [ ] Tests pass in CI  
- [ ] Citation metadata verified against attached paper and OPARA entry