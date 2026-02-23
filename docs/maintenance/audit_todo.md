# Maintenance Audit TODO

Actionable fixes derived from `deep-research-report.md`, grouped by priority.
Each item references the exact files that must be changed.

---

## Critical

- [ ] **Replace OPARA download placeholders with exact commands and add artifact verification
  guidance.**
  Add a "Data download" block to each file below that uses the OPARA bitstream URLs (DOI
  `10.25532/OPARA-862`), recommends `wget -c`/`aria2c` for resumable downloads, and instructs
  users to verify downloaded contents against `archive_listing.txt`.
  - `README.md`
  - `docs/personas/reproduce_and_run.md`
  - `docs/personas/analyze_dataset.md`

- [x] **Document the config file as mandatory and provide a committed example template.**
  Option B chosen: `OGAL_OUTPUT` env-var claims removed from all docs; `.server_access_credentials.cfg`
  is the sole configuration mechanism.  A committed template (`.server_access_credentials.cfg.example`)
  is now provided and all setup instructions reference it.
  Files changed:
  - `docs/personas/reproduce_and_run.md`
  - `README.md`
  - *(created)* `.server_access_credentials.cfg.example`

- [ ] **Restore or replace the "download datasets" entrypoint so the documented pipeline is
  actually executable.**
  `00_download_datasets.py` has large sections commented out and key loops that short-circuit
  with `continue`, making it inert.  Either:
  - **(A – preferred)** Refactor into three explicit scripts (`00a_download_openml.py`,
    `00b_download_kaggle.py`, `00c_generate_splits.py`), each with `--help` and a `--datasets`
    filter.
  - **(B – fastest)** Un-comment/repair the existing download logic in `00_download_datasets.py`
    so it produces the dataset/split artifacts that downstream scripts expect.
  In both cases update the docs to reflect the repaired workflow, add Kaggle API prerequisites,
  and add a "single-dataset smoke run" section.
  Files to change:
  - `00_download_datasets.py` *(and/or new `00a_…`, `00b_…`, `00c_…` scripts)*
  - `docs/personas/reproduce_and_run.md`

---

## High

- [x] **Fix config-key naming mismatch in docs.**
  Replaced `EXP_GRID_NR_QUERIES` (incorrect) with `EXP_GRID_NUM_QUERIES` (actual key used in
  `resources/exp_config.yaml` and declared in `misc/config.py`).  Added missing
  `EXP_GRID_START_POINT` row to the table and example.  Added a "Where config keys come from"
  subsection pointing to both source files.
  - `docs/personas/extend_benchmark.md`

- [x] **Add a canonical "paper reproduction config" section.**
  Added "Reproducing the paper run: `full_exp_jan`" in both `README.md` and
  `docs/personas/reproduce_and_run.md`.  Includes: config name, compute-scale disclaimer,
  exact pipeline commands (verified against repo scripts), expected output trees for the
  `test` (minimal) and `full_exp_jan` (full) runs, and a minimal local subset path.
  - `README.md`
  - `docs/personas/reproduce_and_run.md`

- [x] **Clarify and harden seed handling and determinism claims.**
  Added a "Seed handling and determinism" section to `docs/personas/reproduce_and_run.md`
  documenting: the two-seed design (`RANDOM_SEED` / `EXP_RANDOM_SEED`), where each originates,
  per-framework seed coverage (including the `RANDOM_SEED` vs `EXP_RANDOM_SEED` inconsistency
  in `skactiveml_runner.py`), dataset split/start-set regeneration caveat, and all known
  sources of non-determinism (RF parallelism, scikit-activeml mismatch, MLP Adam, etc.).
  No code changes.
  - `docs/personas/reproduce_and_run.md`

- [x] **Unify project citation metadata across all surfaces.**
  Aligned `docs/personas/research_ideas.md` with the canonical `README.md` citation:
  changed `@article` → `@misc`, restored full title and full author list, replaced
  `journal=` with `eprint`/`archivePrefix`/`primaryClass` fields, and added the
  missing `@dataset` entry.  Added dataset `references` block to `CITATION.cff`
  with the OPARA DOI.  `README.md` was already correct and required no changes.
  - `docs/personas/research_ideas.md`
  - `CITATION.cff`

- [ ] **Fix license metadata mismatch in packaging.**
  The license file and README both state AGPL-3.0, but `pyproject.toml` uses a different (or
  missing) license string.  Align the field and add the SPDX identifier `AGPL-3.0-only` (or
  `AGPL-3.0-or-later` as appropriate).
  - `pyproject.toml`
  - `README.md`

- [ ] **Resolve "5 vs 6 frameworks" and "50+ vs 76 strategies" inconsistency.**
  Decide on the canonical numbers that represent the paper's scope, and update every occurrence
  in the docs to use them.  Optionally add a note distinguishing "paper subset" from "full
  benchmark support".
  - `README.md`
  - `docs/index.md`
  - `docs/personas/extend_benchmark.md`

- [ ] **Add compute-budget context to the quickstart / README.**
  The paper reports ~3.6 M CPU hours; the README currently omits this.  Add a short note so
  that users understand the scale before attempting to reproduce.
  - `README.md`

- [ ] **Fix runtime-limit wording: timeout covers the full AL cycle, not just query selection.**
  Code uses a 300-second timeout for the whole cycle; docs say it limits only query selection.
  Correct the description everywhere it appears.
  - `docs/personas/reproduce_and_run.md`
  - `docs/personas/extend_benchmark.md`

---

## Medium

- [ ] **Add a minimal test suite and CI workflow.**
  Implement the tests described in the report and wire them into GitHub Actions:
  - *(create)* `tests/test_config_smoke.py` — verifies `Config()` can be instantiated from a
    temporary config file and resolves output paths correctly.
  - *(create)* `tests/test_dataset_enum.py` — verifies dataset enum extension order is stable
    given a fixed YAML ordering.
  - *(create)* `.github/workflows/ci.yml` — runs `ruff check .` and `pytest -q`.
  - `docs/contributing.md` — update to reflect the new test/CI setup.

- [x] **Add missing documentation pages that docstrings already reference.**
  Created all three missing pages and added them to the `mkdocs.yml` nav under a
  "Reference" group.  Each page is cross-linked from the others and from
  `reproduce_and_run.md`.
  - *(created)* `docs/pipeline.md` — end-to-end workflow with ASCII diagram and step-by-step commands.
  - *(created)* `docs/configuration.md` — full config reference with required vs optional keys and defaults.
  - *(created)* `docs/results_format.md` — formal results schema with column definitions and Python reading examples.
  - `mkdocs.yml`

- [ ] **Document dataset preprocessing precisely and tie it to paper statistics.**
  Add a "Preprocessing contract" section covering missing-value handling, feature encoding,
  scaling, and class filtering.  Explain how these choices affect reproducibility, and reference
  the paper's dataset statistics table.
  - `docs/personas/analyze_dataset.md`
  - *(create)* `docs/datasets.md`

- [ ] **Document dataset versioning and split / start-set generation completely.**
  The paper describes 5 train-test splits × 20 start sets = 100 repeats per dataset.  The
  code generates these, but the docs do not state the exact algorithm, seed values, or the
  persistence format clearly enough to reproduce without reading the source.
  - `docs/personas/reproduce_and_run.md`
  - `docs/personas/analyze_dataset.md`

- [ ] **Clarify the conda-lock vs Poetry dependency relationship.**
  The docs do not explain why both tools are used together, which one pins git-based
  dependencies, or what the risk of "moving targets" is for git deps.
  - `docs/personas/reproduce_and_run.md`
  - `README.md`

- [ ] **Document Kaggle API prerequisites.**
  The setup instructions do not mention where to place `kaggle.json` or how to set the
  required environment variables.  Add a section covering this and explicitly warn against
  committing credentials.
  - `docs/personas/reproduce_and_run.md`
  - `README.md`

- [ ] **Identify the "paper strategy set" by name in the docs.**
  Docs currently describe many more strategies than the 28 ultimately used in the paper.  Add
  an explicit section or table listing the 28 strategies by name (matching the paper's
  exclusion rationale) so users can reproduce the exact paper run without guessing.
  - `docs/personas/reproduce_and_run.md`
  - `docs/personas/extend_benchmark.md`

- [ ] **Add minimal expected-output examples ("golden" artifacts) to docs.**
  Provide a concrete example output tree (directory layout and key file names) for a minimal
  single-dataset run, and an example Parquet schema, so users can verify correctness.
  - `docs/results_format.md` *(new page; see "missing pages" item above)*
  - `docs/personas/reproduce_and_run.md`

---

## Low

- [ ] **Establish project versioning, a changelog, and release guidance.**
  There are no stable version tags or changelog.  Introduce a minimal semantic versioning
  policy, create an initial changelog, and document how to create a release (and how to cite a
  specific version).
  - *(create)* `CHANGELOG.md`
  - *(create or update)* `docs/contributing.md` — add a "Release process" section.
  - *(optional)* *(create)* `.github/workflows/release.yml` — GitHub Actions release workflow.

- [ ] **Align the Contributing guide with the actual repo state.**
  The contributing page references tests and CI that do not yet exist (or will be added by the
  Medium items above).  After those items are resolved, update the guide to reflect reality.
  - `docs/contributing.md`

- [ ] **Add a "Cite this version" section to the README.**
  Once DOI/arXiv metadata is unified (see High items) and versioning is in place, add a short
  block so users can cite a specific repository snapshot.
  - `README.md`
