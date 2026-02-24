# Determinism and Seeds

## What you need to know

- OGAL has **two seed knobs**: a *global seed* (`RANDOM_SEED`, default `1312`) set once at process start, and a *per-experiment seed* (`EXP_RANDOM_SEED`) drawn from `EXP_GRID_RANDOM_SEED` in the experiment YAML.
- Before each AL loop, both NumPy and Python `random` are re-seeded with `EXP_RANDOM_SEED`, so experiments with the same seed and the same pre-generated splits reproduce on a single thread.
- **Determinism across third-party frameworks is imperfect.** Each framework applies the seed differently, and one (scikit-activeml) currently uses the global seed instead of the per-experiment seed.
- Additional non-determinism can come from parallel `RandomForestClassifier` training, platform-specific BLAS behavior (MLP/Adam), and regenerating dataset splits without a controlled seed.
- For full reproducibility, use the **OPARA-archived datasets and splits** rather than regenerating them.

---

## The two seeds

| Seed | Config key | Default | Scope |
|------|-----------|---------|-------|
| Global seed | `RANDOM_SEED` | `1312` | Set once at process start via `np.random.seed()` and `random.seed()`. Pass `--RANDOM_SEED -1` to disable. |
| Experiment seed | `EXP_RANDOM_SEED` | from `EXP_GRID_RANDOM_SEED` in YAML | Re-seeded before each experiment's AL loop. In the paper run (`full_exp_jan`), this is `0`. |

## Known caveats

| Source | Impact |
|--------|--------|
| **RF parallelism** | `RandomForestClassifier` uses `n_jobs=cpu_count()`; parallel tree building is non-deterministic across hardware. |
| **scikit-activeml seed mismatch** | Receives `RANDOM_SEED` (global) instead of `EXP_RANDOM_SEED`; sweeping `EXP_GRID_RANDOM_SEED` does not change its seed. |
| **MLP / Adam solver** | May vary across platforms and BLAS versions even with a fixed seed. |
| **Dataset split regeneration** | `00_download_datasets.py` uses the global NumPy state; re-running on a fresh machine produces different splits unless the same global seed is active. Use the OPARA archive for the paper's exact splits. |

---

**See also:** [Configuration](configuration.md) (seed config keys) · [Paper Run Definition](paper_subset.md) · [Splits & start sets](splits_and_start_sets.md)
