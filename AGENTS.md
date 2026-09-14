# AGENTS.md — fisseq-data-pipeline

> `docs/` was rewritten alongside the 1.0.0 release refactor and is current.
> This file plus the source and `pyproject.toml` remain the authoritative
> references; `docs/` is the user-facing version of the same material.

---

## Project overview

The **FISSEQ Data Pipeline** is a Nextflow + Python workflow for processing
single-cell CellProfiler morphological profiling data from FISSEQ (Fluorescence
In-Situ Sequencing) experiments. Each cell carries a genetic variant label; the
pipeline measures how each variant's cell population differs from wildtype (WT)
controls using morphological features.

**End-to-end data flow:**

```
params.yaml (experiments: [...])
      │
      ▼
   INPUT            (per experiment)  ──►  input/<batch_stem>.parquet
      │
      ▼
   QC_FILTER        (per experiment)   ← edit distance, barcode count,
      │                                   variant barcode count
      ▼
   NORMALIZE        (per experiment)   ← z-score fit on WT control cells
      │
      ├──► OVWT_BATCHWISE  (per experiment; params.run_ovwt)
      │         └──► GLOBAL_OVWT  (once per active global channel)
      │
      └──► Feature selection, batchwise (params.run_feature_selection):
             AGGREGATE_FEATURE_TYPE      (per feature type)          ─┐
             GENERATE_SPLIT              (per bootstrap replicate)    │
               └─► AGGREGATE_HALF        (per bootstrap × type × half)│
                     └─► CORRELATE_FEATURES (per bootstrap × type)    │
                           └─► BLOCKLIST  (gathers all bootstraps —   │
                                           the one sync point)        │
                                 └─► COMBINE_BLOCKLISTS (all types) ──┘
                                       └─► FINALIZE_FEATURE_SELECT
                                             └──► GLOBAL_FEATURE_SELECT
                                                  (per active global channel)
```

There is a single pipeline mode — `main.nf` includes one workflow and runs it.

**Main components:**
- `src/fisseq_data_pipeline/` — Python package, one module per pipeline step
- `modules/local/*.nf` — Nextflow process wrappers around the Python CLIs
- `workflows/fisseq.nf` — the DAG
- `main.nf` — entry point
- `params.yaml` — every parameter default
- `nextflow.config` — executor/profile/container settings only
- `Dockerfile` — the single image every process runs in

---

## Setup & environment

- **Python 3.13** (pinned in `.python-version`; `requires-python` is
  `>=3.13,<3.14` — Hydra 1.3.x crashes on 3.14's argparse)
- **uv** for dependency and environment management
- **Nextflow ≥ 26.04** to run the pipeline (not needed for Python-only work)
- **Docker** only to build the image

```bash
uv sync --group dev
uv run pre-commit install
```

---

## Build, run, and test commands

All Python commands run via `uv run` — never bare `python`, `pytest`, `ruff`.

```bash
# Install
uv sync --group dev

# Run the pipeline (-params-file is effectively mandatory)
nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml
nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml -profile local   # no container
nextflow run . --pipeline_dir /path/to/experiment -params-file params.yaml -resume

# Run one stage directly
uv run python -m fisseq_data_pipeline.ovwt output_dir=./out input_file=out/normalized.parquet

# Tests
uv run pytest tests/unit          # fast, no external deps
uv run pytest tests/integration   # slow, needs `nextflow` on PATH
uv run pytest                     # everything

# Lint & format
uv run ruff check .
uv run ruff format --check .
uv run ruff format .

# Nextflow lint (not in CI — run after touching any .nf or nextflow.config)
nextflow lint .

# Docs
uv run mkdocs build --strict
uv run mkdocs serve

# Container
docker build -t fisseq-data-pipeline:dev .
```

---

## Code architecture

### Directory map

```
fisseq-data-pipeline/
├── src/fisseq_data_pipeline/
│   ├── config/
│   │   ├── app.py                 # AppConfig (output_dir, output_root, log_level, random_seed)
│   │   └── input.py               # InputConfig, LabeledInputConfig
│   ├── utils/
│   │   ├── constants.py           # column names, polars selectors, EPS
│   │   ├── xgbparams.py           # XGBoost config, DMatrix builders, split + train helpers
│   │   ├── log.py                 # setup_logging
│   │   ├── batches.py             # load_batches
│   │   ├── variant.py             # classify_variant
│   │   ├── metadata.py            # get_aggregate_meta_data
│   │   ├── splits.py              # row-index / index-file helpers
│   │   ├── featuretypes.py        # join_feature_type_files
│   │   ├── vectors.py             # impact score
│   │   ├── dimreduction.py        # compute_pca, compute_umap
│   │   └── nextflow_staging.py    # reconstruct_staged_paths
│   ├── input.py                   # INPUT
│   ├── qcfilter.py                # QC_FILTER
│   ├── normalize.py               # NORMALIZE (Normalizer class + entry point)
│   ├── aggregate.py               # aggregator library + standalone entry point
│   ├── aggregatefeaturetype.py    # AGGREGATE_FEATURE_TYPE / AGGREGATE_HALF
│   ├── generatesplit.py           # GENERATE_SPLIT
│   ├── correlatefeatures.py       # CORRELATE_FEATURES
│   ├── blocklist.py               # BLOCKLIST
│   ├── combineblocklists.py       # COMBINE_BLOCKLISTS
│   ├── featureselect.py           # FINALIZE_FEATURE_SELECT
│   ├── globalfeatureselect.py     # GLOBAL_FEATURE_SELECT
│   ├── ovwt.py                    # OVWT_BATCHWISE
│   └── globalovwt.py              # GLOBAL_OVWT
├── modules/local/*.nf             # one per process
├── workflows/fisseq.nf
├── main.nf
├── params.yaml
├── nextflow.config
├── Dockerfile
├── tests/{unit,integration}/
└── docs/
```

### Key abstractions

**`config/`** — Hydra structured config hierarchy:
```
AppConfig  (output_dir, output_root, log_level, random_seed)
  └── InputConfig (adds input_file)
        └── LabeledInputConfig (adds label_column)
              └── stage configs (NormalizeConfig, OvwtConfig, ...)
```
Every entry point uses `@hydra.main(...)` with its config registered in the
`ConfigStore`.

**`Normalizer`** (`normalize.py`) — fits per-feature z-score stats on a
LazyFrame and applies them. Stats persist to Parquet (not pickle) and reload via
`Normalizer.load`. Zero-variance features produce `null`. Also reused by
`globalovwt.py` to z-score AUROCs against synonymous variants.

**`BaseAggregator`** (`aggregate.py`) — abstract base for the concrete
aggregation strategies. Combining feature types happens in Nextflow:
`aggregatefeaturetype` runs once per `params.feature_select_types` entry and
`featureselect` joins the per-type outputs on the label column.

**`utils/xgbparams.py`** — shared XGBoost infrastructure: `XGBoostParams` /
`XGBoostConfig`, `get_feature_cols`, `get_dmatrix`, `split_indices_stratified`,
`train_binary_xgboost`. Put XGBoost infrastructure here, not in individual
modules.

---

## Conventions

### Column naming

| Pattern | Meaning |
|---------|---------|
| `meta_*` | Metadata (barcode, batch, labels, QC flags, scores, cell index) |
| `UPPERCASE_WITH_UNDERSCORE` | CellProfiler feature columns |
| `tmp_*` | Ephemeral intermediates, dropped before output |

The `meta_` prefix is load-bearing: `FEATURE_SELECTOR` is
`cs.exclude("^meta_.*$")`, so any new non-`meta_` column is treated as a feature.

Key constants (`utils/constants.py`): `CONTROL_COLUMN_NAME`
(`meta_is_control`), `META_BARCODE_COL`, `META_BATCH_COL`,
`META_CELL_INDEX_COL`, `META_EDIT_DISTANCE_COL`, `META_VARIANT_TAG_COL`,
`IMPACT_SCORE_COL`, `FEATURE_SELECTOR`, `META_SELECTOR`.

### DataFrame conventions

- Always use `pl.LazyFrame`; `.collect()` only at I/O boundaries.
- Convert NaN → null with `.fill_nan(None)` before statistical operations.
- Null-containing rows/columns are excluded from aggregations intentionally.

### Configuration pattern

Each module defines a `@dataclasses.dataclass` config, registers it with
`ConfigStore`, and uses
`@hydra.main(version_base=None, config_path=None, config_name="<name>_main")`.
Overrides are `key=value` CLI pairs (dot-notation for nested fields:
`xgboost.params.max_depth=5`).

### Logging

Every `main()` calls `setup_logging(cfg, name)` from `utils/log.py` first.

### Error handling

- `ValueError` for bad inputs or configuration.
- Avoid bare `except` — the sanctioned exception is per-variant failure
  isolation in `ovwt.py:ovwt_batchwise`, which is load-bearing (see gotcha 4).

### Docstring style

NumPy style, enforced by mkdocstrings under `--strict`. Note that prose placed
inside a `Parameters` or `Attributes` block is parsed as a malformed parameter
entry and **fails the docs build** — put narrative text in a `Notes` section
instead.

### Commit style

Lowercase verb, optional scope, PR number in parentheses:
`fix NaN handling, performance improvements (#10)`.

---

## Gotchas & known issues

1. **There is exactly one seed.** `AppConfig.random_seed`. Never add a
   stage-local `random_state`/`seed` field — `tests/unit/test_config.py` sweeps
   every config class and fails if one appears. A stage that must differ from
   its siblings derives a fixed offset (`GENERATE_SPLIT` uses
   `random_seed + bootstrap_idx`; `AGGREGATE_HALF` uses
   `random_seed + bootstrap_idx * 2 + half_num`).

2. **Row order is part of reproducibility.** Polars inner joins are not
   order-preserving under multithreaded execution. `QC_FILTER` assigns
   `meta_cell_index` over the raw input order and sorts on
   `(meta_cell_index, meta_variant_tag)` before publishing. Without that, the
   same input yields the same rows in a different order every run and every
   seeded step downstream silently diverges. If you add a stage that reorders
   published cell-level output, restore a deterministic order before writing.

3. **Never put a `params { }` block in `nextflow.config`**, not even an empty
   one. Combined with `-params-file`, Nextflow's `ConfigBuilder` before 26.04.6
   misattributes every `params.yaml` key into `process{}` scope and raises
   `Unknown config attribute 'params'`. All defaults live in `params.yaml`.

4. **`ovwt.py`'s per-variant `try/except` is load-bearing.** A
   `(barcode, is_wt)` stratum can be too small for the inner nested split even
   after `_stratification_key`'s rare-bucket collapse. Small variants
   legitimately hit this; the alternative is losing every other variant's
   results. Tests assert a rare-barcode variant is silently dropped while its
   peers succeed.

5. **Test fixtures for OvWT must be big enough.** The inner
   `split_indices_stratified` runs inside each outer fold, so a stratum needs
   roughly 8–13 members to survive both levels — merely `>= n_folds` is not
   enough. An undersized fixture sends every variant down the `except` branch,
   and the test then passes against empty output while asserting nothing.

6. **OvWT trains on WT-normalized features, not synonymous-normalized ones.**
   The sibling `fisseq-embeddings-pipeline` (which this implementation was
   ported from) z-scores against synonymous variants before training. Here that
   re-centering happens downstream on the AUROCs, in `globalovwt.py`. This is
   deliberate — do not "fix" it by adding a second normalizer fit in `ovwt.py`.

7. **Two different control baselines.** `normalize.py` uses **WT** cells (SQL
   `control_sample_query`). `aggregate.py:variant_classification` flags
   **synonymous** variants as `meta_is_control`, which is what
   `featureselect.py`, `globalfeatureselect.py` and `globalovwt.py` use. Not
   interchangeable.

8. **`GLOBAL_OVWT` needs ≥2 synonymous variants per experiment.** Its
   per-experiment normalizer fits on synonymous rows; a single one makes std
   (ddof=1) undefined and nulls every score.

9. **Never bind a Nextflow variable named `channel`.** It is a reserved
   lowercase alias for the `Channel` class and silently resolves to
   `nextflow.Channel` instead of failing. Use `chan`. (The
   `channel.fromList(...)` factory call is fine.)

10. **DSL2 forbids bare statements at script scope.** A top-level helper must be
    `def someFunction() { ... }`, not `def x = { ... }`.

11. **`.join()` is not a broadcast operator.** For a many-to-one key
    relationship it silently keeps one match per key and drops the rest. Use
    `.combine(other, by: 0)` to broadcast; reserve `.join()` for cases where
    both sides are already one-per-key.

12. **Pass real path channels to collecting processes, not glob strings.** A
    `val` glob hashes only the glob text, so `-resume` fails to invalidate when
    the underlying files change.

13. **`stageAs` with one file has no index.** Nextflow substitutes the `*` with
    an empty string for a single staged file (`res_input_.parquet`) and
    1-indexes only from two up. `utils/nextflow_staging.py` encodes this.

14. **`output_root` takes priority over `output_dir`** in `aggregate.py` and
    `featureselect.py` — the output lands at `{output_root}.{stem}.parquet`
    regardless of `output_dir`. Several `.nf` scripts therefore use
    `output_dir=.` plus a glob `mv`. If you change output naming there, update
    those.

15. **`errorStrategy 'ignore'` is on every process.** A failed stage drops that
    experiment rather than aborting the run, so a missing output may mean its
    task failed. Check the run log.

16. **`pandas` is a runtime dep but barely used directly.** The codebase uses
    Polars; pandas comes in via `pycytominer`.

17. **`.python-version` must be copied into the image before the first
    `uv sync`.** Without it uv resolves the newest `>=3.13` interpreter, and
    Hydra 1.3.x crashes on Python 3.14's argparse, breaking every stage's CLI.

---

## PR / commit workflow

```bash
git checkout main && git pull
git checkout -b <descriptive-branch-name>
# ... make changes, commit ...
git push -u origin <descriptive-branch-name>
```

Claude pushes the branch but does **not** open the PR — the user does that.

Before merging: `uv run pytest tests/unit`, `uv run ruff check .`,
`uv run ruff format --check .`, `nextflow lint .`, and
`uv run mkdocs build --strict`.

CI (`.github/workflows/`):
- `pr-checks.yml` — `unit-tests`, `integration-tests`, `lint` on every PR to
  `main` (skipped for drafts)
- `docker.yml` — builds the image on every PR; pushes `:latest` + `:<short-sha>`
  on push to `main`, and `:<version>` on a `v*` tag
- `docs.yml` — deploys MkDocs to GitHub Pages on push to `main`

Tag `v<version>` to publish a versioned image.

---

## Documentation maintenance

- Any change to CLI flags / Hydra config fields / Nextflow processes / module
  responsibilities **must** update the relevant `docs/` page in the same change.
  Start from `docs/architecture.md`, `docs/nextflow.md`,
  `docs/configuration.md`, and the relevant `docs/cli/<module>.md` +
  `docs/api/<module>.md` pair.
- Any new source file needs a file-level docstring (Python) or top `//` comment
  block (`.nf`), except `__init__.py`.
- Adding or removing a module means updating `mkdocs.yml`'s nav — `docs.yml`
  runs `mkdocs build --strict`, so a dangling reference fails the build.
- `README.md` stays a thin pointer (overview + quick start + docs link).

---

## Safety / do-not-touch list

| Path | Reason |
|------|--------|
| `site/` | Generated MkDocs output, gitignored. CI publishes it to `gh-pages`. |
| `uv.lock` | Auto-managed by uv. Edit `pyproject.toml`, then `uv sync`. |
| `.venv/` | Managed by uv. |
| `<pipeline_dir>/work/` | Nextflow task working directories. Delete only via `nextflow clean`. |
| Any `*.parquet` under `<pipeline_dir>/` | Pipeline output data. |
| `.github/workflows/docs.yml` | Deploys live documentation. |
