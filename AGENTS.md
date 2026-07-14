# AGENTS.md — fisseq-data-pipeline

> **WARNING: `docs/` is stale and must not be trusted.**
> The Markdown files under `docs/` and the generated `site/` directory were written at an earlier stage of the project and have not been kept in sync with the code. Do not use them to understand current behavior, parameter names, or architecture. This file (AGENTS.md) plus the actual source code and `pyproject.toml` are the authoritative references.

---

## Project overview

The **FISSEQ Data Pipeline** is a Nextflow + Python workflow for processing single-cell CellProfiler morphological profiling data from FISSEQ (Fluorescence In-Situ Sequencing) experiments. Each cell carries a genetic variant label; the pipeline measures how each variant's cell population differs from wildtype (WT) controls using morphological features.

**End-to-end data flow:**

```
input/*.parquet  (one file per batch, CellProfiler morphological features + barcode annotations)
      │
      ▼
QC_FILTER        (per batch)   ← edit distance, barcode count, variant barcode count
      │
      ├──► BATCHVSBATCH (pre)       (global — waits for all QC_FILTER)
      ▼
NORMALIZE        (per batch)   ← z-score fit on WT control cells
      │
      ├──► BATCHVSBATCH (post)      (global — waits for all batches)
      ├──► OVWT_BATCHWISE           (per batch)
      ├──► OVWT_GLOBAL              (global — waits for all batches; skipped if params.global = false)
      └──► Feature selection (batchwise always runs; global waits for all batches, skipped if params.global = false):
             AGGREGATE_FEATURE_TYPE      (per feature type)          ─┐
             GENERATE_SPLIT              (per bootstrap replicate)    │
               └─► AGGREGATE_HALF        (per bootstrap × feature type × half)
                     └─► CORRELATE_FEATURES (per bootstrap × feature type)
                           └─► BLOCKLIST  (gathers all bootstrap replicates per feature type — the one sync point)
                                 └─► COMBINE_BLOCKLISTS (gathers all feature types) ┘
                                       └─► FINALIZE_FEATURE_SELECT (joins AGGREGATE_FEATURE_TYPE outputs + combined blocklist)
```

**Main components:**
- `src/fisseq_data_pipeline/` — Python package with one module per pipeline step
- `modules/local/*.nf` — Nextflow process wrappers that call the Python CLI tools
- `workflows/fisseq.nf` — Nextflow DAG that wires processes together
- `main.nf` — Nextflow entrypoint (parameter definitions + input validation)

---

## Setup & environment

### Requirements

- **Python 3.13** (pinned in `.python-version`; managed by pyenv or similar)
- **uv** for Python dependency and environment management
- **Nextflow ≥ 23.10** (only needed to run the full pipeline, not for Python-only work)
- No required environment variables

### Install

```bash
# Install all runtime + dev dependencies into .venv
uv sync --group dev

# Install pre-commit hooks (one-time)
uv run pre-commit install
```

The package installs in editable mode, so `fisseq-*` CLI commands are immediately available via `uv run fisseq-qc-filter`, etc.

### Cluster / HPC

The repo does not ship a `nextflow.config`. Executor and environment setup are entirely user-provided via a config file passed with `-c`. Copy `example.config` and fill in one of the `beforeScript` options (venv activation, package install) and, if running on a cluster, uncomment and adapt the `sge` profile block.

---

## Build, run, and test commands

All Python commands must be run via `uv run` — never bare `python`, `pytest`, `ruff`, etc.

### Install

```bash
uv sync --group dev
```

### Run the full Nextflow pipeline

```bash
# Local
nextflow run main.nf --input_dir /path/to/experiment

# SGE cluster (supply your own env config)
nextflow run main.nf -c your.config -profile sge --input_dir /path/to/experiment

# Resume after interruption (Nextflow task caching)
nextflow run main.nf --input_dir /path/to/experiment -resume
```

### Run individual Python CLI tools

```bash
uv run fisseq-qc-filter \
    output_dir=./out \
    'cell_files=[data/plate1.parquet]' \
    bc_threshold=10

uv run fisseq-normalize \
    output_dir=./out \
    input_file=out/filtered_cells.parquet

uv run fisseq-aggregate \
    output_dir=./out \
    input_file=out/normalized.parquet \
    aggregator=mean

uv run fisseq-aggregate-feature-type \
    output_dir=./out \
    input_file=out/normalized.parquet \
    aggregator=mean

uv run fisseq-generate-split \
    output_dir=./out \
    input_file=out/normalized.parquet \
    random_state=1

uv run fisseq-correlate-features \
    output_dir=./out \
    half1_file=out/half1.mean.parquet \
    half2_file=out/half2.mean.parquet

uv run fisseq-blocklist \
    output_dir=./out \
    'correlation_files=out/correlations/mean/*.parquet' \
    minimum_correlation=0.5

uv run fisseq-combine-blocklists \
    output_dir=./out \
    'blocklist_files=out/blocklists/*.parquet'

uv run fisseq-feature-select \
    output_dir=./out \
    input_file=out/normalized.parquet \
    'feature_type_files=out/aggregates/*.parquet' \
    block_list_file=out/blocklist.parquet

uv run fisseq-permanova \
    output_dir=./out \
    'input_file=data/batches/*.parquet' \
    n_permutations=999

uv run fisseq-ovwt \
    output_dir=./out \
    input_file=out/features.parquet \
    min_cells=250

uv run fisseq-ovwt-cell-scores \
    output_dir=./out \
    input_file=out/normalized.parquet \
    models_path=out/models.pkl
```

```bash
uv run fisseq-batch-vs-batch \
    output_dir=./out \
    input_file=out/features.parquet \
    batch_column=meta_batch
```

### Tests

```bash
# Fast: unit tests only (~seconds, no external dependencies)
uv run pytest tests/unit

# All tests including integration (slow — requires Nextflow installed)
uv run pytest

# Single module
uv run pytest tests/unit/test_aggregate.py -v
```

The integration test (`tests/integration/test_integration.py`) is session-scoped and runs the full Nextflow pipeline on synthetic data. It can take several minutes and requires `nextflow` on `PATH`.

### Lint & format

```bash
# Lint (ruff)
uv run ruff check .

# Format check
uv run black --check .

# Format apply
uv run black .

# Import sort (not hooked into pre-commit — run manually)
uv run isort src/ tests/
```

> **Note:** The pre-commit hook (`.pre-commit-config.yaml`) only runs `black`. `ruff` and `isort` are in dev dependencies but are **not** run automatically on commit. Run them manually before pushing.

### Docs

```bash
# Serve locally
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

---

## Code architecture

### Directory map

```
fisseq-data-pipeline/
├── src/fisseq_data_pipeline/      # Python package (src layout)
│   ├── config/
│   │   ├── app.py                 # AppConfig dataclass (base: output_dir, output_root, log_level)
│   │   └── input.py               # InputConfig, LabeledInputConfig (inherit AppConfig)
│   ├── utils/                     # Shared, non-CLI internals
│   │   ├── constants.py           # Column names, Polars selectors, EPS
│   │   ├── xgbparams.py           # Shared XGBoost infrastructure (dataclasses, DMatrix builders, split helper)
│   │   ├── log.py                 # setup_logging
│   │   ├── batches.py             # load_batches
│   │   ├── variant.py             # classify_variant
│   │   ├── metadata.py            # get_column, get_aggregate_meta_data
│   │   └── vectors.py             # compute_norm, compute_query_dot, compute_cosine_distance, compute_impact_score
│   ├── qcfilter.py                # QC filtering entry point
│   ├── normalize.py               # Normalizer class + normalize entry point
│   ├── aggregate.py               # 8 aggregation strategies + full-aggregation and per-feature-type entry points
│   ├── features.py                # Bootstrap split/correlate/blocklist stages + final pycytominer selection
│   ├── ovwt.py                    # XGBoost one-vs-WT training + entry point
│   ├── ovwtcellscores.py          # Cell scoring via trained models
│   ├── batchvsbatch.py            # Per-variant multiclass batch classifier; OvR AUC + Mann-Whitney p-value (no pyproject entry)
│   └── permanova.py               # Per-variant pairwise PERMANOVA entry point
├── modules/local/
│   ├── qc_filter.nf
│   ├── normalize.nf
│   ├── permanova.nf
│   ├── batchvsbatch.nf
│   ├── ovwt_batchwise.nf
│   ├── ovwt_global.nf
│   ├── aggregate_feature_type.nf
│   ├── generate_split.nf
│   ├── aggregate_half.nf
│   ├── correlate_features.nf
│   ├── blocklist.nf
│   ├── combine_blocklists.nf
│   └── finalize_feature_select.nf
├── workflows/
│   └── fisseq.nf                  # Main Nextflow workflow DAG
├── main.nf                        # Nextflow entrypoint + parameter defaults
├── example.config                 # Template user config (env setup, venv/conda/singularity)
├── tests/
│   ├── unit/                      # 12 files, fast, synthetic data
│   └── integration/               # 1 file, slow, full pipeline run
├── docs/                          # STALE — do not rely on
├── site/                          # Generated MkDocs output — do not edit
├── pyproject.toml                 # Package metadata, deps, scripts, tool config
├── .python-version                # 3.13
├── .pre-commit-config.yaml        # black only
└── mkdocs.yml                     # Docs config (Read the Docs theme, mkdocstrings NumPy style)
```

### Key abstractions

**`src/fisseq_data_pipeline/config/`** — Hydra structured config hierarchy:
```
AppConfig
  └── InputConfig (adds input_file)
        └── LabeledInputConfig (adds label_column, default "meta_aa_changes")
              └── tool-specific configs (e.g. NormalizeConfig, AggregateConfig, OvwtConfig)
```
Every entry point uses `@hydra.main(...)` with its config class registered in the `ConfigStore`.

**`Normalizer`** (`normalize.py`) — fits per-feature z-score stats (mean, std) on a LazyFrame and applies them. Stats are persisted to Parquet (not pickle) and reloaded with `Normalizer.load(path)`. Zero-variance features produce `null` after normalization.

**`BaseAggregator`** (`aggregate.py`) — abstract base for 8 concrete aggregation strategies (mean, median, MAD, std, EMD, KS, QQ, AUROC). There is no multi-aggregator wrapper; combining feature types happens in Nextflow — `aggregate.feature_type_main` runs once per `params.feature_types` entry, and `features.main` (the final feature-selection stage) joins the per-feature-type outputs on the label column.

**`utils/xgbparams.py`** — shared XGBoost infrastructure imported by `ovwt.py`, `ovwtcellscores.py`, and `batchvsbatch.py`. Contains: `XGBoostParams` and `XGBoostConfig` dataclasses; `get_feature_cols` (CellProfiler column detection); `get_dmatrix` (binary DMatrix builder); `get_dmatrix_multiclass` (multiclass DMatrix with sorted integer encoding); `split_indices_stratified` (80/10/10 stratified split on any label array). Do not add XGBoost-specific infrastructure to individual modules — put it here.

**`load_batches`** (`utils/batches.py`) — accepts a path or glob pattern, reads matching Parquet files, tags each with `meta_batch` = filename stem, returns a concatenated `pl.LazyFrame` plus an output stem string.

**Nextflow synchronization pattern** (`workflows/fisseq.nf`): global processes (BATCHVSBATCH, OVWT_GLOBAL, the feature-selection processes, PERMANOVA) wait for all per-batch outputs to complete by collecting all batch stems into a single signal channel carrying the absolute `input_dir` path. `BATCHVSBATCH` and `PERMANOVA` are each a single parameterized process (`modules/local/batchvsbatch.nf`, `modules/local/permanova.nf`) invoked twice via Nextflow's `include { X as Y }` aliasing (a process cannot be called twice under its own name in one workflow) — `BATCHVSBATCH_PRE` waits on `qc_signal` (all QC_FILTER done) and globs `qc_filter/*/filtered_cells.parquet` with `use_parent_name=true`; `BATCHVSBATCH_POST` waits on `global_signal` (all NORMALIZE done) and globs `normalization/cells/*.parquet`. Likewise `PERMANOVA_NORMALIZED` waits on `global_signal`/`normalization/cells/*.parquet` and `PERMANOVA_BATCH_CORRECTED` waits on `bc_signal`/`batch_correction/cells/*.parquet`. All varying bits (glob path, `use_parent_name`, `publishDir` subpath) are passed in as process input values, not hardcoded per-call.

**Feature-selection pipeline** (`workflows/fisseq.nf`) follows the same aliasing pattern, applied to 7 processes (`AGGREGATE_FEATURE_TYPE`, `GENERATE_SPLIT`, `AGGREGATE_HALF`, `CORRELATE_FEATURES`, `BLOCKLIST`, `COMBINE_BLOCKLISTS`, `FINALIZE_FEATURE_SELECT`), each invoked once as `*_BATCHWISE` and once as `*_GLOBAL`. Channels are crossed via `.combine()` over `feature_types_ch` (`params.feature_types`) and `bootstrap_ch` (`1..params.bootstrap`), split into per-half tuples via `.flatMap()`, and re-paired via `.groupTuple()`. `BLOCKLIST`'s `groupTuple(by: [batch_key, feature_type])` — gathering all `params.bootstrap` correlation replicates for one feature type before computing a median-`r` threshold — is the pipeline's only cross-bootstrap synchronization point; everything else in the split/aggregate/correlate chain is fully parallel across bootstrap × feature type (× half). `params.global` (default `true`) gates `OVWT_GLOBAL` and the entire `*_GLOBAL` feature-selection branch — set it to `false` to skip both and only run the batchwise processes (e.g. for datasets where the global bootstrap × feature-type cross product across all batches combined is prohibitively expensive). The batchwise branch, `BATCHVSBATCH`, `PERMANOVA`, and the batch-correction branch are unaffected by this flag.

### CLI entry points (registered in `pyproject.toml`)

| Command | Module | Purpose |
|---------|--------|---------|
| `fisseq-qc-filter` | `qcfilter:main` | Edit distance + barcode QC |
| `fisseq-normalize` | `normalize:main` | Z-score normalization |
| `fisseq-aggregate` | `aggregate:main` | Standalone per-variant aggregation + normalizer + metadata (not wired into Nextflow) |
| `fisseq-aggregate-feature-type` | `aggregate:feature_type_main` | Lean per-feature-type aggregation, optionally filtered to an index-file row subset |
| `fisseq-generate-split` | `features:generate_split_main` | Generate one stratified 50/50 pseudo-replicate split |
| `fisseq-correlate-features` | `features:correlate_features_main` | Per-feature Pearson correlation between two aggregate halves |
| `fisseq-blocklist` | `features:blocklist_main` | Median-`r`-across-bootstraps blocklist for one feature type |
| `fisseq-combine-blocklists` | `features:combine_blocklists_main` | Concatenate per-feature-type blocklists |
| `fisseq-feature-select` | `features:main` | Final stage: joins per-feature-type aggregates, applies combined blocklist, pycytominer selection |
| `fisseq-ovwt` | `ovwt:main` | One-vs-WT XGBoost training |
| `fisseq-ovwt-cell-scores` | `ovwtcellscores:main` | Score cells against trained OvWT models |
| `fisseq-permanova` | `permanova:main` | Per-variant PERMANOVA (cosine distance) |
| `fisseq-batch-vs-batch` | `batchvsbatch:main` | Per-variant multiclass batch classifier (OvR AUC + Mann-Whitney p per batch) |

All share base Hydra fields: `output_dir` (required), `output_root` (optional prefix), `log_level` (default `"info"`).

### Output layout

All outputs land under `<input_dir>` alongside the `input/` folder:

```
<input_dir>/
  qc_filter/<batch>/
    filtered_cells.parquet
    barcode_counts.parquet
    variants_per_barcode.parquet
  normalization/
    cells/<batch>.parquet
    normalizers/<batch>.normalizer.parquet
  batchvsbatch/
    pre/results.parquet         # pre batch correction (QC-filtered cells); columns: variant, batch, auroc, mw_pvalue, n_batch_cells, n_cells
    post/results.parquet        # post batch correction (normalized cells)
  ovwt_batchwise/<batch>/
    results.parquet
    models.pkl
  ovwt_global/
    results.parquet
    models.pkl
  feature_select_batchwise/<batch>/
    aggregates/<feature_type>.parquet                                     # stage 1
    splits/bootstrap_<n>/half{1,2}.parquet                                # stage 2a
    half_aggregates/bootstrap_<n>/<feature_type>/half{1,2}.parquet        # stage 2b
    correlations/<feature_type>/bootstrap_<n>.parquet                    # stage 2c
    blocklists/<feature_type>.parquet                                    # stage 2d
    blocklist.parquet                                                    # stage 3 (combined)
    output.parquet                                                       # stage 4 (final)
  feature_select_global/
    (same layout, no <batch> nesting)
```

---

## Conventions

### Column naming

| Pattern | Meaning |
|---------|---------|
| `meta_*` | Metadata columns (barcode, batch, labels, QC flags, scores) |
| `UPPERCASE_WITH_UNDERSCORE` | CellProfiler morphological feature columns |
| `tmp_*` | Ephemeral intermediate columns, dropped before output |

Key constants (from `utils/constants.py`):

| Constant | Value | Purpose |
|----------|-------|---------|
| `CONTROL_COLUMN_NAME` | `"meta_is_control"` | Boolean flag for control/WT rows |
| `META_BARCODE_COL` | `"meta_barcode"` | Barcode identifier |
| `META_BATCH_COL` | `"meta_batch"` | Batch identifier (set from filename stem) |
| `META_EDIT_DISTANCE_COL` | `"meta_edit_distance"` | QC metric |
| `IMPACT_SCORE_COL` | `"meta_impact_score"` | Cosine-distance impact score vs WT |
| `FEATURE_SELECTOR` | `cs.exclude(cs.starts_with("meta_"))` | Polars selector for feature columns |
| `META_SELECTOR` | `cs.starts_with("meta_")` | Polars selector for metadata columns |

### DataFrame conventions

- **Always use `pl.LazyFrame`** for processing; call `.collect()` only at I/O boundaries or when an operation requires materialization.
- **NaN → null**: convert with `.fill_nan(None)` before any statistical operations. Both directions (pre- and post-computation) are standard across the codebase.
- **Null exclusion**: null-containing rows/columns are excluded from aggregations — this is intentional and preserves feature columns that have patchy data.

### Configuration pattern

Every module defines its Hydra config class as a `@dataclasses.dataclass`, registers it with `ConfigStore`, and uses `@hydra.main(version_base=None, config_path=None, config_name="<name>_main")`. Overrides are passed on the CLI as `key=value` pairs (Hydra dot-notation for nested fields: `xgboost.params.max_depth=5`).

### Logging

All modules call `setup_logging(cfg, name)` from `utils/log.py` at the start of `main()`. This writes logs to both stdout and a file at `{output_dir}/{output_root}.{name}.log` (or `{output_dir}/{name}.log` when `output_root` is unset). Format: `%(asctime)s [%(levelname)s] [%(funcName)s] %(message)s`.

### Error handling

- Use `ValueError` for bad inputs or configuration (e.g., glob matching no files, wrong row count).
- Use `NotImplementedError` for abstract method stubs.
- Avoid bare `except` — the only exception is variant-level failure isolation in `ovwt.py:profile_variant`, where individual variant failures are logged and skipped to avoid aborting a long run.

### Docstring style

NumPy style (enforced by mkdocstrings). Sections: `Parameters`, `Returns`, `Raises`. Short one-liner + blank line before Parameters is preferred.

### Commit style

Lowercase verb, optional scope, PR number in parentheses:
```
fix NaN handling, performance improvements (#10)
workflow refactor (#12)
implemented feature selection (#2)
```
No enforced prefix convention (feat:/fix:/chore:), but verbs observed: `fix`, `update`, `implement`, `refactor`, `add`, `revert`.

---

## Gotchas & known issues

1. **`fire` dependency is unused.** It appears in `pyproject.toml` runtime deps but no source file imports or uses it. All CLI entry points use Hydra. Do not add `fire`-based CLIs without discussing the inconsistency first.

2. **`pandas` is a runtime dep but barely used directly.** The codebase uses Polars. `pandas` is almost certainly needed transitively by `pycytominer`. Do not assume Polars ↔ pandas interchangeability — `pycytominer.feature_select` receives a Polars DataFrame that gets converted internally.

3. **Pre-commit only runs Black.** `ruff` and `isort` are in dev dependencies but not in `.pre-commit-config.yaml`. Lint and import-sort failures will not block commits. Run `uv run ruff check .` and `uv run isort src/ tests/` manually before opening a PR.

4. **There is no CI for tests.** `.github/workflows/docs.yml` only deploys MkDocs to GitHub Pages on pushes to `main`. Tests are not run in CI — they must be run locally before merging.

5. **Integration tests are slow and require Nextflow.** `tests/integration/test_integration.py` runs the full Nextflow pipeline on synthetic data using a session-scoped fixture. Skipping them (`uv run pytest tests/unit`) is standard for day-to-day development.

6. **Global Nextflow processes glob for published files, not channel outputs.** `BATCHVSBATCH`, `OVWT_GLOBAL`, and `FEATURE_SELECT_GLOBAL` read from disk after all upstream processes finish. This means: (a) relative `input_dir` paths are resolved to absolute at workflow start; (b) `publishDir` paths in upstream modules must stay in sync with the globs passed into the global calls (`BATCHVSBATCH_PRE` globs `qc_filter/*/filtered_cells.parquet`; `BATCHVSBATCH_POST` and the other global processes glob `normalization/cells/*.parquet`).

7. **When `output_root` is set, it takes priority over `output_dir` in the underlying Hydra CLIs** (`aggregate.py`, `features.py`) — the output file lands at `{output_root}.{stem}.parquet` regardless of `output_dir`, matching pre-existing behavior in `aggregate.py:main`/`features.py:main`. `aggregate_feature_type.nf`, `aggregate_half.nf`, and `finalize_feature_select.nf` all pass `output_root` and therefore use `output_dir=.` (not a subdirectory) plus a glob-based `mv` to rename the result to the process's declared output filename. `finalize_feature_select.nf` additionally does `mkdir -p ft && mv ${feature_type_files} ft/` to isolate the multi-file `feature_type_files` input from the co-staged `block_list_file` before globbing. If you change output naming in `features.py`/`aggregate.py`, update these workarounds.

8. **The README contains `pip install` instructions.** Ignore them. This project uses `uv`. See the Setup section above.

9. **`models.pkl` stores XGBoost `Booster` objects as a `dict[str, xgb.Booster]`.** Pickle is used here (not Parquet) because XGBoost's native serialization requires either the Booster API or pickle. Normalizer stats use Parquet (`Normalizer.save`/`Normalizer.load`) — don't confuse the two.

10. **Synonymous variants are used as the control baseline for aggregation**, not WT cells. In `aggregate.py:variant_classification`, synonymous mutations (first and last amino acid identical in `meta_aa_changes`) are flagged as `meta_is_control = True`. In `normalize.py`, the control is WT cells (the SQL `control_sample_query`). These are different steps with different baselines. `aggregate.feature_type_main` relies on the upstream `meta_is_control` (WT-based) column already present on its input and does not call `variant_classification` itself — that only happens later, in `features.main`'s impact-score step, on the aggregated (not cell-level) data.

---

## PR / commit workflow

No CONTRIBUTING.md exists. Based on git history:

- Branch off `main`, name branches descriptively (no enforced pattern observed).
- PR titles match commit message style: lowercase verb + `(#N)`.
- Squash-merge or merge commits both appear in history.
- Before merging: run `uv run pytest tests/unit`, `uv run ruff check .`, `uv run black --check .`.
- The only automated check on `main` is docs deployment — **tests do not run in CI**.

---

## Safety / do-not-touch list

| Path | Reason |
|------|--------|
| `site/` | Generated MkDocs output — rebuilt by `mkdocs build` and by CI. Editing directly is overwritten on next build. |
| `uv.lock` | Auto-managed by uv. Edit `pyproject.toml` deps instead, then run `uv sync`. |
| `.venv/` | Managed by uv. Never manually install packages into it. |
| `<input_dir>/work/` | Nextflow task working directories — created at runtime, contains intermediate data. Delete only via `nextflow clean`. |
| Any `*.parquet` under `<input_dir>/` | Pipeline output data. Do not modify or commit experiment output files. |
| `example.config` | Template for users — only update if the config format changes. Do not add personal cluster paths. |
| `.github/workflows/docs.yml` | Deploys docs to GitHub Pages. Changes here affect live documentation for all users. |
