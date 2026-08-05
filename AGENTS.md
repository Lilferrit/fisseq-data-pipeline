# AGENTS.md — fisseq-data-pipeline

> **WARNING: `docs/` is stale and must not be trusted.**
> The Markdown files under `docs/` and the generated `site/` directory were written at an earlier stage of the project and have not been kept in sync with the code. Do not use them to understand current behavior, parameter names, or architecture — **except `docs/configuration.md`**, which (like `docs/nextflow.md`'s "Per-batch parameter overrides" section before it) is current and was written alongside this code. This file (AGENTS.md) plus the actual source code and `pyproject.toml` are otherwise the authoritative references.

---

## Project overview

The **FISSEQ Data Pipeline** is a Nextflow + Python workflow for processing single-cell CellProfiler morphological profiling data from FISSEQ (Fluorescence In-Situ Sequencing) experiments. Each cell carries a genetic variant label; the pipeline measures how each variant's cell population differs from wildtype (WT) controls using morphological features.

**End-to-end data flow:**

```
configs/*.yaml  (mandatory, one file per batch — variant selection spec)
      │
      ▼
    INPUT      (per config, always runs)
      │
      ▼
input/*.parquet  (one file per batch, CellProfiler morphological features + barcode annotations)
      │
      ▼
QC_FILTER        (per batch)   ← edit distance, barcode count, variant barcode count
      │
      ├──► BATCHVSBATCH (pre, unfiltered)   (per active global channel in params.global_channels;
      │           default null = none run; scoped to that channel's member batches — see
      │           "Global channels" below)
      ├──► BATCH_CORRECT_FIT/TRANSFORM      (per active global channel)
      │           └──► ANOVA (batch-corrected)  (per active global channel)
      ▼
NORMALIZE        (per batch)   ← z-score fit on WT control cells
      │
      ├──► WTVWT_BATCHWISE      (per batch; skipped if params.run_wtvwt = false; wildtype
      │           cells only, one binary XGBoost classifier per pair of wildtype barcodes;
      │           independent of the ANOVA_BLOCKLIST/OvWT chain below)
      │
      ├──► ANOVA (normalized)   (per active global channel)
      │           │
      │           ▼
      │      ANOVA_BLOCKLIST   (per active global channel — feature_ok from ANOVA p_value < params.anova_blocklist_pvalue_threshold)
      │           │
      │           ├──► BATCHVSBATCH (post, filtered)   (per active global channel)
      │           └──► OVWT_GLOBAL (filtered)             (per active global channel)
      │
      ├──► OVWT_BATCHWISE (unfiltered)      (per batch — no dependency on ANOVA_BLOCKLIST;
      │           skipped if params.run_ovwt = false, which also skips the whole
      │           OVWT_CELLSCORES_BATCHWISE/CHECK_BARCODES/BARCODE_BLOCKLIST/
      │           OVWT_BATCHWISE (barcode-filtered) chain below for that batch)
      │           └──► OVWT_CELLSCORES_BATCHWISE  (per batch; skipped unless params.run_single_cell_scores
      │                     = true, always on in OvwtPipeline; scores the
      │                     params.single_cell_scores_split ["test"|"train"] split)
      │                     └──► CHECK_BARCODES  (per batch; skipped unless
      │                               params.run_check_barcodes = true, which also
      │                               forces run_single_cell_scores on — per-variant
      │                               pairwise Tukey HSD across barcodes)
      │                               └──► BARCODE_BLOCKLIST  (per batch; skipped unless
      │                                         params.run_check_barcodes = true AND
      │                                         params.run_barcode_filtered_ovwt = true
      │                                         (default true, but does NOT itself force
      │                                         run_check_barcodes on) — barcode_ok from
      │                                         median p_adj < params.barcode_blocklist_pvalue_threshold)
      │                                         └──► OVWT_BATCHWISE (barcode-filtered)  (per batch;
      │                                                   FisseqPipeline only)
      └──► Feature selection, batchwise (skipped entirely if
                               params.run_feature_selection = false for that batch):
             AGGREGATE_FEATURE_TYPE      (per feature type)          ─┐
             GENERATE_SPLIT              (per bootstrap replicate)    │
               └─► AGGREGATE_HALF        (per bootstrap × feature type × half)
                     └─► CORRELATE_FEATURES (per bootstrap × feature type)
                           └─► BLOCKLIST  (gathers all bootstrap replicates per feature type — the one sync point)
                                 └─► COMBINE_BLOCKLISTS (gathers all feature types) ┘
                                       └─► FINALIZE_FEATURE_SELECT (joins AGGREGATE_FEATURE_TYPE outputs + combined blocklist)
                                             │
                                             ▼
                                      GLOBAL_FEATURE_SELECT (once per active global channel;
                                        reuses member batches' aggregates/blocklist above
                                        directly off pipeline_dir -- no cell-level recompute)
```

**Global channels.** A batch's YAML `global_channel` key (string or list of
strings) names which channel(s) it belongs to; `params.global_channels`
(pipeline-wide, default `null`) lists which of those channels actually run.
Each active channel gets its own `BATCHVSBATCH`/`OVWT_GLOBAL`/`ANOVA`
(both calls)/`BATCH_CORRECT_FIT`+`BATCH_CORRECT_TRANSFORM` run, scoped to
only that channel's member batches, via a small staging process
`STAGE_CHANNEL_CELLS` (`modules/local/stage_channel.nf`, aliased `STAGE_CHANNEL_QC`/
`STAGE_CHANNEL_NORM`) that republishes each member batch's `QC_FILTER`/`NORMALIZE`
output into `<pipeline_dir>/global/<channel>/{qc_filter_cells,normalization_cells}/`.
This makes the entire `ANOVA`/`ANOVA_BLOCKLIST`/`BATCH_CORRECT_FIT`/
`BATCH_CORRECT_TRANSFORM`/`ANOVA_BATCH_CORRECTED` chain a purely per-channel
feature too: with `params.global_channels` unset (default), none of it runs,
and a batch belonging to no active channel is skipped by the whole chain
(processed batchwise as normal otherwise); a batch in multiple channels runs
through the whole chain once per channel, independently, each publishing
under its own channel's subtree.
`GLOBAL_FEATURE_SELECT` also runs once per active channel, but does not use
this cell-staging mechanism at all — it reads each member batch's own
`feature_select_batchwise/<batch>/{aggregates,blocklist.parquet}` directly off
`pipeline_dir` (member batches resolved straight from `resolvedBatchConfigs`
in Groovy, only those with `run_feature_selection` enabled), since it has no
cell-level work to do — see
[Configuration: Global channels](docs/configuration.md#global-channels) and
gotcha 6 below.

`OVWT_BATCHWISE` is a single parameterized process invoked twice (in
`FisseqPipeline`; once, always unfiltered, in `OvwtPipeline`) via
`include { OVWT_BATCHWISE as X }` aliasing (like `ANOVA`/`BATCHVSBATCH`): the
unfiltered call (`ovwt_batchwise/`) keeps the pipeline's original behavior and
scheduling (no dependency on `ANOVA_BLOCKLIST` or `BARCODE_BLOCKLIST`); the
barcode-filtered call (`ovwt_batchwise_barcode_filtered/`) additionally
depends on that batch's `BARCODE_BLOCKLIST` output and only runs when both
`params.run_check_barcodes` (default `false`) and
`params.run_barcode_filtered_ovwt` (default `true`) are true —
`run_barcode_filtered_ovwt` deliberately does *not* force `run_check_barcodes`
on, so the default (`run_check_barcodes=false`) pipeline output is unaffected
by `run_barcode_filtered_ovwt`'s own default. (There used to be a third,
feature-filtered alias, `OVWT_BATCHWISE_FEATURE_FILTERED`, gated by
`params.run_feature_filtered_ovwt` — it was removed once `ANOVA_BLOCKLIST`
became per-channel: a per-batch process broadcasting one of several
per-channel blocklists onto a batch in 0 or 2+ channels had no single
well-defined blocklist to use. `BATCHVSBATCH_POST`/`OVWT_GLOBAL`, both
already per-channel, don't have this problem — each just joins on its own
channel's blocklist.)
`OVWT_GLOBAL` has no unfiltered counterpart — it is always feature-filtered,
and has no barcode-filtered counterpart (`BARCODE_BLOCKLIST` runs per batch,
not globally).
`ANOVA_BLOCKLIST` (note: distinct from the correlation-based `BLOCKLIST`
process in the feature-selection branch above) derives its block-list from
`ANOVA_NORMALIZED` specifically, not `ANOVA_BATCH_CORRECTED` — OvWT and
batch-vs-batch score normalized cells, never batch-corrected ones.
`BARCODE_BLOCKLIST` (per batch, unlike the global `ANOVA_BLOCKLIST`) derives
its block-list from that batch's own `CHECK_BARCODES` output, aggregating
each barcode's `p_adj` values (pooled across both the `barcode` and
`comparison_barcode` columns, since a barcode's pairwise comparisons are
scattered across both depending on alphabetical order) via median.
`WTVWT_BATCHWISE` (per batch) is a single process (`modules/local/wtvwt_batchwise.nf`,
wrapping `python -m fisseq_data_pipeline.wtvwt`) invoked once, not aliased —
unlike `OVWT_BATCHWISE` it has no feature-filtered/barcode-filtered variants
and no global counterpart. It restricts to wildtype-labeled cells, computes a
single 80/10/10 split stratified by barcode (so every wildtype barcode's rows
span all three splits), then trains one binary XGBoost classifier per pair of
wildtype barcodes (`itertools.combinations` over the surviving barcodes).
Gated per batch by `params.run_wtvwt` (default `true`), independent of every
other gate — it only depends on `NORMALIZE`'s output, not `ANOVA_BLOCKLIST`.

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

The package installs in editable mode, so each pipeline step is immediately runnable via `uv run python -m fisseq_data_pipeline.qcfilter`, etc.

### Cluster / HPC

The repo ships a `nextflow.config` at the root with default `params` values and commented-out profile stubs (`venv`, `conda`, `singularity`, `sge`). To run on a cluster, write your own config (or copy and adapt `nextflow.config`), uncomment/fill in a `beforeScript` option (venv activation, package install) in a profile block, and pass it with `-c your.config -profile <name>`.

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
nextflow run main.nf --pipeline_dir /path/to/experiment

# SGE cluster (supply your own env config)
nextflow run main.nf -c your.config -profile sge --pipeline_dir /path/to/experiment

# Resume after interruption (Nextflow task caching)
nextflow run main.nf --pipeline_dir /path/to/experiment -resume
```

### Run individual Python pipeline steps

Each step is a Hydra entry point run as `python -m fisseq_data_pipeline.<module>`:

```bash
uv run python -m fisseq_data_pipeline.qcfilter \
    output_dir=./out \
    'cell_files=[data/plate1.parquet]' \
    bc_threshold=10

uv run python -m fisseq_data_pipeline.normalize \
    output_dir=./out \
    input_file=out/filtered_cells.parquet

uv run python -m fisseq_data_pipeline.aggregate \
    output_dir=./out \
    input_file=out/normalized.parquet \
    aggregator=mean

uv run python -m fisseq_data_pipeline.aggregatefeaturetype \
    output_dir=./out \
    input_file=out/normalized.parquet \
    aggregator=mean

uv run python -m fisseq_data_pipeline.generatesplit \
    output_dir=./out \
    input_file=out/normalized.parquet \
    random_state=1

uv run python -m fisseq_data_pipeline.correlatefeatures \
    output_dir=./out \
    half1_file=out/half1.mean.parquet \
    half2_file=out/half2.mean.parquet

uv run python -m fisseq_data_pipeline.blocklist \
    output_dir=./out \
    'correlation_files=out/correlations/mean/*.parquet' \
    minimum_correlation=0.5

uv run python -m fisseq_data_pipeline.combineblocklists \
    output_dir=./out \
    'blocklist_files=out/blocklists/*.parquet'

uv run python -m fisseq_data_pipeline.featureselect \
    output_dir=./out \
    input_file=out/normalized.parquet \
    'feature_type_files=out/aggregates/*.parquet' \
    block_list_file=out/blocklist.parquet

uv run python -m fisseq_data_pipeline.anova \
    output_dir=./out \
    'input_file=data/batches/*.parquet'

uv run python -m fisseq_data_pipeline.ovwt \
    output_dir=./out \
    input_file=out/features.parquet \
    min_cells=250

uv run python -m fisseq_data_pipeline.ovwtcellscores \
    output_dir=./out \
    input_file=out/normalized.parquet \
    models_path=out/models.pkl
```

```bash
uv run python -m fisseq_data_pipeline.batchvsbatch \
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
│   ├── input.py                   # Optional input-file generation entry point (variant selection)
│   ├── qcfilter.py                # QC filtering entry point (optional pseudo-variant downsampling)
│   ├── normalize.py               # Normalizer class + normalize entry point
│   ├── aggregate.py               # 8 aggregation strategies + standalone full-aggregation entry point
│   ├── aggregatefeaturetype.py    # Lean per-feature-type aggregation entry point (imports aggregate.py)
│   ├── generatesplit.py           # Bootstrap pseudo-replicate split-generation entry point
│   ├── correlatefeatures.py       # Pseudo-replicate feature correlation entry point
│   ├── blocklist.py               # Per-feature-type bootstrap blocklist entry point
│   ├── combineblocklists.py       # Combine per-feature-type blocklists entry point
│   ├── featureselect.py           # Final BATCHWISE pycytominer feature-selection entry point
│   ├── globalfeatureselect.py     # Global (per channel) feature-selection entry point -- reuses BATCHWISE output
│   ├── batchcorrect.py            # BatchCorrector class + batch-correction fit entry point
│   ├── batchcorrecttransform.py   # Batch-correction transform entry point (imports batchcorrect.py)
│   ├── ovwt.py                    # XGBoost one-vs-WT training + entry point
│   ├── ovwtcellscores.py          # Cell scoring via trained models
│   ├── checkbarcodes.py           # Per-variant pairwise Tukey HSD across barcodes
│   ├── barcodeblocklist.py        # CHECK_BARCODES-derived barcode block-list entry point
│   ├── batchvsbatch.py            # Per-variant multiclass batch classifier; OvR AUC + Mann-Whitney p-value
│   ├── wtvwt.py                   # Wildtype-only pairwise-barcode XGBoost classifier + entry point
│   ├── anova.py                   # Per-feature one-way ANOVA entry point
│   └── anovablocklist.py          # ANOVA-derived feature block-list entry point
├── modules/local/
│   ├── input.nf
│   ├── qc_filter.nf
│   ├── normalize.nf
│   ├── anova.nf
│   ├── anova_blocklist.nf
│   ├── batchvsbatch.nf
│   ├── ovwt_batchwise.nf
│   ├── ovwt_global.nf
│   ├── ovwt_cellscores_batchwise.nf
│   ├── wtvwt_batchwise.nf
│   ├── check_barcodes.nf
│   ├── barcode_blocklist.nf
│   ├── aggregate_feature_type.nf
│   ├── generate_split.nf
│   ├── aggregate_half.nf
│   ├── correlate_features.nf
│   ├── blocklist.nf
│   ├── combine_blocklists.nf
│   ├── finalize_feature_select.nf
│   └── global_feature_select.nf
├── workflows/
│   ├── fisseq.nf                  # Main Nextflow workflow DAG
│   └── ovwt.nf                    # Lighter OvWT-only workflow DAG (--pipeline_mode ovwt)
├── lib/
│   └── BatchParams.groovy         # Per-batch YAML override merge/validation (see Gotcha 13)
├── main.nf                        # Nextflow entrypoint (dispatches FisseqPipeline/OvwtPipeline)
├── nextflow.config                # Default params + commented-out profile stubs (venv/conda/singularity/sge)
├── tests/
│   ├── unit/                      # 20 files, fast, synthetic data
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

**`BaseAggregator`** (`aggregate.py`) — abstract base for 8 concrete aggregation strategies (mean, median, MAD, std, KS, signedKS, QQ, AUROC). There is no multi-aggregator wrapper; combining feature types happens in Nextflow — `aggregate.feature_type_main` runs once per `params.feature_select_types` entry, and `features.main` (the final feature-selection stage) joins the per-feature-type outputs on the label column.

**`utils/xgbparams.py`** — shared XGBoost infrastructure imported by `ovwt.py`, `ovwtcellscores.py`, `batchvsbatch.py`, and `wtvwt.py`. Contains: `XGBoostParams` and `XGBoostConfig` dataclasses; `get_feature_cols` (CellProfiler column detection); `get_dmatrix` (binary DMatrix builder); `get_dmatrix_multiclass` (multiclass DMatrix with sorted integer encoding); `split_indices_stratified` (80/10/10 stratified split on any label array). Do not add XGBoost-specific infrastructure to individual modules — put it here. `wtvwt.py` reuses `get_dmatrix`/`get_feature_cols`/`split_indices_stratified` directly (calling `get_dmatrix` with `barcode_column`/one barcode of a pair in place of `label_column`/`wt_label` — the function is already fully generic), rather than adding a barcode-specific DMatrix helper.

**`load_batches`** (`utils/batches.py`) — accepts a path or glob pattern, reads matching Parquet files, tags each with `meta_batch` = filename stem, returns a concatenated `pl.LazyFrame` plus an output stem string.

**Nextflow synchronization pattern** (`workflows/fisseq.nf`): `ANOVA`/`BATCHVSBATCH`/`OVWT_GLOBAL`/`BATCH_CORRECT_FIT` all wait per active global channel, via `channel_qc_signal_ch`/`channel_norm_signal_ch` (each built from `STAGE_CHANNEL_QC`/`STAGE_CHANNEL_NORM`'s output collected with `.groupTuple()`, keyed by channel — see "Global channels" above and gotcha 6 below) rather than one flattened signal. `GLOBAL_FEATURE_SELECT` uses neither signal — it waits on `feature_select_ready_signal` (built the same "collect batch stems -> map to `pipeline_dir_abs`" way, but from `combined_bl_ch`, the last BATCHWISE feature-select artifact) since it only ever needs `feature_select_batchwise/` output, never staged cells. `ANOVA` is a single parameterized process (`modules/local/anova.nf`) invoked twice via Nextflow's `include { X as Y }` aliasing (a process cannot be called twice under its own name in one workflow), each alias additionally invoked once per active global channel and threading `channel` through its output tuple so downstream consumers can join back to the right channel; `BATCHVSBATCH` (`modules/local/batchvsbatch.nf`) is likewise aliased twice, each alias additionally invoked once per active global channel — `BATCHVSBATCH_PRE` waits on `channel_qc_signal_ch` and globs that channel's staged `qc_filter_cells/*.parquet` with `use_parent_name=false` (every `STAGE_CHANNEL_CELLS` output is flattened to `<batch_stem>.parquet`, so parent-dir naming is no longer needed), unfiltered; `BATCHVSBATCH_POST` waits on `channel_norm_signal_ch` joined with `anova_blocklist_ch` (`ANOVA_BLOCKLIST`'s output, now one-per-channel) and globs that channel's staged `normalization_cells/*.parquet`, filtered. `ANOVA_NORMALIZED` waits on `channel_norm_signal_ch`/`normalization_cells/*.parquet` and `ANOVA_BATCH_CORRECTED` waits on `channel_bc_signal_ch`/`batch_correction/cells/*.parquet` — both ANOVA calls run once per active global channel, scoped to only that channel's batches; with no active channels, neither runs at all. `ANOVA_NORMALIZED`'s call sits earlier in `workflows/fisseq.nf` than `ANOVA_BATCH_CORRECTED`'s (right after `channel_norm_signal_ch` is computed, not at the bottom) so its output channel can feed `ANOVA_BLOCKLIST`, which in turn is `.join()`ed (per-channel, not a broadcast) onto `OVWT_GLOBAL` and `BATCHVSBATCH_POST` — `.join()` is safe here specifically because both sides are already collapsed to exactly one item per channel; `BATCH_CORRECT_TRANSFORM` instead uses `.combine(fit_out, by: 0)` for `BATCH_CORRECT_FIT`'s output, since the batch side genuinely fans out N batches per channel (not one-per-channel), the many-to-one shape `.join()` must never be used for. `OVWT_BATCHWISE` is likewise a single parameterized process (`modules/local/ovwt_batchwise.nf`) aliased, in `FisseqPipeline`, into `OVWT_BATCHWISE_UNFILTERED` (no dependency on `ANOVA_BLOCKLIST` or `BARCODE_BLOCKLIST`; optional, gated by `params.run_ovwt`, default `true`) and `OVWT_BATCHWISE_BARCODE_FILTERED` (depends on that batch's `BARCODE_BLOCKLIST` output; optional, gated by `params.run_barcode_filtered_ovwt`, default `true`); a third alias, `OVWT_BATCHWISE_FEATURE_FILTERED` (gated by the now-removed `params.run_feature_filtered_ovwt`), used to exist but was removed once `ANOVA_BLOCKLIST` became per-channel (see "Global channels" above); the varying `feature_block_list_file`/`barcode_block_list_file` values are each passed as a `val` (not a Nextflow-staged `path`) so a Groovy `null` can flow straight through to `python -m fisseq_data_pipeline.ovwt`'s `feature_block_list_file=null`/`barcode_block_list_file=null` CLI args for calls that don't use them, the same `null`-passthrough trick `qc_filter.nf` uses for `qc_downsample_amounts`/`qc_n_variants`. All varying bits (glob path, `use_parent_name`, the two block-list vals, `publishDir` subpath) are passed in as process input values, not hardcoded per-call. Since `BARCODE_BLOCKLIST` runs per batch (unlike the per-channel `ANOVA_BLOCKLIST`), `OVWT_BATCHWISE_BARCODE_FILTERED`'s input channel uses `norm_ch.join(barcode_blocklist_ch)` (both already keyed one-per-batch_stem) rather than `.combine()` (which is reserved for broadcasting a single value across every element of another channel). `OVWT_BATCHWISE`'s output tuple carries both `test_index.parquet` and `train_index.parquet` (the underlying `ovwt.py:main` always writes both, plus `val_index.parquet`, when `save_splits=True`; only the first two are wired into Nextflow).

**Feature-selection pipeline** (`workflows/fisseq.nf`) has two independent parts. The BATCHWISE branch follows the same aliasing pattern as `ANOVA`/`BATCHVSBATCH`, applied to 7 processes (`AGGREGATE_FEATURE_TYPE`, `GENERATE_SPLIT`, `AGGREGATE_HALF`, `CORRELATE_FEATURES`, `BLOCKLIST`, `COMBINE_BLOCKLISTS`, `FINALIZE_FEATURE_SELECT`), each aliased once as `*_BATCHWISE` (no `*_GLOBAL` alias exists anymore — see below). Channels are crossed via `.combine()` over `feature_types_ch` (`params.feature_select_types`) and `bootstrap_ch` (`1..params.feature_select_bootstrap_reps`), split into per-half tuples via `.flatMap()`, and re-paired via `.groupTuple()`. `BLOCKLIST`'s `groupTuple(by: [batch_stem, feature_type])` — gathering all `params.feature_select_bootstrap_reps` correlation replicates for one feature type before computing a median-`r` threshold — is the pipeline's only cross-bootstrap synchronization point; everything else in the split/aggregate/correlate chain is fully parallel across bootstrap × feature type (× half). `params.run_feature_selection` (default `true`, per-batch overridable) gates this whole branch per batch, via `norm_ch_feature_selected`'s `.filter()`.

`GLOBAL_FEATURE_SELECT` (`globalfeatureselect.py`, `modules/local/global_feature_select.nf`) is a separate, single, non-aliased process — not part of the 7-process aliasing pattern above, and no bootstrap/split/correlate machinery of its own. It runs once per active channel in `params.global_channels` (default `null` — none run at all), also gated on the pipeline-wide `params.run_feature_selection`. Per channel, batch membership (only batches with THEIR OWN `run_feature_selection` enabled) is resolved directly from `resolvedBatchConfigs` in Groovy (`batchesByChannel`, built with `.collectEntries`/`.findAll`, no channel involved) and passed as a plain `val(batch_stems)` list; the process then loops over those batch stems in Python, reading each one's already-published `feature_select_batchwise/<batch>/{aggregates,blocklist.parquet}` directly off `pipeline_dir` — join per-feature-type files, normalize to that batch's own synonymous baseline, take the cross-batch median, combine blocklists by agreement threshold (`params.global_feature_select_min_batches_ok`), run `pyc_feature_select`. `OVWT_GLOBAL`, both aliased `BATCHVSBATCH` calls (`_PRE`/`_POST`), `ANOVA` (`_NORMALIZED`/`_BATCH_CORRECTED`), and the batch-correction branch (`BATCH_CORRECT_FIT`/`BATCH_CORRECT_TRANSFORM`) all likewise run once per active channel, independently of `GLOBAL_FEATURE_SELECT` and of each other (e.g. run global processes separately per experiment/site/cohort, or not at all).

**Single-cell scores and barcode-outlier detection** (`workflows/fisseq.nf`, `workflows/ovwt.nf`): `OVWT_CELLSCORES_BATCHWISE` (per batch, scores cells against `OVWT_BATCHWISE_UNFILTERED`'s models) is gated by `params.run_single_cell_scores` (default `false`) in `FisseqPipeline`, but always runs in `OvwtPipeline` (that workflow's entire purpose). In `FisseqPipeline`, this whole chain (`OVWT_CELLSCORES_BATCHWISE` → `CHECK_BARCODES` → `BARCODE_BLOCKLIST` → `OVWT_BATCHWISE_BARCODE_FILTERED`) also implicitly requires `params.run_ovwt` (default `true`) for that batch, since it consumes `OVWT_BATCHWISE_UNFILTERED`'s output directly — no separate "implies" check is needed for this since an empty upstream channel just produces an empty downstream one. In both workflows, `params.single_cell_scores_split` (`"test"` or `"train"`, default `"test"`; any other value fails fast with a clear error) selects which of `OVWT_BATCHWISE`'s two split-index outputs to score. Downstream, `CHECK_BARCODES` (per batch; per variant, a pairwise Tukey HSD across that variant's barcodes using each cell's own-model score as the response variable, via `checkbarcodes.py:compute_barcode_tukey` — computed as a single vectorized sufficient-statistics groupby + self-join, following `anova.py`'s pattern, rather than looping `statsmodels.stats.multicomp.pairwise_tukeyhsd` per variant) is gated by `params.run_check_barcodes` (default `false`), which also forces `run_single_cell_scores` on — so setting `run_check_barcodes = true` alone is sufficient; you don't need to also set `run_single_cell_scores`. `params.barcode_check_min_cells` (default `10`) drops barcodes with fewer cells before comparison; variants left with fewer than 2 qualifying barcodes are skipped (nothing to compare). `params.barcode_check_alpha` (default `0.05`) is the family-wise significance level for the `reject` flag.

`BARCODE_BLOCKLIST` (per batch, `FisseqPipeline` only), downstream of `CHECK_BARCODES`, only runs when both `params.run_check_barcodes` (default `false`) and `params.run_barcode_filtered_ovwt` (default `true`) are true — unlike `run_check_barcodes`/`run_single_cell_scores`, `run_barcode_filtered_ovwt` deliberately does *not* force `run_check_barcodes` on, so the default pipeline output is unaffected by `run_barcode_filtered_ovwt`'s own default; you must explicitly set `run_check_barcodes = true` to get `BARCODE_BLOCKLIST`/`OVWT_BATCHWISE_BARCODE_FILTERED` output. It aggregates each barcode's `p_adj` via `barcodeblocklist.py:compute_barcode_blocklist`: both the `barcode` and `comparison_barcode` columns are unioned before `group_by("barcode")`, since a barcode's pairwise comparisons land in either column depending on alphabetical order relative to its partner — grouping on `barcode` alone would silently drop/undercount some barcodes' p_adj values. `barcode_ok = median(p_adj) >= params.barcode_blocklist_pvalue_threshold` (default `0.05`). Its output feeds `OVWT_BATCHWISE_BARCODE_FILTERED` via `python -m fisseq_data_pipeline.ovwt`'s `barcode_block_list_file`, which drops cell *rows* whose `barcode_column` (default `meta_barcode`) value is blocked — independent of and additive with `feature_block_list_file`, which drops feature *columns*. In `ovwt.py:train_test_val_split`, the barcode row-filter runs immediately after `with_row_index` and before the feature-column `select` (so `meta_barcode` is still present and `__row_idx__` still reflects true original position), and therefore before `min_cells` filtering — a variant that drops below `min_cells` purely because its barcode(s) got blocked is correctly excluded.

### Pipeline step entry points (run via `python -m fisseq_data_pipeline.<module>`)

| Invocation | Purpose |
|---|---|
| `python -m fisseq_data_pipeline.input` | Optional upstream stage: variant selection from a YAML spec, producing one `input/*.parquet` file |
| `python -m fisseq_data_pipeline.qcfilter` | Edit distance + barcode QC (optional pseudo-variant downsampling) |
| `python -m fisseq_data_pipeline.normalize` | Z-score normalization |
| `python -m fisseq_data_pipeline.aggregate` | Standalone per-variant aggregation + normalizer + metadata (not wired into Nextflow) |
| `python -m fisseq_data_pipeline.aggregatefeaturetype` | Lean per-feature-type aggregation, optionally filtered to an index-file row subset |
| `python -m fisseq_data_pipeline.generatesplit` | Generate one stratified 50/50 pseudo-replicate split |
| `python -m fisseq_data_pipeline.correlatefeatures` | Per-feature Pearson correlation between two aggregate halves |
| `python -m fisseq_data_pipeline.blocklist` | Median-`r`-across-bootstraps blocklist for one feature type |
| `python -m fisseq_data_pipeline.combineblocklists` | Concatenate per-feature-type blocklists |
| `python -m fisseq_data_pipeline.featureselect` | Final BATCHWISE stage: joins per-feature-type aggregates, applies combined blocklist, pycytominer selection |
| `python -m fisseq_data_pipeline.globalfeatureselect` | Global (per channel) stage: reuses member batches' BATCHWISE aggregates/blocklist directly, no cell-level recompute |
| `python -m fisseq_data_pipeline.batchcorrect` | Fit two-pass centroid batch correction across all batches |
| `python -m fisseq_data_pipeline.batchcorrecttransform` | Apply a fitted batch correction to a single batch |
| `python -m fisseq_data_pipeline.ovwt` | One-vs-WT XGBoost training |
| `python -m fisseq_data_pipeline.ovwtcellscores` | Score cells against trained OvWT models |
| `python -m fisseq_data_pipeline.checkbarcodes` | Per-variant pairwise Tukey HSD across barcodes (single-cell scores as response) |
| `python -m fisseq_data_pipeline.barcodeblocklist` | Barcode block-list derived from CHECK_BARCODES p-values (per batch) |
| `python -m fisseq_data_pipeline.anova` | Per-feature one-way ANOVA |
| `python -m fisseq_data_pipeline.anovablocklist` | Feature block-list derived from ANOVA p-values |
| `python -m fisseq_data_pipeline.batchvsbatch` | Per-variant multiclass batch classifier (OvR AUC + Mann-Whitney p per batch) |
| `python -m fisseq_data_pipeline.wtvwt` | Wildtype-only: one binary XGBoost classifier per pair of wildtype barcodes |

All share base Hydra fields: `output_dir` (required), `output_root` (optional prefix), `log_level` (default `"info"`).

### Output layout

All outputs land under `<pipeline_dir>` alongside the mandatory `configs/`
and `input/` folders:

```
<pipeline_dir>/
  configs/*.yaml                # mandatory, one per batch
  qc_filter/<batch>/
    filtered_cells.parquet
    barcode_counts.parquet
    variants_per_barcode.parquet
  normalization/
    cells/<batch>.parquet
    normalizers/<batch>.normalizer.parquet
  ovwt_batchwise/<batch>/       # unfiltered — full feature set
    results.parquet
    models.pkl
    test_index.parquet          # columns: row_idx, origin_file
    train_index.parquet         # columns: row_idx, origin_file
                                 # optional: params.run_ovwt (FisseqPipeline only; always
                                 # runs in OvwtPipeline)
  ovwt_batchwise_barcode_filtered/<batch>/  # filtered against that batch's
                                 # barcode_blocklist/<batch>/barcode_blocklist.parquet;
                                 # FisseqPipeline only, optional: params.run_barcode_filtered_ovwt
    results.parquet
    models.pkl
    test_index.parquet
    train_index.parquet
  wtvwt_batchwise/<batch>/      # wildtype cells only; optional: params.run_wtvwt
    results.parquet             # one row per unordered barcode pair; columns: barcode_a, barcode_b,
                                 # train/val/test_auroc, train/val/test_accuracy, n_cells_a, n_cells_b
    models.pkl                  # dict[(barcode_a, barcode_b) -> xgb.Booster]
  ovwt_cellscores_batchwise/<batch>/  # optional: params.run_single_cell_scores (always on in OvwtPipeline)
    cell_scores.parquet         # one column per variant model, plus meta_* columns
  check_barcodes/<batch>/       # optional: params.run_check_barcodes (implies run_single_cell_scores)
    results.parquet             # one row per compared barcode pair; columns: variant, barcode,
                                 # group_mean, comparison_barcode, comparison_group_mean, mean_diff,
                                 # p_adj, reject
  barcode_blocklist/<batch>/    # optional: params.run_check_barcodes AND params.run_barcode_filtered_ovwt both true
    barcode_blocklist.parquet   # one row per distinct barcode; columns: barcode, p_adj (median), barcode_ok
  feature_select_batchwise/<batch>/
    aggregates/<feature_type>.parquet                                     # stage 1
    splits/bootstrap_<n>/half{1,2}.parquet                                # stage 2a
    half_aggregates/bootstrap_<n>/<feature_type>/half{1,2}.parquet        # stage 2b
    correlations/<feature_type>/bootstrap_<n>.parquet                    # stage 2c
    blocklists/<feature_type>.parquet                                    # stage 2d
    blocklist.parquet                                                    # stage 3 (combined)
    output.parquet                                                       # stage 4 (final)
  global/<channel>/               # one subtree per channel in params.global_channels
                                   # (default none) -- only that channel's member batches
                                   # (per-batch YAML global_channel) contribute. This is
                                   # the ONLY place ANOVA/ANOVA_BLOCKLIST/BATCH_CORRECT_*
                                   # output lands -- with no active channels, none of it
                                   # exists anywhere in pipeline_dir.
    qc_filter_cells/<batch>.parquet        # STAGE_CHANNEL_QC staging output
    normalization_cells/<batch>.parquet    # STAGE_CHANNEL_NORM staging output
    batchvsbatch/
      pre/results.parquet         # pre batch correction (QC-filtered cells), unfiltered; columns: variant, batch, auroc, mw_pvalue, n_batch_cells, n_cells
      post/results.parquet        # post batch correction (normalized cells), filtered against this channel's own anova_blocklist/anova_blocklist.parquet
    anova/
      anova.parquet                # from this channel's normalized cells
    anova_blocklist/
      anova_blocklist.parquet      # derived from this channel's anova/anova.parquet
    ovwt_global/                  # always filtered against this channel's own anova_blocklist/anova_blocklist.parquet
      results.parquet
      models.pkl
    batch_correction/
      fit/stats_vb.parquet         # fit over this channel's own QC-filtered batches only
      fit/centroids.parquet
      cells/<batch>.parquet        # one task per (channel, batch) pair -- a batch in
                                    # multiple channels gets a differently-corrected copy
                                    # under each channel's own subtree
      anova/anova.parquet          # from this channel's batch-corrected cells
    feature_select/                # GLOBAL_FEATURE_SELECT -- reuses member batches'
                                    # feature_select_batchwise/<batch>/ above directly,
                                    # not staged cells; no splits/half_aggregates/
                                    # correlations/blocklists subtree of its own
      aggregate.parquet            # cross-batch median aggregate, pycytominer-selected
      blocklist.parquet            # combined blocklist (agreement threshold across
                                    # member batches' own blocklist.parquet)
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

6. **Global Nextflow processes glob for published files, not channel outputs.** `BATCHVSBATCH`, `OVWT_GLOBAL`, `ANOVA`, and `BATCH_CORRECT_FIT` read from disk after all upstream processes finish. This means: (a) relative `pipeline_dir` paths are resolved to absolute at workflow start; (b) `publishDir` paths in upstream modules must stay in sync with the globs passed into the global calls. Since these processes run once per active global channel rather than once total, they don't glob the whole pipeline tree -- `STAGE_CHANNEL_CELLS` (`modules/local/stage_channel.nf`) first republishes each channel's member batches' `QC_FILTER`/`NORMALIZE` output into `global/<channel>/{qc_filter_cells,normalization_cells}/` (flattened to `<batch_stem>.parquet` regardless of source), and `BATCHVSBATCH_PRE`/`BATCH_CORRECT_FIT` glob that channel's `qc_filter_cells/*.parquet` while `BATCHVSBATCH_POST`/`OVWT_GLOBAL`/`ANOVA_NORMALIZED` glob that channel's `normalization_cells/*.parquet`. `GLOBAL_FEATURE_SELECT` follows the same "glob published output, not channel output" idiom but against a different tree entirely: it globs each member batch's `feature_select_batchwise/<batch>/{aggregates,blocklist.parquet}` directly (member batches passed in as a plain `val(batch_stems)` list, not a staged directory), so it has no dependency on `STAGE_CHANNEL_CELLS`/`normalization_cells/` at all.

7. **When `output_root` is set, it takes priority over `output_dir` in the underlying Hydra CLIs** (`aggregate.py`, `featureselect.py`) — the output file lands at `{output_root}.{stem}.parquet` regardless of `output_dir`, matching pre-existing behavior in `aggregate.py:main`/`featureselect.py:main`. `aggregate_feature_type.nf`, `aggregate_half.nf`, and `finalize_feature_select.nf` all pass `output_root` and therefore use `output_dir=.` (not a subdirectory) plus a glob-based `mv` to rename the result to the process's declared output filename. `finalize_feature_select.nf` additionally does `mkdir -p ft && mv ${feature_type_files} ft/` to isolate the multi-file `feature_type_files` input from the co-staged `block_list_file` before globbing. If you change output naming in `featureselect.py`/`aggregate.py`, update these workarounds. `globalfeatureselect.py`/`global_feature_select.nf` do not follow this pattern at all — no `output_root`, no `mv`; output filenames (`aggregate.parquet`, `blocklist.parquet`) are hardcoded, matching `blocklist.py`/`combineblocklists.py`'s simpler convention instead.

8. **The README contains `pip install` instructions.** Ignore them. This project uses `uv`. See the Setup section above.

9. **`models.pkl` stores XGBoost `Booster` objects as a `dict[str, xgb.Booster]`.** Pickle is used here (not Parquet) because XGBoost's native serialization requires either the Booster API or pickle. Normalizer stats use Parquet (`Normalizer.save`/`Normalizer.load`) — don't confuse the two.

10. **Synonymous variants are used as the control baseline for aggregation**, not WT cells. In `aggregate.py:variant_classification`, synonymous mutations (first and last amino acid identical in `meta_aa_changes`) are flagged as `meta_is_control = True`. In `normalize.py`, the control is WT cells (the SQL `control_sample_query`). These are different steps with different baselines. `aggregate.feature_type_main` relies on the upstream `meta_is_control` (WT-based) column already present on its input and does not call `variant_classification` itself — that only happens later, in `features.main`'s impact-score step, on the aggregated (not cell-level) data.

11. **`aaChanges`/`meta_aa_changes` may carry a `:<tag>` metadata suffix** (e.g. `V123A:downsampled-half`). Two independent things can put a tag there: (a) upstream raw data may already arrive pre-tagged (any raw file fed into `QC_FILTER` via a batch's `input_paths`); (b) `qcfilter.py`'s own optional pseudo-variant downsampling step (`downsample_amounts`), which runs *after* `filter_columns` inside `qcfilter.py:main` and sets `meta_variant_tag` directly on the pseudo rows it generates — those rows never pass through the tag-split step at all, since it already ran earlier in the same call. Each amount in `downsample_amounts` gets its own tag, `downsample-{amount}` (e.g. `downsample-0.5`, `downsample-500`), not a single fixed `:downsampled` suffix. For any row that arrives with a raw, still-suffixed label, the tag is stripped exactly once, in `qcfilter.py:filter_columns` — `meta_aa_changes` is always the tag-stripped base and `meta_variant_tag` holds the tag (`null` when absent). Every stage after `QC_FILTER` therefore sees clean, pooled variant labels either way; do not re-strip or re-parse tags downstream — if you need to segment on the tag, use `meta_variant_tag` directly. `utils/variant.py:classify_variant` assumes its input is already tag-stripped. Note: `barcode_counts.parquet`/`variants_per_barcode.parquet` are computed by `add_qc_queries` *before* the pseudo-variant downsampling step runs (but *after* the optional `n_variants` variant-level selection step, which now also lives in `qcfilter.py`), so they never include pseudo rows, even when `downsample_amounts` is set.

12. **`<pipeline_dir>/configs/` is mandatory** — every batch is declared by a YAML config file there; `INPUT` generates `input/*.parquet` from it (one task per config file). There is no pre-staged-parquet mode. Like every other process, `INPUT` uses `errorStrategy 'ignore'` — a failed config conversion just drops that batch from the run rather than aborting, so a "missing" batch in the output may mean its `INPUT` task failed, not that it was never requested.

13. **Any batch YAML in `configs/` can override most `nextflow.config` params for just that batch** — resolved once per batch, in Groovy, by `lib/BatchParams.groovy`'s `resolve()` (see `docs/configuration.md`'s "Per-batch parameter overrides" and "Global channels" sections for the full mechanism; unlike most of `docs/`, that page is current and was written alongside this code). Every override is logged via `log.info` at workflow-construction time — never buried in a process script. Four things to keep in mind:
    - `input_paths` is the one required, batch-YAML-only key with **no** `nextflow.config` default — there's no sensible pipeline-wide default for a per-batch list of raw data files. A batch YAML that omits it fails clearly.
    - `global_channel` (string or list of strings) is the other batch-YAML-only key, but **optional** — a batch omitting it simply never contributes to any global run (see "Global channels" above and gotcha 6).
    - Not every param has a per-batch meaning. `--global_channels`, `--batchvsbatch_min_cells`, `--batchvsbatch_min_batches`, `--anova_blocklist_pvalue_threshold`, `--feature_select_types`, `--feature_select_bootstrap_reps`, and `--global_feature_select_min_batches_ok` are consumed only by processes that run once across *all* (or, for most of these, all of one channel's) batches (`BATCHVSBATCH`, `OVWT_GLOBAL`, `ANOVA_BLOCKLIST`, `GLOBAL_FEATURE_SELECT`), so they're pipeline-wide-only — a batch YAML that tries to set one of these gets a clear rejection error (distinct wording from the plain-unrecognized-key error), not a silent no-op.
    - Processes reading a batch-overridable param never receive the whole batch YAML or a merged config map — only the individual resolved scalar(s) they consume, as `val()` inputs. This is deliberate: passing a whole file/map would make Nextflow's `-resume` cache key sensitive to every key in it, so an unrelated change in one batch's YAML would bust the cache for every process that batch feeds. `INPUT` itself no longer takes the raw YAML file as a process input for this same reason — it takes the resolved scalars and rebuilds a minimal YAML from them inside the process script.

---

## PR / commit workflow

No CONTRIBUTING.md exists. Based on git history:

- Branch off `main`, name branches descriptively (no enforced pattern observed).
- PR titles match commit message style: lowercase verb + `(#N)`.
- Squash-merge or merge commits both appear in history.
- Before merging: run `uv run pytest tests/unit`, `uv run ruff check .`, `uv run black --check .`.
- The only automated check on `main` is docs deployment — **tests do not run in CI**.

---

## Documentation maintenance

- Any change to CLI flags/Hydra config fields, Nextflow processes/workflow
  structure, or module responsibilities **must** update the relevant page(s)
  under `docs/` in the same change — not deferred to a follow-up. Start from
  `docs/architecture.md` (pipeline DAG/stages), `docs/nextflow.md` (process/param
  reference), and the relevant `docs/cli/<module>.md` + `docs/api/<module>.md`
  pair.
- Any new source file requires a file-level docstring (Python) or top `//`
  comment block (`.nf`) in the same change, except `__init__.py` — see the
  existing files for the convention.
- If a change makes existing documentation inaccurate, fix or remove the stale
  content in that same change — do not leave it "for later."
- `README.md` stays a thin pointer (overview + quick start + docs link). If a
  change would require README content beyond that scope (e.g. describing a new
  module's behavior), put that content in `docs/` instead, with at most a
  one-line mention added to the README's stage-flow summary.

---

## Safety / do-not-touch list

| Path | Reason |
|------|--------|
| `site/` | Generated MkDocs output, gitignored — not tracked in `main`. CI builds it and publishes it directly to the `gh-pages` branch. Build it locally with `mkdocs build`; editing it directly is pointless, it's regenerated every build. |
| `uv.lock` | Auto-managed by uv. Edit `pyproject.toml` deps instead, then run `uv sync`. |
| `.venv/` | Managed by uv. Never manually install packages into it. |
| `<pipeline_dir>/work/` | Nextflow task working directories — created at runtime, contains intermediate data. Delete only via `nextflow clean`. |
| Any `*.parquet` under `<pipeline_dir>/` | Pipeline output data. Do not modify or commit experiment output files. |
| `nextflow.config` | Ships default `params` and profile stubs for all users — only update if a param's default or a profile template changes. Do not add personal cluster paths; those belong in a user-supplied `-c your.config`. |
| `.github/workflows/docs.yml` | Deploys docs to GitHub Pages. Changes here affect live documentation for all users. |
