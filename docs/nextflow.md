# Nextflow Workflow

## Entry point

`main.nf` dispatches to one of two DSL2 workflows based on `--pipeline_mode`:

```groovy
workflow {
    if (params.pipeline_mode == "ovwt") {
        OvwtPipeline()
    } else {
        FisseqPipeline()
    }
}
```

| `--pipeline_mode` | Workflow | Definition | Use when |
| ------------- | -------- | ---------- | -------- |
| `fisseq` (default) | `FisseqPipeline` | `workflows/fisseq.nf` | Full end-to-end analysis (QC → normalize → batch-effect checks → OvWT → feature selection → batch correction → ANOVA) |
| `ovwt` | `OvwtPipeline` | `workflows/ovwt.nf` | OvWT classification only: `QC_FILTER` → `OVWT_BATCHWISE` → `OVWT_CELLSCORES_BATCHWISE` → `CHECK_BARCODES` (optional, `params.run_check_barcodes`), no normalization, batch correction, or feature selection |

Both workflows validate that `--pipeline_dir` is set and that
`<pipeline_dir>/configs/` exists and contains at least one `*.yaml` file --
every batch is declared by a YAML config there (see
[Configuration](configuration.md#pipeline-directory-layout)); there is no
mode where the pipeline scans a directory of pre-staged parquet files
directly.

## Quickstart

Run directly from GitHub — no cloning required:

```bash
nextflow run Lilferrit/fisseq-data-pipeline \
    -c your.config \
    --pipeline_dir /path/to/experiment
```

Use `-r` to pin to a branch or release tag; Nextflow caches the pulled revision in
`~/.nextflow/assets` (pass `-latest` to force a refresh). Or, from a local clone:

```bash
nextflow run . --pipeline_dir /path/to/experiment
```

Resume after an interruption using Nextflow's task-level caching:

```bash
nextflow run . --pipeline_dir /path/to/experiment -resume
```

## Processes

Every process wraps one `python -m fisseq_data_pipeline.<module>` invocation (see
[CLI Reference](cli/qcfilter.md) for each tool's config fields) and sets
`errorStrategy 'ignore'` so a single failed task doesn't abort the whole run.

| Process | `modules/local/*.nf` | Wraps | Cadence |
| ------- | --------------------- | ----- | ------- |
| `INPUT` | `input.nf` | `python -m fisseq_data_pipeline.input` | per config file, always (`<pipeline_dir>/configs/` is mandatory) |
| `QC_FILTER` | `qc_filter.nf` | `python -m fisseq_data_pipeline.qcfilter` | per batch |
| `NORMALIZE` | `normalize.nf` | `python -m fisseq_data_pipeline.normalize` | per batch |
| `STAGE_CHANNEL_CELLS` (aliased `_QC` / `_NORM`) | `stage_channel.nf` | (no Python wrapper — republishes a staged file) | per (active global channel × member batch); stages that batch's `QC_FILTER`/`NORMALIZE` output into `global/<channel>/{qc_filter_cells,normalization_cells}/` so the global processes below can glob a channel-scoped directory |
| `BATCHVSBATCH` (aliased `_PRE` / `_POST`) | `batchvsbatch.nf` | `python -m fisseq_data_pipeline.batchvsbatch` | per active global channel, twice (`params.global_channels`, default none run); `_PRE` unfiltered, `_POST` filtered against `ANOVA_BLOCKLIST` |
| `OVWT_BATCHWISE` (aliased `_UNFILTERED` / `_BARCODE_FILTERED`) | `ovwt_batchwise.nf` | `python -m fisseq_data_pipeline.ovwt` | per batch, twice (`FisseqPipeline`); `_UNFILTERED` has no dependency on `ANOVA_BLOCKLIST`/`BARCODE_BLOCKLIST` and is optional (`params.run_ovwt`), `_BARCODE_FILTERED` depends on that batch's `BARCODE_BLOCKLIST` output and is optional (`params.run_barcode_filtered_ovwt`). A third alias, `_FEATURE_FILTERED` (gated by `params.run_feature_filtered_ovwt`), was removed once `ANOVA_BLOCKLIST` became per-channel. |
| `OVWT_GLOBAL` | `ovwt_global.nf` | `python -m fisseq_data_pipeline.ovwt` | per active global channel (`params.global_channels`, default none run); always feature-filtered against `ANOVA_BLOCKLIST` |
| `WTVWT_BATCHWISE` | `wtvwt_batchwise.nf` | `python -m fisseq_data_pipeline.wtvwt` | per batch, optional (`params.run_wtvwt`); restricted to wildtype cells, trains one binary classifier per pair of wildtype barcodes; independent of the `ANOVA_BLOCKLIST`/OvWT chain |
| `OVWT_CELLSCORES_BATCHWISE` | `ovwt_cellscores_batchwise.nf` | `python -m fisseq_data_pipeline.ovwtcellscores` | per batch; optional in `FisseqPipeline` (`params.run_single_cell_scores`), always runs in `OvwtPipeline` |
| `CHECK_BARCODES` | `check_barcodes.nf` | `python -m fisseq_data_pipeline.checkbarcodes` | per batch, optional (`params.run_check_barcodes`, which also forces `run_single_cell_scores` on) |
| `BARCODE_BLOCKLIST` | `barcode_blocklist.nf` | `python -m fisseq_data_pipeline.barcodeblocklist` | per batch, requires both `params.run_check_barcodes` and `params.run_barcode_filtered_ovwt` true (the latter does not force the former on); consumes that batch's `CHECK_BARCODES` output; `FisseqPipeline` only |
| `AGGREGATE_FEATURE_TYPE` (`_BATCHWISE`) | `aggregate_feature_type.nf` | `python -m fisseq_data_pipeline.aggregatefeaturetype` | per batch × feature type |
| `GENERATE_SPLIT` (`_BATCHWISE`) | `generate_split.nf` | `python -m fisseq_data_pipeline.generatesplit` | per batch × bootstrap replicate |
| `AGGREGATE_HALF` (`_BATCHWISE`) | `aggregate_half.nf` | `python -m fisseq_data_pipeline.aggregatefeaturetype` (with `index_file`) | per batch × bootstrap × feature type × half |
| `CORRELATE_FEATURES` (`_BATCHWISE`) | `correlate_features.nf` | `python -m fisseq_data_pipeline.correlatefeatures` | per batch × bootstrap × feature type |
| `BLOCKLIST` (`_BATCHWISE`) | `blocklist.nf` | `python -m fisseq_data_pipeline.blocklist` | per batch × feature type — gathers all bootstrap replicates |
| `COMBINE_BLOCKLISTS` (`_BATCHWISE`) | `combine_blocklists.nf` | `python -m fisseq_data_pipeline.combineblocklists` | per batch — gathers all feature types |
| `FINALIZE_FEATURE_SELECT` (`_BATCHWISE`) | `finalize_feature_select.nf` | `python -m fisseq_data_pipeline.featureselect` | per batch |
| `GLOBAL_FEATURE_SELECT` | `global_feature_select.nf` | `python -m fisseq_data_pipeline.globalfeatureselect` | per active global channel (`params.global_channels`, default none run); reuses each member batch's `FINALIZE_FEATURE_SELECT_BATCHWISE`-chain aggregates/blocklist directly, no cell-level recomputation |
| `BATCH_CORRECT_FIT` | `batch_correct_fit.nf` | `python -m fisseq_data_pipeline.batchcorrect` | per active global channel, waits for that channel's `STAGE_CHANNEL_QC` batches |
| `BATCH_CORRECT_TRANSFORM` | `batch_correct_transform.nf` | `python -m fisseq_data_pipeline.batchcorrecttransform` | per (active global channel × member batch) pair |
| `ANOVA` (aliased `_NORMALIZED` / `_BATCH_CORRECTED`) | `anova.nf` | `python -m fisseq_data_pipeline.anova` | per active global channel, twice |
| `ANOVA_BLOCKLIST` | `anova_blocklist.nf` | `python -m fisseq_data_pipeline.anovablocklist` | per active global channel; consumes that channel's `ANOVA_NORMALIZED` output |

"Aliased" processes are declared once and invoked twice in `workflows/fisseq.nf` via
`include { X as Y }` (Nextflow forbids calling one process twice under its own name
in a single workflow) — see [Architecture](architecture.md) for what each aliased
invocation does differently (which cells glob, which `publishDir` subpath).

### `INPUT` stage

`INPUT` runs once per mandatory `*.yaml` config file in
`<pipeline_dir>/configs/` (`python -m fisseq_data_pipeline.input`, see
[CLI Reference: Input](cli/input.md)) and publishes its output into
`<pipeline_dir>/input/`. Every batch goes through this stage -- there is no
pre-staged-parquet mode. Like every other process, `INPUT` uses
`errorStrategy 'ignore'`: a failed conversion for one config file simply
drops that batch from the run, rather than aborting the whole pipeline.

`INPUT` does not receive the batch's hand-authored YAML file directly. Each
batch's YAML is parsed and merged with the pipeline-wide defaults once, at
workflow-construction time (see
[Configuration: Per-batch parameter overrides](configuration.md#per-batch-parameter-overrides)),
and `INPUT` instead receives the resolved `input_paths` / `feature_allowlist_file`
/ `feature_blocklist_file` / `csv_schema_scan_rows` values as individual
process inputs. The process
script rebuilds a minimal YAML from those values before invoking
`python -m fisseq_data_pipeline.input` — see
[Configuration: Why resolved scalars, not the whole config, are passed to processes](configuration.md#why-resolved-scalars-not-the-whole-config-are-passed-to-processes)
for why.

### Feature-selection channel wiring

The BATCHWISE feature-selection branch (`AGGREGATE_FEATURE_TYPE` →
`GENERATE_SPLIT` → `AGGREGATE_HALF` → `CORRELATE_FEATURES` → `BLOCKLIST` →
`COMBINE_BLOCKLISTS` → `FINALIZE_FEATURE_SELECT`, all per batch) is the most
complex part of the DAG — a bootstrap-correlation pipeline that determines,
per batch, which features are reproducible enough to keep. In
`workflows/fisseq.nf`:

- `feature_types_ch` (`Channel.fromList(params.feature_select_types)`) and `bootstrap_ch`
  (`Channel.of(1..params.feature_select_bootstrap_reps)`) are crossed via `.combine()` to fan out one
  task per (feature type, bootstrap replicate).
- Each `GENERATE_SPLIT` output is expanded into two per-half tuples via
  `.flatMap()`, then re-paired after `AGGREGATE_HALF` via
  `.groupTuple(by: [batch_stem, bootstrap_idx, feature_type])` before correlation.
- `BLOCKLIST`'s `.groupTuple(by: [batch_stem, feature_type])` is the pipeline's only
  cross-bootstrap synchronization point — it gathers all `params.feature_select_bootstrap_reps`
  correlation replicates for one feature type before computing a median-`r`
  threshold.
- This whole branch is per-batch gated on that batch's resolved
  `run_feature_selection`; it does not depend on `params.global_channels` at all.

`GLOBAL_FEATURE_SELECT` is a separate, much simpler branch, run once per
active channel in `params.global_channels` (default `null` = none active).
Unlike the batchwise branch, it does no bootstrap recomputation from cells: it
reuses each member batch's already-published `feature_select_batchwise/<batch>/`
aggregates and blocklist directly off `pipeline_dir`, so it needs no
Nextflow-level fan-out — the whole "join per-feature-type files, normalize to
that batch's synonymous baseline, take the cross-batch median, combine
blocklists by batch-agreement threshold, run pycytominer selection" sequence
runs as one Python invocation per channel. Batch → channel membership (only
batches with `run_feature_selection` enabled) is resolved once in Groovy from
`resolvedBatchConfigs`, the same way `resolvedBatchConfigs` itself is built —
see [Configuration: Global channels](configuration.md#global-channels) for the
parameter reference and output layout. `BATCHVSBATCH`, `OVWT_GLOBAL`, `ANOVA`
(both calls), and `BATCH_CORRECT_FIT`/`BATCH_CORRECT_TRANSFORM` all likewise
run once per active channel.

## Parameters, pipeline directory layout, and per-batch overrides

See [Configuration](configuration.md) for the full parameter reference, the
`<pipeline_dir>` directory layout (including the mandatory `configs/`
subdirectory and the `global/<channel>/` output tree), how a batch YAML can
override most parameters for just that batch, and how named global channels
scope `BATCHVSBATCH`/`OVWT_GLOBAL`/`ANOVA`/`BATCH_CORRECT_FIT`+`TRANSFORM`/the
global feature-selection branch.

## Profiles

`nextflow.config` also ships commented-out profile stubs for `venv`, `conda`,
`singularity`, and `sge` executors — see [Installation](installation.md#cluster-hpc)
for how to enable one via a user-supplied `-c your.config -profile <name>`.
