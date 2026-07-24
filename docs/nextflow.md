# Nextflow Workflow

## Entry point

`main.nf` dispatches to one of two DSL2 workflows based on `--workflow`:

```groovy
workflow {
    if (params.workflow == "ovwt") {
        OvwtPipeline()
    } else {
        FisseqPipeline()
    }
}
```

| `--workflow` | Workflow | Definition | Use when |
| ------------- | -------- | ---------- | -------- |
| `fisseq` (default) | `FisseqPipeline` | `workflows/fisseq.nf` | Full end-to-end analysis (QC → normalize → batch-effect checks → OvWT → feature selection → batch correction → ANOVA) |
| `ovwt` | `OvwtPipeline` | `workflows/ovwt.nf` | OvWT classification only: `QC_FILTER` → `OVWT_BATCHWISE` → `OVWT_CELLSCORES_BATCHWISE`, no normalization, batch correction, or feature selection |

Both workflows validate that `--input_dir` is set and that `<input_dir>/input/`
exists and contains at least one `.parquet` file, then build a per-batch channel
from `<input_dir>/input/*.parquet` (one tuple per file, keyed by filename stem).
If `--config_dir` is also set, that non-empty-check is relaxed (see
[Optional INPUT stage](#optional-input-stage) below) since `<input_dir>/input/`
may not exist yet on a first config-driven run.

## Quickstart

Run directly from GitHub — no cloning required:

```bash
nextflow run Lilferrit/fisseq-data-pipeline \
    -c your.config \
    --input_dir /path/to/experiment
```

Use `-r` to pin to a branch or release tag; Nextflow caches the pulled revision in
`~/.nextflow/assets` (pass `-latest` to force a refresh). Or, from a local clone:

```bash
nextflow run . --input_dir /path/to/experiment
```

Resume after an interruption using Nextflow's task-level caching:

```bash
nextflow run . --input_dir /path/to/experiment -resume
```

## Processes

Every process wraps one `fisseq-*` CLI command (see [CLI Reference](cli/qcfilter.md)
for each tool's config fields) and sets `errorStrategy 'ignore'` so a single failed
task doesn't abort the whole run.

| Process | `modules/local/*.nf` | Wraps | Cadence |
| ------- | --------------------- | ----- | ------- |
| `INPUT` | `input.nf` | `fisseq-input` | per config file, optional (`params.config_dir`) |
| `QC_FILTER` | `qc_filter.nf` | `fisseq-qc-filter` | per batch |
| `NORMALIZE` | `normalize.nf` | `fisseq-normalize` | per batch |
| `BATCHVSBATCH` (aliased `_PRE` / `_POST`) | `batchvsbatch.nf` | `fisseq-batch-vs-batch` | global, twice, optional (`params.global`) |
| `OVWT_BATCHWISE` | `ovwt_batchwise.nf` | `fisseq-ovwt` | per batch |
| `OVWT_GLOBAL` | `ovwt_global.nf` | `fisseq-ovwt` | global, optional (`params.global`) |
| `OVWT_CELLSCORES_BATCHWISE` | `ovwt_cellscores_batchwise.nf` | `fisseq-ovwt-cell-scores` | per batch |
| `AGGREGATE_FEATURE_TYPE` (aliased `_BATCHWISE` / `_GLOBAL`) | `aggregate_feature_type.nf` | `fisseq-aggregate-feature-type` | per (batch or global) × feature type |
| `GENERATE_SPLIT` (aliased) | `generate_split.nf` | `fisseq-generate-split` | per (batch or global) × bootstrap replicate |
| `AGGREGATE_HALF` (aliased) | `aggregate_half.nf` | `fisseq-aggregate-feature-type` (with `index_file`) | per (batch or global) × bootstrap × feature type × half |
| `CORRELATE_FEATURES` (aliased) | `correlate_features.nf` | `fisseq-correlate-features` | per (batch or global) × bootstrap × feature type |
| `BLOCKLIST` (aliased) | `blocklist.nf` | `fisseq-blocklist` | per (batch or global) × feature type — gathers all bootstrap replicates |
| `COMBINE_BLOCKLISTS` (aliased) | `combine_blocklists.nf` | `fisseq-combine-blocklists` | per (batch or global) — gathers all feature types |
| `FINALIZE_FEATURE_SELECT` (aliased) | `finalize_feature_select.nf` | `fisseq-feature-select` | per (batch or global) |
| `BATCH_CORRECT_FIT` | `batch_correct_fit.nf` | `fisseq-batch-correct-fit` | global, waits for all `QC_FILTER` |
| `BATCH_CORRECT_TRANSFORM` | `batch_correct_transform.nf` | `fisseq-batch-correct-transform` | per batch |
| `ANOVA` (aliased `_NORMALIZED` / `_BATCH_CORRECTED`) | `anova.nf` | `fisseq-anova` | global, twice, always runs |

"Aliased" processes are declared once and invoked twice in `workflows/fisseq.nf` via
`include { X as Y }` (Nextflow forbids calling one process twice under its own name
in a single workflow) — see [Architecture](architecture.md) for what each aliased
invocation does differently (which cells glob, which `publishDir` subpath).

### Optional `INPUT` stage

When `--config_dir` is set, `INPUT` runs once per `*.yaml` file found there
(`fisseq-input`, see [CLI Reference: Input](cli/input.md)) and publishes its output
into `<input_dir>/input/`, the same directory pre-staged batch files live in. Both
`workflows/fisseq.nf` and `workflows/ovwt.nf` merge this generated channel with the
pre-existing `Channel.fromPath("<input_dir>/input/*.parquet")` glob channel via
`.mix()`, so `QC_FILTER` sees one unified stream regardless of which code path
produced a given batch.

Two subtleties worth knowing:

- **Double-processing guard.** `Channel.fromPath(glob)` is evaluated once, at
  workflow-construction time — it does not wait for `INPUT` to finish. On a re-run
  where `<input_dir>/input/` already contains a file `INPUT` previously published
  there, the glob channel would match it independently of `INPUT`'s live output for
  this run, feeding the same batch into `QC_FILTER` twice. Both workflows avoid this
  by eagerly listing `config_dir`'s `*.yaml` basenames up front and filtering those
  names out of the glob channel before `.mix()`-ing in `INPUT`'s real output.
- **Precedence.** If a batch name exists both as a pre-staged file in `input/` and as
  a `config_dir/*.yaml`, the config-derived version silently wins — the pre-staged
  file is excluded from the glob channel by the same filter.

Like every other process, `INPUT` uses `errorStrategy 'ignore'`: a failed conversion
for one config file simply drops that batch from the run (it is excluded from both
the glob and the generated channel), rather than aborting the whole pipeline.

### Feature-selection channel wiring

The feature-selection branch (`AGGREGATE_FEATURE_TYPE` → `GENERATE_SPLIT` →
`AGGREGATE_HALF` → `CORRELATE_FEATURES` → `BLOCKLIST` → `COMBINE_BLOCKLISTS` →
`FINALIZE_FEATURE_SELECT`) is the most complex part of the DAG. In
`workflows/fisseq.nf`:

- `feature_types_ch` (`Channel.fromList(params.feature_types)`) and `bootstrap_ch`
  (`Channel.of(1..params.bootstrap)`) are crossed via `.combine()` to fan out one
  task per (feature type, bootstrap replicate).
- Each `GENERATE_SPLIT` output is expanded into two per-half tuples via
  `.flatMap()`, then re-paired after `AGGREGATE_HALF` via
  `.groupTuple(by: [batch_key, bootstrap_idx, feature_type])` before correlation.
- `BLOCKLIST`'s `.groupTuple(by: [batch_key, feature_type])` is the pipeline's only
  cross-bootstrap synchronization point — it gathers all `params.bootstrap`
  correlation replicates for one feature type before computing a median-`r`
  threshold.
- The batchwise branch always runs; the global branch (constant `global_key =
  "global"` standing in for `batch_stem`) is gated behind
  `params.global.toString().toBoolean()` (an explicit string-to-boolean parse,
  since Nextflow CLI overrides like `--global false` arrive as the truthy Groovy
  string `"false"`).

## Parameters

Defaults live in `nextflow.config` at the repo root:

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--input_dir` | `null` (**required**) | Root directory containing `input/*.parquet` batch files. |
| `--config_dir` | `null` | Optional directory of YAML configs; each generates one `input/*.parquet` via `INPUT`, merged with any pre-staged files in `input/`. See [Optional INPUT stage](#optional-input-stage). |
| `--workflow` | `"fisseq"` | Which workflow to run: `"fisseq"` or `"ovwt"`. |
| `--bc_threshold` | `10` | Minimum cells per barcode (QC filter). |
| `--variant_bc_threshold` | `4` | Minimum distinct barcodes per variant (QC filter). |
| `--edit_distance_threshold` | `1` | Maximum allowed edit distance (QC filter). |
| `--downsample_fraction` | `null` | Optional QC-filter pseudo-variant downsampling fraction `(0, 1]`; drawn from cells that already passed QC. `null` disables it. |
| `--downsample_seed` | `0` | Seed for the deterministic downsample selection. |
| `--bvb_min_cells` | `50` | Minimum total cells for a variant to be profiled in batch-vs-batch. |
| `--bvb_min_batches` | `2` | Minimum unique batches a variant must appear in for batch-vs-batch. |
| `--anova_feature_batch_size` | `null` | Optional int: ANOVA per-feature-chunk collect size, bounds peak memory when there are many features; `null` collects all features in one query. |
| `--ovwt_min_cells` | `100` | Minimum cells required per variant for OvWT classification (overrides the Python CLI's own default of `250`). |
| `--downsample_wt` | `5000` | Wildtype downsample target for OvWT classification. |
| `--aggregate_downsample_wt` | `null` | Optional wildtype downsample for `AGGREGATE_HALF`/`AGGREGATE_FEATURE_TYPE`: a float `(0, 1)` keeps that fraction of control rows, an int keeps that many, `null` disables it. `AGGREGATE_HALF` seeds each `(bootstrap_idx, half_num)` independently so every pseudo-replicate half draws a different WT subsample. See [CLI Reference: aggregate](cli/aggregate.md#fisseq-aggregate-feature-type-config-fields). |
| `--feature_types` | `["mean", "median", "MAD", "std", "KS", "QQ", "AUROC"]` | Aggregators used in feature selection (all 7 of `aggregate.py`'s aggregators). |
| `--bootstrap` | `10` | Number of pseudo-replicate bootstrap splits for feature selection. |
| `--global` | `true` | Whether to run `OVWT_GLOBAL` and the global feature-selection branch. |

## Profiles

`nextflow.config` also ships commented-out profile stubs for `venv`, `conda`,
`singularity`, and `sge` executors — see [Installation](installation.md#cluster-hpc)
for how to enable one via a user-supplied `-c your.config -profile <name>`.
