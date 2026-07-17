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
| `fisseq` (default) | `FisseqPipeline` | `workflows/fisseq.nf` | Full end-to-end analysis (QC → normalize → batch-effect checks → OvWT → feature selection → batch correction → PERMANOVA) |
| `ovwt` | `OvwtPipeline` | `workflows/ovwt.nf` | OvWT classification only: `QC_FILTER` → `OVWT_BATCHWISE` → `OVWT_CELLSCORES_BATCHWISE`, no normalization, batch correction, or feature selection |

Both workflows validate that `--input_dir` is set and that `<input_dir>/input/`
exists and contains at least one `.parquet` file, then build a per-batch channel
from `<input_dir>/input/*.parquet` (one tuple per file, keyed by filename stem).

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
| `QC_FILTER` | `qc_filter.nf` | `fisseq-qc-filter` | per batch |
| `NORMALIZE` | `normalize.nf` | `fisseq-normalize` | per batch |
| `BATCHVSBATCH` (aliased `_PRE` / `_POST`) | `batchvsbatch.nf` | `fisseq-batch-vs-batch` | global, twice |
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
| `PERMANOVA` (aliased `_NORMALIZED` / `_BATCH_CORRECTED`) | `permanova.nf` | `fisseq-permanova` | global, twice |

"Aliased" processes are declared once and invoked twice in `workflows/fisseq.nf` via
`include { X as Y }` (Nextflow forbids calling one process twice under its own name
in a single workflow) — see [Architecture](architecture.md) for what each aliased
invocation does differently (which cells glob, which `publishDir` subpath).

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
| `--workflow` | `"fisseq"` | Which workflow to run: `"fisseq"` or `"ovwt"`. |
| `--bc_threshold` | `10` | Minimum cells per barcode (QC filter). |
| `--variant_bc_threshold` | `4` | Minimum distinct barcodes per variant (QC filter). |
| `--edit_distance_threshold` | `1` | Maximum allowed edit distance (QC filter). |
| `--bvb_min_cells` | `50` | Minimum total cells for a variant to be profiled in batch-vs-batch. |
| `--bvb_min_batches` | `2` | Minimum unique batches a variant must appear in for batch-vs-batch. |
| `--permanova_n_permutations` | `999` | Label permutations for the per-variant PERMANOVA p-value. |
| `--ovwt_min_cells` | `100` | Minimum cells required per variant for OvWT classification (overrides the Python CLI's own default of `250`). |
| `--downsample_wt` | `5000` | Wildtype downsample target for OvWT classification. |
| `--feature_types` | `["mean", "median", "MAD", "std", "KS", "QQ", "AUROC"]` | Aggregators used in feature selection (all 7 of `aggregate.py`'s aggregators). |
| `--bootstrap` | `10` | Number of pseudo-replicate bootstrap splits for feature selection. |
| `--global` | `true` | Whether to run `OVWT_GLOBAL` and the global feature-selection branch. |

## Profiles

`nextflow.config` also ships commented-out profile stubs for `venv`, `conda`,
`singularity`, and `sge` executors — see [Installation](installation.md#cluster-hpc)
for how to enable one via a user-supplied `-c your.config -profile <name>`.
