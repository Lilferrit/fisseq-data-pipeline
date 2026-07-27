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

Both workflows validate that `--input_dir` is set and that `<input_dir>/input/`
exists and contains at least one `.parquet` file, then build a per-batch channel
from `<input_dir>/input/*.parquet` (one tuple per file, keyed by filename stem).
If `--yaml_config_dir` is also set, that non-empty-check is relaxed (see
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
| `INPUT` | `input.nf` | `fisseq-input` | per config file, optional (`params.yaml_config_dir`) |
| `QC_FILTER` | `qc_filter.nf` | `fisseq-qc-filter` | per batch |
| `NORMALIZE` | `normalize.nf` | `fisseq-normalize` | per batch |
| `BATCHVSBATCH` (aliased `_PRE` / `_POST`) | `batchvsbatch.nf` | `fisseq-batch-vs-batch` | global, twice, optional (`params.run_global`); `_PRE` unfiltered, `_POST` filtered against `ANOVA_BLOCKLIST` |
| `OVWT_BATCHWISE` (aliased `_UNFILTERED` / `_FEATURE_FILTERED` / `_BARCODE_FILTERED`) | `ovwt_batchwise.nf` | `fisseq-ovwt` | per batch, three times (`FisseqPipeline`); `_UNFILTERED` has no dependency on `ANOVA_BLOCKLIST`/`BARCODE_BLOCKLIST`, `_FEATURE_FILTERED` depends on `ANOVA_BLOCKLIST` and is optional (`params.run_feature_filtered_ovwt`), `_BARCODE_FILTERED` depends on that batch's `BARCODE_BLOCKLIST` output and is optional (`params.run_barcode_filtered_ovwt`) |
| `OVWT_GLOBAL` | `ovwt_global.nf` | `fisseq-ovwt` | global, optional (`params.run_global`); always feature-filtered against `ANOVA_BLOCKLIST` |
| `OVWT_CELLSCORES_BATCHWISE` | `ovwt_cellscores_batchwise.nf` | `fisseq-ovwt-cell-scores` | per batch; optional in `FisseqPipeline` (`params.run_single_cell_scores`), always runs in `OvwtPipeline` |
| `CHECK_BARCODES` | `check_barcodes.nf` | `fisseq-check-barcodes` | per batch, optional (`params.run_check_barcodes`, which also forces `run_single_cell_scores` on) |
| `BARCODE_BLOCKLIST` | `barcode_blocklist.nf` | `fisseq-barcode-blocklist` | per batch, requires both `params.run_check_barcodes` and `params.run_barcode_filtered_ovwt` true (the latter does not force the former on); consumes that batch's `CHECK_BARCODES` output; `FisseqPipeline` only |
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
| `ANOVA_BLOCKLIST` | `anova_blocklist.nf` | `fisseq-anova-blocklist` | global, always runs; consumes `ANOVA_NORMALIZED`'s output |

"Aliased" processes are declared once and invoked twice in `workflows/fisseq.nf` via
`include { X as Y }` (Nextflow forbids calling one process twice under its own name
in a single workflow) — see [Architecture](architecture.md) for what each aliased
invocation does differently (which cells glob, which `publishDir` subpath).

### Optional `INPUT` stage

When `--yaml_config_dir` is set, `INPUT` runs once per `*.yaml` file found there
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
  by eagerly listing `yaml_config_dir`'s `*.yaml` basenames up front and filtering those
  names out of the glob channel before `.mix()`-ing in `INPUT`'s real output.
- **Precedence.** If a batch name exists both as a pre-staged file in `input/` and as
  a `yaml_config_dir/*.yaml`, the config-derived version silently wins — the pre-staged
  file is excluded from the glob channel by the same filter.

Like every other process, `INPUT` uses `errorStrategy 'ignore'`: a failed conversion
for one config file simply drops that batch from the run (it is excluded from both
the glob and the generated channel), rather than aborting the whole pipeline.

### Feature-selection channel wiring

The feature-selection branch (`AGGREGATE_FEATURE_TYPE` → `GENERATE_SPLIT` →
`AGGREGATE_HALF` → `CORRELATE_FEATURES` → `BLOCKLIST` → `COMBINE_BLOCKLISTS` →
`FINALIZE_FEATURE_SELECT`) is the most complex part of the DAG. In
`workflows/fisseq.nf`:

- `feature_types_ch` (`Channel.fromList(params.feature_select_types)`) and `bootstrap_ch`
  (`Channel.of(1..params.feature_select_bootstrap_reps)`) are crossed via `.combine()` to fan out one
  task per (feature type, bootstrap replicate).
- Each `GENERATE_SPLIT` output is expanded into two per-half tuples via
  `.flatMap()`, then re-paired after `AGGREGATE_HALF` via
  `.groupTuple(by: [batch_key, bootstrap_idx, feature_type])` before correlation.
- `BLOCKLIST`'s `.groupTuple(by: [batch_key, feature_type])` is the pipeline's only
  cross-bootstrap synchronization point — it gathers all `params.feature_select_bootstrap_reps`
  correlation replicates for one feature type before computing a median-`r`
  threshold.
- The batchwise branch always runs; the global branch (constant `global_key =
  "global"` standing in for `batch_stem`) is gated behind
  `params.run_global.toString().toBoolean()` (an explicit string-to-boolean parse,
  since Nextflow CLI overrides like `--run_global false` arrive as the truthy Groovy
  string `"false"`).

## Parameters

Defaults live in `nextflow.config` at the repo root:

### Pipeline selection

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--pipeline_mode` | `"fisseq"` | Which workflow to run: `"fisseq"` or `"ovwt"`. |
| `--input_dir` | `null` (**required**) | Root directory containing `input/*.parquet` batch files. |
| `--yaml_config_dir` | `null` | Optional directory of YAML configs; each generates one `input/*.parquet` via `INPUT`, merged with any pre-staged files in `input/`. See [Optional INPUT stage](#optional-input-stage). |

### Branch toggles

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--run_global` | `true` | Whether to run `OVWT_GLOBAL`, `BATCHVSBATCH`, and the global feature-selection branch. |
| `--run_feature_selection` | `true` | Whether to run the feature-selection branch (batchwise + global) at all. |
| `--run_feature_filtered_ovwt` | `true` | Run `OVWT_BATCHWISE_FEATURE_FILTERED` (per-batch OvWT filtered against `anova_blocklist/anova_blocklist.parquet`). Set `false` to skip it and keep only the unfiltered pass. (Renamed from `--run_filtered_ovwt`.) |
| `--run_single_cell_scores` | `false` | `FisseqPipeline` only: run `OVWT_CELLSCORES_BATCHWISE` per batch after `OVWT_BATCHWISE_UNFILTERED`. Always on in `OvwtPipeline`. Forced on if `--run_check_barcodes true`. |
| `--run_check_barcodes` | `false` | Run `CHECK_BARCODES` (per-batch pairwise Tukey HSD of single-cell scores across each variant's barcodes). Implies `--run_single_cell_scores true`. |
| `--run_barcode_filtered_ovwt` | `true` | Run `OVWT_BATCHWISE_BARCODE_FILTERED` (per-batch OvWT filtered against `barcode_blocklist/<batch>/barcode_blocklist.parquet`). Only takes effect when `--run_check_barcodes true` is also set (default `false`) -- does NOT force it on, so the default pipeline output is unaffected. |

### QC filtering (`QC_FILTER`)

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--barcode_count_threshold` | `10` | Minimum cells per barcode (QC filter). |
| `--variant_barcode_count_threshold` | `4` | Minimum distinct barcodes per variant (QC filter). |
| `--edit_distance_threshold` | `1` | Maximum allowed edit distance (QC filter). |
| `--qc_downsample_fraction` | `null` | Optional QC-filter pseudo-variant downsampling fraction `(0, 1]`; drawn from cells that already passed QC. `null` disables it. |
| `--qc_downsample_seed` | `0` | Seed for the deterministic downsample selection. |

### Batch-effect detection (`ANOVA_BLOCKLIST`)

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--anova_blocklist_pvalue_threshold` | `0.05` | A feature is blocked (`feature_ok = false`) when its `ANOVA_NORMALIZED` p-value is strictly less than this threshold (a statistically significant batch effect was detected). |

### Barcode-level blocklist (`BARCODE_BLOCKLIST`)

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--barcode_blocklist_pvalue_threshold` | `0.05` | A barcode is blocked (`barcode_ok = false`) when the median of its `CHECK_BARCODES` `p_adj` values is strictly less than this threshold. |

### Batch-vs-batch comparison (`BATCHVSBATCH`)

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--batchvsbatch_min_cells` | `50` | Minimum total cells for a variant to be profiled in batch-vs-batch. |
| `--batchvsbatch_min_batches` | `2` | Minimum unique batches a variant must appear in for batch-vs-batch. |

### One-vs-wildtype classification (`OVWT_GLOBAL` / `OVWT_BATCHWISE`)

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--ovwt_min_cells` | `100` | Minimum cells required per variant for OvWT classification (overrides the Python CLI's own default of `250`). |
| `--ovwt_downsample_wt` | `5000` | Wildtype downsample target for OvWT classification. |

### Feature selection (bootstrap + aggregation + correlation)

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--feature_select_types` | `["mean", "median", "MAD", "std", "KS", "QQ", "AUROC"]` | Aggregators used in feature selection (all 7 of `aggregate.py`'s aggregators). |
| `--feature_select_bootstrap_reps` | `10` | Number of pseudo-replicate bootstrap splits for feature selection. |
| `--feature_select_downsample_wt` | `null` | Optional wildtype downsample for `AGGREGATE_HALF`/`AGGREGATE_FEATURE_TYPE`: a float `(0, 1)` keeps that fraction of control rows, an int keeps that many, `null` disables it. `AGGREGATE_HALF` seeds each `(bootstrap_idx, half_num)` independently so every pseudo-replicate half draws a different WT subsample. See [CLI Reference: aggregate](cli/aggregate.md#fisseq-aggregate-feature-type-config-fields). |
| `--feature_select_min_correlation` | `0.5` | Minimum median Pearson `r` required for a feature to pass `BLOCKLIST`. |

### Single-cell scoring & barcode QC (`OVWT_CELLSCORES_BATCHWISE` / `CHECK_BARCODES`)

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--single_cell_scores_split` | `"test"` | Which `OVWT_BATCHWISE` split to score: `"test"` or `"train"`. Any other value fails fast with a clear error. |
| `--barcode_check_min_cells` | `10` | `CHECK_BARCODES`: minimum cells required per barcode (within a variant) to include it in the comparison. |
| `--barcode_check_alpha` | `0.05` | `CHECK_BARCODES`: family-wise significance level for Tukey HSD; a barcode pair is flagged when its adjusted p-value is below this. |

## Profiles

`nextflow.config` also ships commented-out profile stubs for `venv`, `conda`,
`singularity`, and `sge` executors — see [Installation](installation.md#cluster-hpc)
for how to enable one via a user-supplied `-c your.config -profile <name>`.
