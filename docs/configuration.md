# Configuration

This page is the authoritative reference for pointing the pipeline at data and
tuning it: the `pipeline_dir` layout, every `nextflow.config` parameter, how a
batch's YAML config can override those parameters for just that batch, and
how named **global channels** control which batches contribute to the
pipeline's global (cross-batch) processes. See [Nextflow Workflow](nextflow.md)
for the process DAG itself.

## Pipeline directory layout

Every pipeline run is rooted at a single directory, `--pipeline_dir`:

```
<pipeline_dir>/
  configs/                       # mandatory -- one *.yaml per batch
    batch1.yaml
    batch2.yaml
    ...
  input/                         # INPUT output -- one *.parquet per batch, always
  qc_filter/<batch_stem>/
  normalization/{cells,normalizers}/
  wtvwt_batchwise/<batch_stem>/
  ovwt_batchwise/<batch_stem>/
  ovwt_batchwise_barcode_filtered/<batch_stem>/
  ovwt_cellscores_batchwise/<batch_stem>/
  check_barcodes/<batch_stem>/
  barcode_blocklist/<batch_stem>/
  feature_select_batchwise/<batch_stem>/
  global/
    <channel>/
      qc_filter_cells/<batch_stem>.parquet        # per-channel staged copy
      normalization_cells/<batch_stem>.parquet    # per-channel staged copy
      batchvsbatch/{pre,post}/results.parquet
      anova/anova.parquet
      anova_blocklist/anova_blocklist.parquet
      ovwt_global/{results.parquet,models.pkl}
      batch_correction/{fit,cells,anova}/...
      feature_select/{aggregate.parquet,blocklist.parquet}
```

`configs/` is **mandatory** -- every batch is declared by a YAML config file
there; there is no mode where the pipeline scans a directory of pre-staged
parquet files directly. `INPUT` (see
[CLI Reference: Input](cli/input.md)) always runs once per config file,
producing that batch's `input/<batch_stem>.parquet`.

`global/<channel>/` only exists for channels actually listed in
`--global_channels` (see [Global channels](#global-channels) below); by
default (`--global_channels` unset) no `global/` directory is produced at
all. This includes `ANOVA`/`ANOVA_BLOCKLIST`/`BATCH_CORRECT_FIT`/
`BATCH_CORRECT_TRANSFORM`/`ANOVA_BATCH_CORRECTED` -- unlike every other stage
on this page, that entire chain is a purely per-channel feature with no
pipeline-wide (ungated) counterpart, so it produces no output at all unless
at least one channel is active.

## Parameters

Defaults live in `nextflow.config` at the repo root.

### Pipeline selection

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--pipeline_mode` | `"fisseq"` | Which workflow to run: `"fisseq"` or `"ovwt"`. |
| `--pipeline_dir` | `null` (**required**) | Root directory. Must contain `configs/`, a directory of per-batch YAML config files -- see [Pipeline directory layout](#pipeline-directory-layout). |

### Branch toggles

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--global_channels` | `null` | List of named channels to run `OVWT_GLOBAL`, `BATCHVSBATCH`, `ANOVA` (both calls), `BATCH_CORRECT_FIT`/`BATCH_CORRECT_TRANSFORM`, and `GLOBAL_FEATURE_SELECT` for -- once per channel, scoped to only the batches whose YAML `global_channel` key names that channel. `null` or `[]` (the default): no global processes run at all. See [Global channels](#global-channels). |
| `--run_feature_selection` | `true` | Whether to run the feature-selection branch (batchwise + global) at all. |
| `--run_ovwt` | `true` | `FisseqPipeline` only: run `OVWT_BATCHWISE_UNFILTERED` (the normal/unfiltered per-batch OvWT pass) for that batch. Setting `false` also disables `run_single_cell_scores`/`run_check_barcodes`/`run_barcode_filtered_ovwt` for that batch regardless of their own settings, since all three consume this pass's output. No effect in `OvwtPipeline` (its entire purpose is this pass, so it always runs there). |
| `--run_single_cell_scores` | `false` | `FisseqPipeline` only: run `OVWT_CELLSCORES_BATCHWISE` per batch after `OVWT_BATCHWISE_UNFILTERED`. Always on in `OvwtPipeline`. Forced on if `--run_check_barcodes true`. |
| `--run_check_barcodes` | `false` | Run `CHECK_BARCODES` (per-batch pairwise Tukey HSD of single-cell scores across each variant's barcodes). Implies `--run_single_cell_scores true`. |
| `--run_barcode_filtered_ovwt` | `true` | Run `OVWT_BATCHWISE_BARCODE_FILTERED` (per-batch OvWT filtered against `barcode_blocklist/<batch>/barcode_blocklist.parquet`). Only takes effect when `--run_check_barcodes true` is also set (default `false`) -- does NOT force it on, so the default pipeline output is unaffected. |
| `--run_wtvwt` | `true` | `FisseqPipeline` only: run `WTVWT_BATCHWISE` (per-batch, wildtype-only pairwise barcode classification) for that batch. Independent of `--run_ovwt` and every other gate. |

### INPUT stage tunables

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--feature_allowlist_file` | `null` | `INPUT`: optional path to a glob-pattern feature allowlist file. |
| `--feature_blocklist_file` | `null` | `INPUT`: optional path to a glob-pattern feature blocklist file. |
| `--csv_schema_scan_rows` | `100` | `INPUT`: rows scanned from each CSV `input_paths` source to infer column dtypes (polars `scan_csv`'s `infer_schema_length`). `null` scans every row. No effect on parquet sources. |

These mirror `INPUT`'s per-batch YAML `config_path` schema (see
[CLI Reference: Input](cli/input.md#config_path-yaml-schema)) and are
overridable per batch exactly like every other parameter here — see
[Per-batch parameter overrides](#per-batch-parameter-overrides).

### QC filtering (`QC_FILTER`)

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--barcode_count_threshold` | `10` | Minimum cells per barcode (QC filter). |
| `--variant_barcode_count_threshold` | `4` | Minimum distinct barcodes per variant (QC filter). |
| `--edit_distance_threshold` | `1` | Maximum allowed edit distance (QC filter). |
| `--qc_n_variants` | `null` | Optional: restricts `qc_variant_downsample_classes` to at most this many distinct variants, before QC thresholding. `null` disables it. |
| `--qc_variant_downsample_classes` | `['Single Missense']` | Classes eligible for the `qc_n_variants` restriction. |
| `--qc_variant_downsample_mode` | `'top'` | `'top'` keeps the highest-cell-count variants; `'random'` keeps a seeded random sample. |
| `--qc_downsample_amounts` | `null` | Optional single float `(0, 1]`/int, or list of them: QC-filter pseudo-variant downsampling drawn from cells that already passed QC — a float keeps that fraction per variant, an int keeps that many cells (skipping variants with fewer). `null` disables it. A genuine multi-element *list* is only settable via a batch YAML override or by editing the Groovy list literal in `nextflow.config` directly — a bare `--qc_downsample_amounts` CLI flag only supports a single scalar. |
| `--qc_downsample_classes` | `['Synonymous', 'Single Missense']` | Classes eligible for `qc_downsample_amounts` pseudo-variant generation. |
| `--qc_downsample_seed` | `0` | Seed for the deterministic downsample selection, shared by `qc_downsample_amounts` and `qc_variant_downsample_mode="random"`. |

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
| `--max_cells_per_barcode_wt` | `null` | Optional cap on cells per wildtype barcode; any wildtype barcode exceeding this is randomly downsampled to exactly this count. `null` disables the cap. |
| `--max_cells_per_barcode_variant` | `null` | Optional cap on cells per non-wildtype barcode, analogous to `--max_cells_per_barcode_wt`. `null` disables the cap. |

### Wildtype-vs-wildtype pairwise barcode classification (`WTVWT_BATCHWISE`)

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--wtvwt_min_cells_per_barcode` | `100` | Minimum wildtype cells a barcode must have to be included in pairwise classification. |
| `--wtvwt_max_barcodes` | `null` | Optional: after `--wtvwt_min_cells_per_barcode` filtering, caps the number of wildtype barcodes profiled to at most this many. `null` disables it. |
| `--wtvwt_barcode_downsample_mode` | `'top'` | `'top'` keeps the highest-cell-count barcodes; `'random'` keeps a seeded random sample (using `random_state`). |

### Feature selection (bootstrap + aggregation + correlation)

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--feature_select_types` | `["mean", "median", "MAD", "std", "KS", "QQ", "AUROC"]` | Aggregators used in feature selection (the default subset of `aggregate.py`'s aggregators; `signedKS` is also available but not enabled by default). |
| `--feature_select_bootstrap_reps` | `10` | Number of pseudo-replicate bootstrap splits for feature selection. |
| `--feature_select_downsample_wt` | `null` | Optional wildtype downsample for `AGGREGATE_HALF`/`AGGREGATE_FEATURE_TYPE`: a float `(0, 1)` keeps that fraction of control rows, an int keeps that many, `null` disables it. `AGGREGATE_HALF` seeds each `(bootstrap_idx, half_num)` independently so every pseudo-replicate half draws a different WT subsample. See [CLI Reference: aggregate](cli/aggregate.md#python-m-fisseq_data_pipelineaggregatefeaturetype-config-fields). |
| `--feature_select_min_correlation` | `0.5` | Minimum median Pearson `r` required for a feature to pass `BLOCKLIST`. |
| `--global_feature_select_min_batches_ok` | `null` | `GLOBAL_FEATURE_SELECT` only: minimum number of a global channel's member batches that must mark a feature ok (in their own `FINALIZE_FEATURE_SELECT_BATCHWISE`-chain blocklist) for it to be globally ok. `null` (the default) requires unanimity -- ok in every member batch that reports on it. Pipeline-wide only, no per-batch meaning. |

### Single-cell scoring & barcode QC (`OVWT_CELLSCORES_BATCHWISE` / `CHECK_BARCODES`)

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `--single_cell_scores_split` | `"test"` | Which `OVWT_BATCHWISE` split to score: `"test"` or `"train"`. Any other value fails fast with a clear error. |
| `--barcode_check_min_cells` | `10` | `CHECK_BARCODES`: minimum cells required per barcode (within a variant) to include it in the comparison. |
| `--barcode_check_alpha` | `0.05` | `CHECK_BARCODES`: family-wise significance level for Tukey HSD; a barcode pair is flagged when its adjusted p-value is below this. |

## Global channels

By default, the pipeline's global (cross-batch) processes —
`BATCHVSBATCH`, `OVWT_GLOBAL`, `ANOVA` (both calls),
`BATCH_CORRECT_FIT`/`BATCH_CORRECT_TRANSFORM`, and `GLOBAL_FEATURE_SELECT` —
**do not run at all**. To run them, tag batches into named channels and list
which channels should actually run:

- **`global_channel`** — an optional key in a batch's YAML config, either a
  bare string or a list of strings, naming which channel(s) that batch
  belongs to. A batch that omits this key never contributes to any global
  run.
- **`--global_channels`** — a pipeline-wide list parameter (default `null`)
  naming which of those channels actually run. Each name in this list gets
  its own full `BATCHVSBATCH`/`OVWT_GLOBAL`/`ANOVA`/
  `BATCH_CORRECT_FIT`+`TRANSFORM`/`GLOBAL_FEATURE_SELECT` run, scoped to
  only the batches whose `global_channel` list contains that name,
  published under `<pipeline_dir>/global/<channel>/` (see
  [Pipeline directory layout](#pipeline-directory-layout)).

A batch can belong to multiple channels (via a list `global_channel`), in
which case it contributes to each of those channels' global runs
independently — the global processes are not deduplicated or merged across
channels. This includes `BATCH_CORRECT_TRANSFORM`: a batch in two channels is
batch-corrected once per channel, each using that channel's own fit (over
only that channel's member batches), producing two independently-corrected
copies of the batch's cells, one under each channel's own subtree.

`GLOBAL_FEATURE_SELECT` additionally requires a member batch to have its own
`run_feature_selection` enabled — it reads that batch's
`feature_select_batchwise/<batch_stem>/{aggregates,blocklist.parquet}`
directly, so a batch with `run_feature_selection: false` contributes to
`BATCHVSBATCH`/`OVWT_GLOBAL`/`ANOVA`/`BATCH_CORRECT_FIT`+`TRANSFORM` for its
channel(s) but not to `GLOBAL_FEATURE_SELECT`.

### Worked example

Two batches, two channels, with `batch2` contributing to both:

```yaml
# configs/batch1.yaml
input_paths: [/data/batch1.parquet]
global_channel: siteA
```

```yaml
# configs/batch2.yaml
input_paths: [/data/batch2.parquet]
global_channel: [siteA, siteB]
```

```yaml
# configs/batch3.yaml
input_paths: [/data/batch3.parquet]
global_channel: siteB
```

Running with `--global_channels '["siteA", "siteB"]'` produces:

```
<pipeline_dir>/global/siteA/...   # batch1 + batch2
<pipeline_dir>/global/siteB/...   # batch2 + batch3
```

A fourth batch with no `global_channel` key would be fully processed by every
batchwise stage (`QC_FILTER`, `NORMALIZE`, `OVWT_BATCHWISE`, ...) but would
never appear in either `global/siteA/` or `global/siteB/` — including the
`ANOVA`/`ANOVA_BLOCKLIST`/`BATCH_CORRECT_FIT`/`TRANSFORM`/
`ANOVA_BATCH_CORRECTED` chain, which now runs exclusively inside those
per-channel subtrees.

!!! note "Passing a list on the command line"
    A bare `--global_channels siteA,siteB` CLI flag does **not** produce a
    Groovy list — Nextflow's CLI parser treats it as the single string
    `"siteA,siteB"`. To pass a genuine list, either set
    `params.global_channels = ['siteA', 'siteB']` in a `-c your.config` file,
    or pass `-params-file params.json` with `{"global_channels": ["siteA", "siteB"]}`.
    This is the same limitation `--qc_downsample_amounts` and
    `--feature_select_types` already have for list-valued overrides.

### `global_channel` is validated, not consumed, by `INPUT`

`global_channel` lives in the same batch YAML file as `input_paths` (see
[CLI Reference: Input](cli/input.md#config_path-yaml-schema)), but it has no
effect on `python -m fisseq_data_pipeline.input` itself — it is read and
validated by the Nextflow workflow layer (`lib/BatchParams.groovy`) at
workflow-construction time, before `INPUT` ever runs.

### `OvwtPipeline` accepts but ignores `global_channel`

`OvwtPipeline` (`--pipeline_mode ovwt`) has no global processes at all, so
`global_channel`/`--global_channels` have no effect there. A batch YAML's
`global_channel` key is still validated the same way for consistency, but it
is simply never read.

## Per-batch parameter overrides

Any batch YAML in `<pipeline_dir>/configs/` may set additional keys beyond
`input_paths` to override that batch's own value for a `nextflow.config`
parameter — without affecting any other batch. This is resolved once per
batch, in Groovy, at workflow-construction time, by
[`lib/BatchParams.groovy`](https://github.com/Lilferrit/fisseq-data-pipeline/blob/main/lib/BatchParams.groovy)'s
`resolve()` function: it merges that batch's YAML on top of the pipeline-wide
defaults, validates every key, and returns the merged result plus a list of
which keys were actually overridden. Both `workflows/fisseq.nf` and
`workflows/ovwt.nf` call this once per batch YAML file and log each override
via `log.info` — e.g.:

```text
Batch 'batch3': overriding barcode_count_threshold (default=10) -> 3
```

### Three kinds of batch YAML keys

Every key a batch YAML can set falls into one of three buckets:

- **Batch-overridable** — most `nextflow.config` parameters (all the gating
  booleans, QC/OvWT/feature-selection thresholds, etc.). Setting one in a
  batch YAML overrides the pipeline-wide default for that batch only.
- **Pipeline-wide-only** — a `nextflow.config` parameter with no per-batch
  meaning, because it either gates/consumes ALL batches uniformly (e.g.
  `--anova_blocklist_pvalue_threshold`, `--global_channels`) or is a
  meta/bootstrap parameter resolved before any batch config can even be
  located (`--pipeline_mode`, `--pipeline_dir`). A batch YAML that tries to
  set one of these gets a clear rejection error, distinct from the
  unrecognized-key error below.
- **Batch-YAML-only** — a key with no `nextflow.config` default at all,
  meaningful only inside a batch YAML: `input_paths` (required — see below)
  and `global_channel` (optional — see [Global channels](#global-channels)).

### Not every parameter is batch-overridable

Several processes run once per active global channel (or not at all) rather
than per batch — `BATCHVSBATCH`, `OVWT_GLOBAL`, `ANOVA` (both calls),
`ANOVA_BLOCKLIST`, `BATCH_CORRECT_FIT`/`BATCH_CORRECT_TRANSFORM`, and
`GLOBAL_FEATURE_SELECT`. Params consumed only by those
processes — `--global_channels`, `--batchvsbatch_min_cells`,
`--batchvsbatch_min_batches`, `--anova_blocklist_pvalue_threshold`,
`--feature_select_types`, `--feature_select_bootstrap_reps`,
`--global_feature_select_min_batches_ok` — have no batch
to attach a per-batch override to, so they stay pipeline-wide-only.
(`--feature_select_types` and `--feature_select_bootstrap_reps` specifically
determine shared fan-out *cardinality* — how many feature-type/bootstrap
tasks exist at all — not a per-batch scalar value, so letting them vary per
batch would require a much larger restructuring than a simple value
override. `--global_feature_select_min_batches_ok` is inherently a
channel-level agreement threshold across batches, with no per-batch meaning
at all.)

Every other `nextflow.config` parameter — including the gating booleans
`--run_ovwt`, `--run_single_cell_scores`, `--run_check_barcodes`,
`--run_barcode_filtered_ovwt`, `--run_wtvwt`, and the *batchwise* effect of
`--run_feature_selection` — is genuinely per-batch overridable.
`--wtvwt_min_cells_per_barcode`, `--wtvwt_max_barcodes`, and
`--wtvwt_barcode_downsample_mode` are likewise per-batch overridable, same
bucket as `--ovwt_min_cells`. Each of these gates only a
per-batch-only process or chain, so `workflows/fisseq.nf` implements them as
a per-batch channel `.filter()` (via a `batchGates()` helper that also
encodes the "`run_check_barcodes` implies `run_single_cell_scores`" /
"`run_barcode_filtered_ovwt` only takes effect once `run_check_barcodes` is
true" rules) rather than a workflow-scope `if`. `--run_feature_selection`'s
*global* effect (`GLOBAL_FEATURE_SELECT`) runs once per active global channel
instead, gated on `params.run_feature_selection`, since that process has no
per-batch identity either — though it still only reads a member batch's
`feature_select_batchwise/` output if that batch's own resolved
`run_feature_selection` is true (see [Global channels](#global-channels)).

Parameters shared between a per-batch process and a global-only process
(`--ovwt_min_cells`, `--ovwt_downsample_wt`, `--feature_select_downsample_wt`,
`--feature_select_min_correlation`) are overridable per batch for their
batchwise consumer only (`OVWT_BATCHWISE`, `AGGREGATE_FEATURE_TYPE_BATCHWISE`
/ `AGGREGATE_HALF_BATCHWISE`, `BLOCKLIST_BATCHWISE`) — their global
counterpart (`OVWT_GLOBAL`, the `_GLOBAL` feature-selection processes)
always uses the plain pipeline-wide value directly, regardless of any
batch's override.

### `input_paths`: required, batch-YAML-only

`input_paths` (the list of raw cell-score file paths for a batch) is
**required in every batch YAML and has no `nextflow.config` default at
all** — it is intentionally excluded from the override-symmetry described
above. There is no sensible pipeline-wide default for a per-batch list of
data files: unlike a threshold or a boolean gate, a default `input_paths`
would look like something batches inherit, when in practice every batch
must supply its own. A batch YAML that omits it fails clearly at
config-resolution time rather than silently proceeding with no data.

### `global_channel`: optional, batch-YAML-only

`global_channel` (see [Global channels](#global-channels)) is the other
batch-YAML-only key, but unlike `input_paths` it's optional — a batch
omitting it simply never contributes to any global run, rather than failing.

### Why resolved scalars, not the whole config, are passed to processes

Every process that consumes a per-batch-overridable value receives it as an
individual `val()` input (e.g. `QC_FILTER` takes
`barcode_count_threshold`/`variant_barcode_count_threshold`/... as nine
separate `val()`s), never a whole config map or the raw YAML file. Nextflow's
`-resume` cache key for a task is derived from its declared inputs — if a
whole file or map were passed, changing *any* key in it (even one that
process doesn't consume) would bust the cache for every task built from it.
Passing only the specific scalars a process actually reads means an
unrelated key change in one batch's YAML — or a non-semantic edit to it —
does not invalidate that batch's tasks on the next `-resume` run. This is
also why `INPUT` doesn't receive the batch's YAML file directly: it takes
the resolved scalars as `val()` inputs and rebuilds a minimal YAML from them
inside the process script, so only those scalars — not the original file's
bytes — determine its cache key.
