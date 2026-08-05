# Architecture

## Pipeline DAG

The pipeline is orchestrated by Nextflow (`main.nf` → `workflows/fisseq.nf`, the
default `FisseqPipeline`; see [Nextflow Workflow](nextflow.md) for the lighter
`OvwtPipeline` alternative). Each Nextflow process shells out to a Python CLI tool
documented in the [CLI Reference](cli/qcfilter.md).

```text
configs/*.yaml  (mandatory, one file per batch — variant selection spec)
     │
     ▼
   INPUT       (per config, always runs)
     │
     ▼
input/*.parquet  (one file per batch, CellProfiler morphological features + barcode annotations)
     │
     ▼
QC_FILTER        (per batch)   ← edit distance, barcode count, variant barcode count
     │
     ├──► BATCHVSBATCH (pre, unfiltered)  (per active global channel in params.global_channels;
     │           default null = none run — see Configuration: Global channels)
     ├──► BATCH_CORRECT_FIT/TRANSFORM     (per active global channel)
     │           └──► ANOVA (batch-corrected)  (per active global channel)
     ▼
NORMALIZE        (per batch)   ← z-score fit on WT control cells
     │
     ├──► WTVWT_BATCHWISE  (per batch; skipped if params.run_wtvwt = false; wildtype
     │           cells only, one binary classifier per pair of wildtype barcodes)
     │
     ├──► ANOVA (normalized)  (per active global channel)
     │           │
     │           ▼
     │      ANOVA_BLOCKLIST  (per active global channel — marks feature_ok from ANOVA p-values)
     │           │
     │           ├──► BATCHVSBATCH (post, filtered)   (per active global channel)
     │           └──► OVWT_GLOBAL (filtered)            (per active global channel)
     │
     ├──► OVWT_BATCHWISE (unfiltered)     (per batch — no dependency on ANOVA_BLOCKLIST)
     │           └──► OVWT_CELLSCORES_BATCHWISE  (per batch; optional, params.run_single_cell_scores)
     │                     └──► CHECK_BARCODES  (per batch; optional, params.run_check_barcodes,
     │                               which also forces run_single_cell_scores on)
     │                               └──► BARCODE_BLOCKLIST  (per batch; requires both
     │                                         params.run_check_barcodes and
     │                                         params.run_barcode_filtered_ovwt true --
     │                                         the latter does NOT force the former on)
     │                                         └──► OVWT_BATCHWISE (barcode-filtered)  (per batch)
     └──► Feature selection, batchwise (per batch; always runs unless
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
                                       reuses member batches' aggregates/blocklist above --
                                       no cell-level recomputation)
```

A batch belonging to no active global channel is skipped by the entire
`BATCHVSBATCH`/`ANOVA`/`ANOVA_BLOCKLIST`/`BATCH_CORRECT_FIT`/`TRANSFORM`/
`OVWT_GLOBAL`/`GLOBAL_FEATURE_SELECT` chain (still fully processed batchwise
otherwise); a batch in multiple channels runs through that whole chain once
per channel, independently, each publishing under its own channel's subtree.

`ANOVA` and `BATCHVSBATCH` are each a single parameterized Nextflow process
invoked twice via `include { X as Y }` aliasing (a process cannot be called twice
under its own name in one workflow), each alias additionally invoked once per
active global channel; `OVWT_BATCHWISE` is likewise aliased, in
`FisseqPipeline`, into an unfiltered and a barcode-filtered invocation:

- `BATCHVSBATCH_PRE` runs on that channel's QC-filtered cells
  (`global/<channel>/qc_filter_cells/*.parquet`), unfiltered;
  `BATCHVSBATCH_POST` on that channel's normalized cells
  (`global/<channel>/normalization_cells/*.parquet`), filtered against that
  channel's own `ANOVA_BLOCKLIST`.
- `ANOVA_NORMALIZED` runs on that channel's normalized cells,
  `ANOVA_BATCH_CORRECTED` on that channel's batch-corrected cells
  (`global/<channel>/batch_correction/cells/*.parquet`).
- `OVWT_BATCHWISE_UNFILTERED` (published under `ovwt_batchwise/`) has no
  dependency on `ANOVA_BLOCKLIST` or `BARCODE_BLOCKLIST` and keeps the
  pipeline's original per-batch, no-wait-for-all-batches behavior.
  A third alias, `OVWT_BATCHWISE_FEATURE_FILTERED` (published under
  `ovwt_batchwise_feature_filtered/`, gated by `params.run_feature_filtered_ovwt`),
  used to depend on `ANOVA_BLOCKLIST` directly, but was removed once
  `ANOVA_BLOCKLIST` became per-channel: a per-batch process broadcasting one
  of several per-channel blocklists onto a batch in 0 or 2+ channels had no
  single well-defined blocklist to use. `BATCHVSBATCH_POST`/`OVWT_GLOBAL`,
  both already per-channel, don't have this problem.
  `OVWT_BATCHWISE_BARCODE_FILTERED` (published under
  `ovwt_batchwise_barcode_filtered/`) additionally depends on that batch's own
  `BARCODE_BLOCKLIST` output, so it runs after that batch's
  `CHECK_BARCODES`/`BARCODE_BLOCKLIST` complete. It only runs when both
  `params.run_check_barcodes` (default `false`) and
  `params.run_barcode_filtered_ovwt` (default `true`) are true --
  `run_barcode_filtered_ovwt` deliberately does not force `run_check_barcodes`
  on, so the default pipeline output is unaffected by its own default.
  `OVWT_GLOBAL` has no unfiltered counterpart (always feature-filtered)
  and no barcode-filtered counterpart (`BARCODE_BLOCKLIST` runs per batch, not
  globally).

`OVWT_CELLSCORES_BATCHWISE` scores `OVWT_BATCHWISE_UNFILTERED`'s models
against the `params.single_cell_scores_split` (`"test"` or `"train"`,
default `"test"`) split — gated by `params.run_single_cell_scores` (default
`false`) in `FisseqPipeline`, but always on in `OvwtPipeline`. Downstream,
`CHECK_BARCODES` runs a per-variant pairwise Tukey HSD across barcodes (see
[Check Barcodes](cli/checkbarcodes.md)), gated by `params.run_check_barcodes`
(default `false`), which also forces `single_cell_scores` on. Further
downstream (`FisseqPipeline` only), `BARCODE_BLOCKLIST` (see
[Barcode Block-list](cli/barcodeblocklist.md)) aggregates each barcode's
median `p_adj` from `CHECK_BARCODES`' output and feeds
`OVWT_BATCHWISE_BARCODE_FILTERED`; requires both `params.run_check_barcodes`
(default `false`) and `params.run_barcode_filtered_ovwt` (default `true`) to
be true -- the latter does not force the former on, so the default pipeline
output is unaffected by `run_barcode_filtered_ovwt`'s own default.

`ANOVA_BLOCKLIST` derives a feature block-list from `ANOVA_NORMALIZED`'s
p-values (not `ANOVA_BATCH_CORRECTED` — OvWT and batch-vs-batch score
normalized cells, never batch-corrected ones). A feature is blocked
(`feature_ok = false`) when its ANOVA `p_value` is strictly less than
`params.anova_blocklist_pvalue_threshold` (default `0.05`), i.e. when a statistically
significant batch effect was detected. It runs once per active global
channel — not gated by `params.run_feature_selection` — since
`BATCHVSBATCH_POST` and `OVWT_GLOBAL` both need their own channel's copy.

`BARCODE_BLOCKLIST` derives a barcode block-list from that batch's own
`CHECK_BARCODES` output (per batch, unlike the global `ANOVA_BLOCKLIST`). A
barcode is blocked (`barcode_ok = false`) when the median of its `p_adj`
values (pooled across both the `barcode` and `comparison_barcode` columns) is
strictly less than `params.barcode_blocklist_pvalue_threshold` (default
`0.05`).

`WTVWT_BATCHWISE` (see [Wildtype vs. Wildtype](cli/wtvwt.md)) is a single,
non-aliased per-batch process that restricts to wildtype-labeled cells and
trains one binary XGBoost classifier per pair of wildtype barcodes. It has no
feature-filtered/barcode-filtered variants and no global counterpart, and
depends only on `NORMALIZE`'s output (not `ANOVA_BLOCKLIST`). Gated by
`params.run_wtvwt` (default `true`), independent of every other gate.

Global processes (`BATCHVSBATCH`, `OVWT_GLOBAL`, `GLOBAL_FEATURE_SELECT`,
`ANOVA`, `ANOVA_BLOCKLIST`, `BATCH_CORRECT_FIT`) read published output
files from disk via glob patterns (or, for `ANOVA_BLOCKLIST`, consume a real
Nextflow channel output) after all upstream per-batch processes finish, rather
than consuming Nextflow channel outputs directly in the general case.
All of them —`BATCHVSBATCH`, `OVWT_GLOBAL`, `GLOBAL_FEATURE_SELECT`, `ANOVA`
(both calls), and `BATCH_CORRECT_FIT`/`BATCH_CORRECT_TRANSFORM` — run once
per named channel listed in `params.global_channels` (default `null` —
none run at all), each scoped to only the batches whose YAML `global_channel`
key names that channel — see [Configuration: Global channels](configuration.md#global-channels).
With no active channels, none of this runs at all — including the entire
`ANOVA`/`ANOVA_BLOCKLIST`/`BATCH_CORRECT_FIT`/`TRANSFORM`/
`ANOVA_BATCH_CORRECTED` chain, which has no pipeline-wide (ungated)
counterpart.
`GLOBAL_FEATURE_SELECT` additionally only reads a member batch's
`feature_select_batchwise/` output if that batch's own `run_feature_selection`
is enabled.

## Stages

| Stage | Python module | Nextflow process(es) | Produces |
| ----- | -------------- | --------------------- | -------- |
| Input generation (optional) | `input.py` | `INPUT` | `input/<name>.parquet`, from a YAML variant-selection spec |
| QC filtering | `qcfilter.py` | `QC_FILTER` | `filtered_cells.parquet` (optionally augmented with downsampled pseudo-variant rows for QC/calibration, drawn from QC-surviving cells), `barcode_counts.parquet`, `variants_per_barcode.parquet` |
| Batch-effect check (pre) | `batchvsbatch.py` | `BATCHVSBATCH` (pre) | `results.parquet` |
| Normalization | `normalize.py` | `NORMALIZE` | normalized cells + `normalizer.parquet` |
| Batch-effect check (post) | `batchvsbatch.py` | `BATCHVSBATCH` (post) | `results.parquet` |
| One-vs-WT classification | `ovwt.py` | `OVWT_BATCHWISE` (unfiltered + feature-filtered + barcode-filtered), `OVWT_GLOBAL` (feature-filtered) | `results.parquet`, `models.pkl` |
| Wildtype-vs-wildtype barcode classification | `wtvwt.py` | `WTVWT_BATCHWISE` | `results.parquet`, `models.pkl` |
| OvWT cell scoring | `ovwtcellscores.py` | `OVWT_CELLSCORES_BATCHWISE` | `cell_scores.parquet` |
| Barcode-outlier check | `checkbarcodes.py` | `CHECK_BARCODES` | `results.parquet` (per-variant pairwise Tukey HSD across barcodes) |
| Barcode block-list | `barcodeblocklist.py` | `BARCODE_BLOCKLIST` | `barcode_blocklist.parquet` (per batch) |
| Feature selection (batchwise) | `aggregate.py`, `featureselect.py` | `AGGREGATE_FEATURE_TYPE`, `GENERATE_SPLIT`, `AGGREGATE_HALF`, `CORRELATE_FEATURES`, `BLOCKLIST`, `COMBINE_BLOCKLISTS`, `FINALIZE_FEATURE_SELECT` | `output.parquet` (final per-batch per-variant aggregate) |
| Feature selection (global) | `globalfeatureselect.py` | `GLOBAL_FEATURE_SELECT` | `aggregate.parquet` (cross-batch median aggregate, pycytominer-selected), `blocklist.parquet` (combined global blocklist) |
| Batch correction | `batchcorrect.py` | `BATCH_CORRECT_FIT`, `BATCH_CORRECT_TRANSFORM` | `stats_vb.parquet`, `centroids.parquet`, corrected cells |
| Batch-effect assessment | `anova.py` | `ANOVA` (normalized and batch-corrected) | `anova.parquet` |
| ANOVA feature block-list | `anovablocklist.py` | `ANOVA_BLOCKLIST` | `anova_blocklist.parquet` |

See the [CLI Reference](cli/qcfilter.md) pages for each module's config fields and
the [API Reference](api/qcfilter.md) pages for full function documentation.

## Key abstractions

**`src/fisseq_data_pipeline/config/`** — Hydra structured config hierarchy:

```text
AppConfig
  └── InputConfig (adds input_file)
        └── LabeledInputConfig (adds label_column, default "meta_aa_changes")
              └── tool-specific configs (e.g. NormalizeConfig, AggregateConfig, OvwtConfig)
```

Every entry point uses `@hydra.main(...)` with its config class registered in the
Hydra `ConfigStore`. See [API Reference: config](api/config.md).

**`Normalizer`** (`normalize.py`) — fits per-feature z-score statistics (mean, std)
on a LazyFrame and applies them. Stats are persisted to Parquet (not pickle) and
reloaded with `Normalizer.load(path)`. Zero-variance features produce `null` after
normalization. Used by both `normalize.py` (fit on WT cells) and `aggregate.py`
(fit on synonymous-variant aggregates).

**`BaseAggregator`** (`aggregate.py`) — abstract base for 8 concrete aggregation
strategies: mean, median, MAD, std, KS, signedKS, QQ, AUROC. There is no multi-aggregator
wrapper — combining feature types happens in Nextflow: `aggregate.feature_type_main`
runs once per `params.feature_select_types` entry, and `features.main` (the final
feature-selection stage) joins the per-feature-type outputs on the label column.

**`BatchCorrector`** (`batchcorrect.py`) — fits per-(variant, batch) statistics and
per-variant centroids across all batches, then applies a two-pass rescale (to the
variant's own centroid, then to the wildtype centroid) to correct each batch's cells.

**`utils/xgbparams.py`** — shared XGBoost infrastructure imported by `ovwt.py`,
`ovwtcellscores.py`, and `batchvsbatch.py`: `XGBoostParams`/`XGBoostConfig`
dataclasses, `get_feature_cols` (CellProfiler column detection), `get_dmatrix` /
`get_dmatrix_multiclass` (DMatrix builders), and `split_indices_stratified`
(80/10/10 stratified split).

**`load_batches`** (`utils/batches.py`) — accepts a path or glob pattern, reads
matching Parquet files, tags each with `meta_batch` (filename stem or parent
directory name), returns a concatenated `pl.LazyFrame` plus an output stem string.
Used by nearly every entry point whose `input_file` accepts a glob.

## Output layout

All outputs land under `<pipeline_dir>`, alongside the mandatory `configs/`
and `input/` folders:

```text
<pipeline_dir>/
  configs/*.yaml               # mandatory, one per batch
  qc_filter/<batch>/
    filtered_cells.parquet
    barcode_counts.parquet
    variants_per_barcode.parquet
  normalization/
    cells/<batch>.parquet
    normalizers/<batch>.normalizer.parquet
  ovwt_batchwise/<batch>/     # unfiltered — full feature set
    results.parquet
    models.pkl
    test_index.parquet        # columns: row_idx, origin_file
    train_index.parquet       # columns: row_idx, origin_file
  ovwt_batchwise_barcode_filtered/<batch>/   # filtered against that batch's
                               # barcode_blocklist/<batch>/barcode_blocklist.parquet;
                               # requires params.run_check_barcodes AND
                               # params.run_barcode_filtered_ovwt both true
    results.parquet
    models.pkl
    test_index.parquet
    train_index.parquet
  wtvwt_batchwise/<batch>/    # wildtype cells only; optional: params.run_wtvwt
    results.parquet           # columns: barcode_a, barcode_b, train/val/test_auroc,
                               # train/val/test_accuracy, n_cells_a, n_cells_b
    models.pkl                # dict[(barcode_a, barcode_b) -> xgb.Booster]
  ovwt_cellscores_batchwise/<batch>/   # optional: params.run_single_cell_scores (always on in OvwtPipeline)
    cell_scores.parquet
  check_barcodes/<batch>/     # optional: params.run_check_barcodes (implies run_single_cell_scores)
    results.parquet           # columns: variant, barcode, group_mean, comparison_barcode,
                               # comparison_group_mean, mean_diff, p_adj, reject
  barcode_blocklist/<batch>/  # optional: params.run_check_barcodes AND params.run_barcode_filtered_ovwt both true
    barcode_blocklist.parquet # columns: barcode, p_adj (median), barcode_ok
  feature_select_batchwise/<batch>/
    aggregates/<feature_type>.parquet                                     # stage 1
    splits/bootstrap_<n>/half{1,2}.parquet                                # stage 2a
    half_aggregates/bootstrap_<n>/<feature_type>/half{1,2}.parquet        # stage 2b
    correlations/<feature_type>/bootstrap_<n>.parquet                     # stage 2c
    blocklists/<feature_type>.parquet                                     # stage 2d
    blocklist.parquet                                                    # stage 3 (combined)
    output.parquet                                                       # stage 4 (final)
  global/<channel>/              # one subtree per channel in params.global_channels
                                  # (default null -- none); only that channel's member
                                  # batches (per-batch YAML global_channel) contribute.
                                  # This is the ONLY place ANOVA/ANOVA_BLOCKLIST/
                                  # BATCH_CORRECT_* output lands.
    qc_filter_cells/<batch>.parquet        # staged copy for BATCHVSBATCH_PRE/BATCH_CORRECT_FIT
    normalization_cells/<batch>.parquet    # staged copy for BATCHVSBATCH_POST/OVWT_GLOBAL/ANOVA
    batchvsbatch/
      pre/results.parquet        # pre batch correction (QC-filtered cells), unfiltered
      post/results.parquet       # post batch correction (normalized cells), filtered against this channel's anova_blocklist
    anova/
      anova.parquet               # from this channel's normalized cells
    anova_blocklist/
      anova_blocklist.parquet     # derived from this channel's anova/anova.parquet
    ovwt_global/                 # always filtered against this channel's own anova_blocklist/anova_blocklist.parquet
      results.parquet
      models.pkl
    batch_correction/
      fit/stats_vb.parquet        # fit over this channel's own QC-filtered batches only
      fit/centroids.parquet
      cells/<batch>.parquet       # one task per (channel, batch) pair
      anova/anova.parquet         # from this channel's batch-corrected cells
    feature_select/
      aggregate.parquet          # cross-batch median aggregate, pycytominer-selected
      blocklist.parquet          # combined blocklist (agreement threshold across member
                                  # batches' feature_select_batchwise/<batch>/blocklist.parquet)
```

`global/<channel>/feature_select/` reuses each member batch's already-published
`feature_select_batchwise/<batch>/{aggregates,blocklist.parquet}` directly — it
does not glob `normalization_cells/` and has no `splits/`, `half_aggregates/`,
`correlations/`, or `blocklists/` subtree of its own (those only exist per
batch, under `feature_select_batchwise/<batch>/`, above).
