# Architecture

## Pipeline DAG

The pipeline is orchestrated by Nextflow (`main.nf` → `workflows/fisseq.nf`, the
default `FisseqPipeline`; see [Nextflow Workflow](nextflow.md) for the lighter
`OvwtPipeline` alternative). Each Nextflow process shells out to a Python CLI tool
documented in the [CLI Reference](cli/qcfilter.md).

```text
config/*.yaml  (optional, one file per batch — variant selection spec)
     │
     ▼
   INPUT       (per config, optional — gated by params.yaml_config_dir)
     │
     ▼
input/*.parquet  (one file per batch, CellProfiler morphological features + barcode annotations)
     │
     ▼
QC_FILTER        (per batch)   ← edit distance, barcode count, variant barcode count
     │
     ├──► BATCHVSBATCH (pre, unfiltered)  (global — waits for all QC_FILTER; skipped if params.run_global = false)
     ▼
NORMALIZE        (per batch)   ← z-score fit on WT control cells
     │
     ├──► WTVWT_BATCHWISE  (per batch; skipped if params.run_wtvwt = false; wildtype
     │           cells only, one binary classifier per pair of wildtype barcodes)
     │
     ├──► ANOVA (normalized)  (global — waits for all batches; always runs)
     │           │
     │           ▼
     │      ANOVA_BLOCKLIST  (global; always runs — marks feature_ok from ANOVA p-values)
     │           │
     │           ├──► BATCHVSBATCH (post, filtered)   (global; skipped if params.run_global = false)
     │           ├──► OVWT_BATCHWISE (feature-filtered) (per batch; skipped if params.run_feature_filtered_ovwt = false)
     │           └──► OVWT_GLOBAL (filtered)            (global; skipped if params.run_global = false)
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
     └──► Feature selection (batchwise always runs; global waits for all batches, skipped if params.run_global = false):
            AGGREGATE_FEATURE_TYPE      (per feature type)          ─┐
            GENERATE_SPLIT              (per bootstrap replicate)    │
              └─► AGGREGATE_HALF        (per bootstrap × feature type × half)
                    └─► CORRELATE_FEATURES (per bootstrap × feature type)
                          └─► BLOCKLIST  (gathers all bootstrap replicates per feature type — the one sync point)
                                └─► COMBINE_BLOCKLISTS (gathers all feature types) ┘
                                      └─► FINALIZE_FEATURE_SELECT (joins AGGREGATE_FEATURE_TYPE outputs + combined blocklist)

QC_FILTER ──► BATCH_CORRECT_FIT (global, waits for all QC_FILTER)
                    │
                    ▼
             BATCH_CORRECT_TRANSFORM (per batch)
                    │
                    ▼
             ANOVA (batch-corrected)  (always runs)
```

`ANOVA` and `BATCHVSBATCH` are each a single parameterized Nextflow process
invoked twice via `include { X as Y }` aliasing (a process cannot be called twice
under its own name in one workflow); `OVWT_BATCHWISE` is likewise aliased, in
`FisseqPipeline`, into an unfiltered, a feature-filtered, and a
barcode-filtered invocation:

- `BATCHVSBATCH_PRE` runs on QC-filtered cells (`qc_filter/*/filtered_cells.parquet`),
  unfiltered; `BATCHVSBATCH_POST` on normalized cells (`normalization/cells/*.parquet`),
  filtered against `ANOVA_BLOCKLIST`.
- `ANOVA_NORMALIZED` runs on normalized cells, `ANOVA_BATCH_CORRECTED` on
  batch-corrected cells (`batch_correction/cells/*.parquet`).
- `OVWT_BATCHWISE_UNFILTERED` (published under `ovwt_batchwise/`) has no
  dependency on `ANOVA_BLOCKLIST` or `BARCODE_BLOCKLIST` and keeps the
  pipeline's original per-batch, no-wait-for-all-batches behavior.
  `OVWT_BATCHWISE_FEATURE_FILTERED` (published under
  `ovwt_batchwise_feature_filtered/`, renamed from `ovwt_batchwise_filtered/`)
  additionally depends on `ANOVA_BLOCKLIST`, so it runs once all batches have
  normalized and `ANOVA_NORMALIZED`/`ANOVA_BLOCKLIST` have completed. It's
  optional, gated by `params.run_feature_filtered_ovwt` (default `true`,
  preserving the pipeline's original always-on behavior; renamed from
  `run_filtered_ovwt`). `OVWT_BATCHWISE_BARCODE_FILTERED` (published under
  `ovwt_batchwise_barcode_filtered/`) additionally depends on that batch's own
  `BARCODE_BLOCKLIST` output, so it runs after that batch's
  `CHECK_BARCODES`/`BARCODE_BLOCKLIST` complete. It only runs when both
  `params.run_check_barcodes` (default `false`) and
  `params.run_barcode_filtered_ovwt` (default `true`) are true --
  `run_barcode_filtered_ovwt` deliberately does not force `run_check_barcodes`
  on, so the default pipeline output is unaffected by its own default. The
  feature- and barcode-filtered variants are independent — one drops feature
  columns, the other drops cell rows — and either, both, or neither may be
  enabled. `OVWT_GLOBAL` has no unfiltered counterpart (always feature-filtered)
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
significant batch effect was detected. It always runs, unconditionally — not
gated by `params.run_global`/`params.run_feature_selection` — since
`OVWT_BATCHWISE_FEATURE_FILTERED` (when enabled), `BATCHVSBATCH_POST`, and
`OVWT_GLOBAL` all need it.

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

Global processes (`BATCHVSBATCH`, `OVWT_GLOBAL`, the `*_GLOBAL` feature-selection
branch, `ANOVA`, `ANOVA_BLOCKLIST`, `BATCH_CORRECT_FIT`) read published output
files from disk via glob patterns (or, for `ANOVA_BLOCKLIST`, consume a real
Nextflow channel output) after all upstream per-batch processes finish, rather
than consuming Nextflow channel outputs directly in the general case.
`params.run_global` (default `true`) gates `BATCHVSBATCH`, `OVWT_GLOBAL`, and the
`*_GLOBAL` feature-selection branch — `ANOVA`, `ANOVA_BLOCKLIST`,
`BATCH_CORRECT_FIT`/`BATCH_CORRECT_TRANSFORM` always run.

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
| Feature selection | `aggregate.py`, `features.py` | `AGGREGATE_FEATURE_TYPE`, `GENERATE_SPLIT`, `AGGREGATE_HALF`, `CORRELATE_FEATURES`, `BLOCKLIST`, `COMBINE_BLOCKLISTS`, `FINALIZE_FEATURE_SELECT` | `output.parquet` (final per-variant aggregate) |
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

**`BaseAggregator`** (`aggregate.py`) — abstract base for 7 concrete aggregation
strategies: mean, median, MAD, std, KS, QQ, AUROC. There is no multi-aggregator
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

All outputs land under `<input_dir>`, alongside the `input/` folder:

```text
<input_dir>/
  qc_filter/<batch>/
    filtered_cells.parquet
    barcode_counts.parquet
    variants_per_barcode.parquet
  normalization/
    cells/<batch>.parquet
    normalizers/<batch>.normalizer.parquet
  batchvsbatch/
    pre/results.parquet         # pre batch correction (QC-filtered cells), unfiltered
    post/results.parquet        # post batch correction (normalized cells), filtered against anova_blocklist
  ovwt_batchwise/<batch>/     # unfiltered — full feature set
    results.parquet
    models.pkl
    test_index.parquet        # columns: row_idx, origin_file
    train_index.parquet       # columns: row_idx, origin_file
  ovwt_batchwise_feature_filtered/<batch>/   # filtered against anova_blocklist/anova_blocklist.parquet
                               # (renamed from ovwt_batchwise_filtered/)
    results.parquet
    models.pkl
    test_index.parquet
    train_index.parquet
  ovwt_batchwise_barcode_filtered/<batch>/   # filtered against that batch's
                               # barcode_blocklist/<batch>/barcode_blocklist.parquet;
                               # requires params.run_check_barcodes AND
                               # params.run_barcode_filtered_ovwt both true
    results.parquet
    models.pkl
    test_index.parquet
    train_index.parquet
  ovwt_global/                # always filtered against anova_blocklist/anova_blocklist.parquet
    results.parquet
    models.pkl
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
  feature_select_global/
    (same layout, no <batch> nesting — only present if params.run_global = true)
  batch_correction/
    fit/stats_vb.parquet
    fit/centroids.parquet
    cells/<batch>.parquet
    anova/anova.parquet
  anova/
    anova.parquet                # from normalized cells
  anova_blocklist/
    anova_blocklist.parquet      # derived from anova/anova.parquet
```
