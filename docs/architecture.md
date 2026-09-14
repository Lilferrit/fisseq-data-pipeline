# Architecture

The FISSEQ Data Pipeline is a Nextflow + Python workflow for processing
single-cell CellProfiler morphological profiling data from FISSEQ
(Fluorescence In-Situ Sequencing) experiments. Each cell carries a genetic
variant label; the pipeline measures how each variant's cell population differs
from wildtype (WT) controls using morphological features.

## Pipeline DAG

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
      ├──► OVWT_BATCHWISE  (per experiment; gated by run_ovwt)
      │         │  k-fold CV one-vs-wildtype scoring
      │         └──► GLOBAL_OVWT  (once per active global channel)
      │                  per-experiment synonymous z-score,
      │                  then cross-experiment median
      │
      └──► Feature selection, batchwise (gated by run_feature_selection):
             AGGREGATE_FEATURE_TYPE      (per feature type)          ─┐
             GENERATE_SPLIT              (per bootstrap replicate)    │
               └─► AGGREGATE_HALF        (per bootstrap × type × half)│
                     └─► CORRELATE_FEATURES  (per bootstrap × type)   │
                           └─► BLOCKLIST     (gathers all bootstraps — │
                                              the one sync point)      │
                                 └─► COMBINE_BLOCKLISTS (all types) ──┘
                                       └─► FINALIZE_FEATURE_SELECT
                                             │
                                             ▼
                                      GLOBAL_FEATURE_SELECT
                                      (once per active global channel)
```

There is a single pipeline mode. `main.nf` includes one workflow,
`workflows/fisseq.nf`, and runs it.

## Experiments

Every experiment is declared as a map in the `experiments:` list in
`params.yaml`. There is no `<pipeline_dir>/configs/` directory, and no
per-experiment override of arbitrary pipeline parameters — an entry may set only
`batch_stem`, `input_paths`, `global_channel`, and the three INPUT-stage fields.
Anything else is rejected with an error naming the key. See
[Configuration](configuration.md).

## Global channels

An experiment's `global_channel` key (string or list of strings) names which
channel(s) it belongs to; `params.global_channels` (pipeline-wide, default
`null`) lists which of those channels actually run. Each active channel gets its
own `GLOBAL_OVWT` and `GLOBAL_FEATURE_SELECT` run, scoped to only that channel's
member experiments, publishing under `global/<channel>/`.

With `global_channels` unset (the default), neither global stage runs at all and
the pipeline is purely per-experiment. An experiment belonging to no active
channel is still processed batchwise as normal; one in several channels
contributes to each independently.

Both global stages read published output rather than staged cells:
`GLOBAL_FEATURE_SELECT` reads each member experiment's
`feature_select_batchwise/<batch_stem>/{aggregates,blocklist.parquet}` directly
off `pipeline_dir`, while `GLOBAL_OVWT` receives its members' `results.parquet`
files as real staged paths (not a glob string, so `-resume` invalidates
correctly).

## Reproducibility

`params.random_seed` is the only seed in the pipeline. Every stochastic step
reads it: QC pseudo-variant downsampling, the feature-selection bootstrap splits
and their wildtype subsampling, OvWT's fold shuffle / inner calibration split /
XGBoost `seed`, PCA's solver, and UMAP's fit.

Stages that must differ from one another derive a fixed offset rather than
owning a seed — `GENERATE_SPLIT` uses `random_seed + bootstrap_idx`, and
`AGGREGATE_HALF` uses `random_seed + bootstrap_idx * 2 + half_num`, so bootstrap
replicates still draw independent subsamples. `tests/unit/test_config.py` sweeps
every config class to keep a second seed from reappearing.

Reproducibility also depends on **row order** being stable, which is easy to
lose: polars inner joins are not order-preserving under multithreaded execution.
`QC_FILTER` therefore assigns `meta_cell_index` over the raw input order and
sorts on it before publishing. Without that, the same input yields the same rows
in a different order on every run, and every seeded step downstream silently
diverges despite a fixed seed.

## Two different control baselines

These are easy to confuse:

| Stage | Control population |
| ----- | ------------------ |
| `NORMALIZE` (cell level) | **Wildtype** cells (`meta_aa_changes = 'WT'`) |
| `FINALIZE_FEATURE_SELECT` / `GLOBAL_FEATURE_SELECT` (aggregate level) | **Synonymous** variants |
| `GLOBAL_OVWT` (score level) | **Synonymous** variants |

`aggregate.variant_classification` flags synonymous, untagged labels as
`meta_is_control = True`; `normalize.py` uses the WT SQL query instead.

Note that `OVWT_BATCHWISE` consumes `NORMALIZE`'s wildtype-normalized output.
The sibling `fisseq-embeddings-pipeline`, from which the OvWT implementation was
ported, z-scores its features against synonymous variants before training
instead. The difference is deliberate: the synonymous re-centering happens here
at the score level, in `GLOBAL_OVWT`.

## Variant tags

`aaChanges` values may carry a `:<tag>` suffix (e.g. `V123A:downsampled-half`).
The tag is stripped exactly once, in `qcfilter.py:filter_columns`:
`meta_aa_changes` is always the tag-stripped base label and `meta_variant_tag`
holds the tag (`null` when absent). Every stage after `QC_FILTER` therefore sees
clean, pooled variant labels. Do not re-strip or re-parse tags downstream — use
`meta_variant_tag` directly.

Two independent things can put a tag there: upstream raw data may arrive
pre-tagged, or `qcfilter.py`'s optional pseudo-variant downsampling
(`qc_downsample_amounts`) may generate one (`downsample-{amount}`) for the rows
it creates.

## Output layout

```
<pipeline_dir>/
  input/<batch_stem>.parquet
  qc_filter/<batch_stem>/
    filtered_cells.parquet
    barcode_counts.parquet
    variants_per_barcode.parquet
  normalization/
    cells/<batch_stem>.parquet
    normalizers/<batch_stem>.normalizer.parquet
  ovwt_batchwise/<batch_stem>/
    results.parquet          # per variant: auroc_pooled, auroc_median_barcode,
                             # meta_n_barcodes, meta_n_cells
    cell_scores.parquet      # per cell per variant scored against: meta_* +
                             # score + meta_variant_scored_against
    models.pkl               # dict[variant -> list[(Booster, calibrator|None)]]
  feature_select_batchwise/<batch_stem>/
    aggregates/<feature_type>.parquet
    splits/bootstrap_<n>/half{1,2}.parquet
    half_aggregates/bootstrap_<n>/<feature_type>/half{1,2}.parquet
    correlations/<feature_type>/bootstrap_<n>.parquet
    blocklists/<feature_type>.parquet
    blocklist.parquet
    output.parquet
  global/<channel>/          # one subtree per active global channel
    ovwt_distinguishability/global_scores.parquet
    feature_select/
      aggregate.parquet
      blocklist.parquet
```

## Column naming

| Pattern | Meaning |
| ------- | ------- |
| `meta_*` | Metadata (barcode, batch, labels, QC flags, scores, cell index) |
| `UPPERCASE_WITH_UNDERSCORE` | CellProfiler morphological feature columns |
| `tmp_*` | Ephemeral intermediates, dropped before output |

The `meta_` prefix is load-bearing: `FEATURE_SELECTOR` is defined as
`exclude("^meta_.*$")`, so any new non-`meta_` column is treated as a feature.

## Components

- `src/fisseq_data_pipeline/` — Python package, one module per pipeline step,
  each a Hydra entry point run as `python -m fisseq_data_pipeline.<module>`
- `modules/local/*.nf` — Nextflow process wrappers around those CLIs
- `workflows/fisseq.nf` — the DAG
- `main.nf` — entry point
- `params.yaml` — every parameter default
- `nextflow.config` — executor/profile/container settings only
- `Dockerfile` — the single image every process runs in
