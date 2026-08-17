# Aggregate

Cell-level aggregation is implemented as two Hydra entry points across two modules:

- **`python -m fisseq_data_pipeline.aggregate`** — standalone: aggregates
  cell-level data to one row per variant, then normalizes the result to a
  synonymous-variant baseline and attaches per-variant metadata. Not wired into
  the Nextflow pipeline directly.
- **`python -m fisseq_data_pipeline.aggregatefeaturetype`** (Nextflow process
  `AGGREGATE_FEATURE_TYPE`) — a leaner version used by the feature-selection
  branch: runs a single aggregator, writes only
  `[label_column] + <stat columns>`, with no normalizer, metadata join, or impact
  score. Imports `aggregate()` and `downsample_control()` from
  `fisseq_data_pipeline.aggregate`. The WT-null bootstrap branch's per-replicate
  aggregation (`WT_NULL_AGGREGATE`) is a separate entry point,
  `fisseq_data_pipeline.wtnullaggregate` — see [Feature Selection](features.md#1-python-m-fisseq_data_pipelinewtnullaggregate-wt_null_aggregate).

Both accept `input_file` as a glob pattern (via `load_batches`) or a concrete
single-file path.

Both also support an optional **per-barcode aggregation mode**
(`per_barcode`/`barcode_column`): instead of pooling all of a variant's cells
directly, each aggregator's statistic is computed per (variant, barcode)
first, then reduced to one value per variant by taking the median across that
variant's barcodes. Reference-based aggregators (`KS`, `signedKS`, `QQ`,
`AUROC`, ...) still compare every (variant, barcode) group against the SAME
full control pool — the reference frame is built once per feature, not per
barcode, so this mode changes only the variant-side grouping, not what each
group is compared against.

## Aggregators

Eight strategies are available via the `aggregator` field — there is **no**
`"multi"`/combined option; combining feature types happens in Nextflow by running
`AGGREGATE_FEATURE_TYPE` once per `params.feature_select_types` entry. Note that
`signedKS` is not included in the default `params.feature_select_types` list (see
[Parameters](../configuration.md#parameters)) — it must be opted into explicitly.

| Value | Description |
| ----- | ----------- |
| `mean` | Per-variant feature mean |
| `median` | Per-variant feature median |
| `MAD` | Per-variant median absolute deviation |
| `std` | Per-variant standard deviation |
| `KS` | Kolmogorov-Smirnov statistic vs. WT/control distribution |
| `signedKS` | Same magnitude as `KS`, but signed by which empirical CDF is larger at the maximizing point: positive when the variant group's CDF exceeds the reference's there (group skews lower), negative when the reference's CDF is larger (group skews higher). |
| `QQ` | Q-Q Pearson correlation vs. WT/control distribution |
| `AUROC` | AUROC vs. WT/control distribution. Directional: `0.5` means identical distributions, `1.0` means the variant is consistently higher than the reference, `0.0` means consistently lower (not symmetrized to `[0.5, 1]`). |

## `python -m fisseq_data_pipeline.aggregate` config fields

Extends `LabeledInputConfig` (adds `input_file`, `label_column`) plus the
[common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Glob pattern or path to cell-level data. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `aggregator` | **required** | One of the eight aggregators above. |
| `save_normalizer` | `true` | Write the synonymous-baseline normalizer. |
| `block_list_file` | `null` | Parquet with `feature` and `feature_ok` columns; blocked features are skipped. |
| `compute_impact_score` | `true` | Append an impact score column derived from variant classification. |
| `per_barcode` | `false` | Compute each statistic per (variant, barcode) first, then reduce to one value per variant by median across barcodes, instead of pooling all of a variant's cells directly. |
| `barcode_column` | `"meta_barcode"` | Column identifying the barcode a cell was measured from. Only consulted when `per_barcode` is `true`. |

**Output**: glob input → `{output_root}.output.parquet` or `{output_dir}/output.parquet`;
single-file input → `{output_root}.{stem}.{ext}` or `{output_dir}/{filename}`. Plus
`normalizer.parquet` when `save_normalizer=true`.

```bash
uv run python -m fisseq_data_pipeline.aggregate \
    output_dir=./out \
    'input_file=data/batches/*.parquet' \
    aggregator=KS
```

## `python -m fisseq_data_pipeline.aggregatefeaturetype` config fields

Extends `LabeledInputConfig` plus the [common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Glob pattern or path to cell-level data. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `aggregator` | **required** | One of the eight aggregators above. |
| `downsample_wt` | `null` | Optional downsample of control (wildtype) rows before aggregation. A float in `(0, 1)` keeps that fraction; an int keeps that many. `null` disables downsampling. |
| `seed` | `0` | Random seed for the `downsample_wt` draw. Ignored when `downsample_wt` is `null`. |
| `per_barcode` | `false` | Compute each statistic per (variant, barcode) first, then reduce to one value per variant by median across barcodes. Must match `WT_NULL_AGGREGATE`'s setting for the same batch, or the WT-null reproducibility check stops being apples-to-apples. |
| `barcode_column` | `"meta_barcode"` | Column identifying the barcode a cell was measured from. Only consulted when `per_barcode` is `true`. |

**Output**: glob input → `{output_root}.output.parquet` or `{output_dir}/output.parquet`;
single-file input → `{output_root}.{stem}.parquet` or `{output_dir}/{stem}.parquet`.

```bash
uv run python -m fisseq_data_pipeline.aggregatefeaturetype \
    output_dir=./out \
    input_file=data/normalized.parquet \
    aggregator=mean \
    downsample_wt=0.5 \
    seed=1 \
    per_barcode=true \
    barcode_column=meta_barcode
```

In the Nextflow pipeline, `downsample_wt`/`seed` are driven by `params.feature_select_downsample_wt`
(see [Parameters](../configuration.md#parameters)); `AGGREGATE_FEATURE_TYPE` (the full,
un-split aggregation) uses a fixed seed, since it runs once per batch/feature type with no
repeated per-instance identity to vary by — unlike `WT_NULL_AGGREGATE`, which derives a
distinct seed per `(bootstrap_idx, half_num)` so each split half of each bootstrap replicate
draws an independent wildtype subsample (see
[Feature Selection](features.md#1-python-m-fisseq_data_pipelinewtnullaggregate-wt_null_aggregate)).
`per_barcode`/`barcode_column` are driven by
`params.feature_select_per_barcode`/`params.feature_select_barcode_column` and are passed
identically to both `AGGREGATE_FEATURE_TYPE` and `WT_NULL_AGGREGATE` for a given batch (resolved
from the same per-batch config), so both always use the same mode.

See [API Reference: aggregate](../api/aggregate.md) for full function
documentation, including the `BaseAggregator` class hierarchy.
