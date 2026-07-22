# Aggregate

`aggregate.py` provides two Hydra entry points:

- **`fisseq-aggregate`** — standalone: aggregates cell-level data to one row per
  variant, then normalizes the result to a synonymous-variant baseline and
  attaches per-variant metadata. Not wired into the Nextflow pipeline directly.
- **`fisseq-aggregate-feature-type`** (Nextflow processes `AGGREGATE_FEATURE_TYPE`
  and `AGGREGATE_HALF`) — a leaner version used by the feature-selection branch:
  runs a single aggregator, writes only `[label_column] + <stat columns>`, with no
  normalizer, metadata join, or impact score.

Both accept `input_file` as a glob pattern (via `load_batches`) or a concrete
single-file path.

## Aggregators

Seven strategies are available via the `aggregator` field — there is **no**
`"multi"`/combined option; combining feature types happens in Nextflow by running
`AGGREGATE_FEATURE_TYPE` once per `params.feature_types` entry.

| Value | Description |
| ----- | ----------- |
| `mean` | Per-variant feature mean |
| `median` | Per-variant feature median |
| `MAD` | Per-variant median absolute deviation |
| `std` | Per-variant standard deviation |
| `KS` | Kolmogorov-Smirnov statistic vs. WT/control distribution |
| `QQ` | Q-Q Pearson correlation vs. WT/control distribution |
| `AUROC` | AUROC vs. WT/control distribution. Directional: `0.5` means identical distributions, `1.0` means the variant is consistently higher than the reference, `0.0` means consistently lower (not symmetrized to `[0.5, 1]`). |

## `fisseq-aggregate` config fields

Extends `LabeledInputConfig` (adds `input_file`, `label_column`) plus the
[common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Glob pattern or path to cell-level data. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `aggregator` | **required** | One of the seven aggregators above. |
| `save_normalizer` | `true` | Write the synonymous-baseline normalizer. |
| `block_list_file` | `null` | Parquet with `feature` and `feature_ok` columns; blocked features are skipped. |
| `compute_impact_score` | `true` | Append an impact score column derived from variant classification. |

**Output**: glob input → `{output_root}.output.parquet` or `{output_dir}/output.parquet`;
single-file input → `{output_root}.{stem}.{ext}` or `{output_dir}/{filename}`. Plus
`normalizer.parquet` when `save_normalizer=true`.

```bash
uv run fisseq-aggregate \
    output_dir=./out \
    'input_file=data/batches/*.parquet' \
    aggregator=KS
```

## `fisseq-aggregate-feature-type` config fields

Extends `LabeledInputConfig` plus the [common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Glob pattern or path to cell-level data. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `aggregator` | **required** | One of the seven aggregators above. |
| `index_file` | `null` | Optional path to a single-column row-index parquet (as written by `fisseq-generate-split`) restricting aggregation to a pseudo-replicate half. |
| `downsample_wt` | `null` | Optional downsample of control (wildtype) rows before aggregation. A float in `(0, 1)` keeps that fraction; an int keeps that many. `null` disables downsampling. |
| `seed` | `0` | Random seed for the `downsample_wt` draw. Ignored when `downsample_wt` is `null`. |

**Output**: glob input → `{output_root}.output.parquet` or `{output_dir}/output.parquet`;
single-file input → `{output_root}.{stem}.parquet` or `{output_dir}/{stem}.parquet`.

```bash
uv run fisseq-aggregate-feature-type \
    output_dir=./out \
    input_file=data/normalized.parquet \
    aggregator=mean \
    index_file=./half1.parquet \
    downsample_wt=0.5 \
    seed=1
```

In the Nextflow pipeline, `downsample_wt`/`seed` are driven by `params.aggregate_downsample_wt`
(see [Parameters](../nextflow.md#parameters)) — `AGGREGATE_HALF` derives a distinct seed per
`(bootstrap_idx, half_num)` so each pseudo-replicate half draws an independent wildtype
subsample, which is what lets the bootstrap comparison test feature reproducibility against
different WT samples rather than reusing one fixed sample everywhere. `AGGREGATE_FEATURE_TYPE`
(the full, un-split aggregation) uses a fixed seed, since it has no repeated per-instance
identity to vary by.

See [API Reference: aggregate](../api/aggregate.md) for full function
documentation, including the `BaseAggregator` class hierarchy.
