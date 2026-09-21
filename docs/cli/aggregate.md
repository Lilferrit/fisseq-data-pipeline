# Aggregate

Cell-level aggregation is implemented as two Hydra entry points across two modules:

- **`python -m fisseq_data_pipeline.aggregate`** — standalone: aggregates
  cell-level data to one row per variant, then normalizes the result to a
  synonymous-variant baseline and attaches per-variant metadata. Not wired into
  the Nextflow pipeline directly.
- **`python -m fisseq_data_pipeline.aggregatefeaturetype`** (Nextflow processes
  `AGGREGATE_FEATURE_TYPE` and `AGGREGATE_HALF`) — a leaner version used by the
  feature-selection branch: runs a single aggregator, writes only
  `[label_column] + <stat columns>`, with no normalizer, metadata join, or impact
  score. Imports `aggregate()` and `downsample_control()` from
  `fisseq_data_pipeline.aggregate`.

Both accept `input_file` as a glob pattern (via `load_batches`) or a concrete
single-file path.

## Aggregators

Eight strategies are available via the `aggregator` field — there is **no**
`"multi"`/combined option; combining feature types happens in Nextflow by running
`AGGREGATE_FEATURE_TYPE` once per `params.feature_select_types` entry. Note that
`signedKS` is not included in the default `params.feature_select_types` list (see
[Parameters](../configuration.md#parameter-reference)) — it must be opted into explicitly.

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
| `feature_chunk_size` | `32` | Feature columns aggregated per Polars query; `null` disables chunking. See [Feature chunking](#feature-chunking). |

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
| `index_file` | `null` | Optional path to a single-column row-index parquet (as written by `python -m fisseq_data_pipeline.generatesplit`) restricting aggregation to a pseudo-replicate half. |
| `downsample_wt` | `null` | Optional downsample of control (wildtype) rows before aggregation. A float in `(0, 1)` keeps that fraction; an int keeps that many. `null` disables downsampling. |
| `seed` | `0` | Random seed for the `downsample_wt` draw. Ignored when `downsample_wt` is `null`. |
| `feature_chunk_size` | `32` | Feature columns aggregated per Polars query; `null` disables chunking. Driven by `params.aggregate_feature_chunk_size`. See [Feature chunking](#feature-chunking). |

**Output**: glob input → `{output_root}.output.parquet` or `{output_dir}/output.parquet`;
single-file input → `{output_root}.{stem}.parquet` or `{output_dir}/{stem}.parquet`.

```bash
uv run python -m fisseq_data_pipeline.aggregatefeaturetype \
    output_dir=./out \
    input_file=data/normalized.parquet \
    aggregator=mean \
    index_file=./half1.parquet \
    downsample_wt=0.5 \
    seed=1
```

In the Nextflow pipeline, `downsample_wt`/`seed` are driven by `params.feature_select_downsample_wt`
(see [Parameters](../configuration.md#parameter-reference)) — `AGGREGATE_HALF` derives a distinct seed per
`(bootstrap_idx, half_num)` so each pseudo-replicate half draws an independent wildtype
subsample, which is what lets the bootstrap comparison test feature reproducibility against
different WT samples rather than reusing one fixed sample everywhere. `AGGREGATE_FEATURE_TYPE`
(the full, un-split aggregation) uses a fixed seed, since it has no repeated per-instance
identity to vary by.

## Feature chunking

Both entry points evaluate `feature_chunk_size` feature columns per Polars
query rather than all of them at once, joining the per-chunk results back
together on the label column. The statistic is unaffected — each chunk runs the
same expression over a narrower projection — but peak memory becomes
proportional to the chunk width instead of the full feature count.

This is not a tuning nicety. Aggregating all ~1731 production features in one
query OOM-killed (exit 137) every `KS`, `AUROC`, `KSnegLogP` and
`AUROCnegLogP` task of the 111925 run, inside `sink_parquet`. Because every
`AGGREGATE_*` process carries `errorStrategy 'ignore'`, those tasks vanished
silently and the feature-selection branch never reached `CORRELATE_FEATURES`.

Peak memory scales with `feature_chunk_size × n_variant_labels`, and for the
reference-based aggregators (`KS`, `signedKS`, `QQ`, `AUROC`, and the two
p-value variants) with the size of the cross-joined control pool on top.
Runtime, by contrast, is essentially **flat** in this value for those
aggregators — so it is a memory dial, not a speed/memory trade-off, and
halving it when a task is killed costs close to nothing.

`KS` is the aggregator that pins the default; it needs a smaller chunk than
`AUROC` does. If `KS`/`AUROC` tasks still come back killed, halve
`params.aggregate_feature_chunk_size`; a run using only `mean`/`median`/`std`/
`MAD` can raise it. Measured numbers and the benchmark harness live in
`tests/benchmarks/`.

Setting `feature_chunk_size` to `null` turns chunking off: every feature is
evaluated in one query, which is exactly the shape that OOM-killed the
production tasks described above. It exists for small inputs and for
reproducing the pre-chunking behaviour — not for a full-size batch. The
aggregated values are identical either way; only the number of queries (and
the peak memory) differs.

Output rows are sorted by the label column, so aggregate output is
reproducible run to run — `group_by` alone is not order-preserving under
Polars' multithreaded execution.

See [API Reference: aggregate](../api/aggregate.md) for full function
documentation, including the `BaseAggregator` class hierarchy.
