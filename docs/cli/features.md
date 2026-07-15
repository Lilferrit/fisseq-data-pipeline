# Feature Selection

`features.py` implements a bootstrap pseudo-replicate feature-selection pipeline
as five Hydra entry points, each a Nextflow process (see
[Nextflow Workflow](../nextflow.md)). Cells are split into stratified 50/50
pseudo-replicate halves across `params.bootstrap` replicates; each half is
aggregated per feature type (via [`fisseq-aggregate-feature-type`](aggregate.md)),
correlated against its partner half, and a per-feature blocklist is derived from
the median correlation across all bootstrap replicates. The final stage joins the
per-feature-type aggregates, applies the blocklist, and runs pycytominer feature
selection.

All configs extend the [common config fields](qcfilter.md#common-config-fields).

## 1. `fisseq-generate-split` (`GENERATE_SPLIT`)

Generates one stratified 50/50 pseudo-replicate split.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Glob pattern or path to cell-level data. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `random_state` | **required** | Seed for the stratified split — set to the bootstrap-loop index in Nextflow, so each replicate is distinct and reproducible. |

**Output**: `half1.parquet`, `half2.parquet` (single-column row-index files).

```bash
uv run fisseq-generate-split \
    output_dir=./out \
    input_file=data/normalized.parquet \
    random_state=3
```

## 2. `fisseq-correlate-features` (`CORRELATE_FEATURES`)

Computes per-feature Pearson correlation between two aggregate halves for the same
feature type.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `half1_file` | **required** | First half's per-feature-type aggregate parquet. |
| `half2_file` | **required** | Second half's per-feature-type aggregate parquet. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |

**Output**: `correlations.parquet` (columns: `feature`, `r`, `r_squared`, `p_value`).

```bash
uv run fisseq-correlate-features \
    output_dir=./out \
    half1_file=out/half1.mean.parquet \
    half2_file=out/half2.mean.parquet
```

## 3. `fisseq-blocklist` (`BLOCKLIST`)

The one intentional cross-bootstrap synchronization point: gathers every bootstrap
replicate's correlation table for one feature type and computes each feature's
median `r` across replicates.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `correlation_files` | **required** | Glob pattern matching all bootstrap-replicate correlation parquet files for one feature type. |
| `minimum_correlation` | `0.5` | Minimum median Pearson `r` required for a feature to pass. |

**Output**: `blocklist.parquet` (columns: `feature`, `median_r`, `feature_ok`).

```bash
uv run fisseq-blocklist \
    output_dir=./out \
    'correlation_files=out/correlations/mean/*.parquet' \
    minimum_correlation=0.5
```

## 4. `fisseq-combine-blocklists` (`COMBINE_BLOCKLISTS`)

Concatenates every feature type's blocklist into one combined blocklist (a plain
concat is correct — stat-suffixed feature names never collide across feature
types).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `blocklist_files` | **required** | Glob pattern matching all per-feature-type blocklist parquet files. |

**Output**: `blocklist.parquet`.

```bash
uv run fisseq-combine-blocklists \
    output_dir=./out \
    'blocklist_files=out/blocklists/*.parquet'
```

## 5. `fisseq-feature-select` (`FINALIZE_FEATURE_SELECT`)

The final stage: joins every feature type's full aggregate (from
[`fisseq-aggregate-feature-type`](aggregate.md)) on `label_column`, drops blocked
feature columns, and runs `pycytominer.feature_select` (variance threshold,
built-in blocklist, correlation threshold).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Raw/normalized cell-level input — used only to derive per-variant metadata. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `feature_type_files` | **required** | Glob pattern matching per-feature-type full aggregate parquet files. |
| `block_list_file` | **required** | Combined blocklist parquet, with `feature` and `feature_ok` columns. |
| `compute_impact_score` | `true` | Compute per-variant impact score (cosine distance vs. synonymous baseline) after feature selection. |

**Output**: glob input → `{output_root}.output.parquet` or `{output_dir}/output.parquet`;
single-file input → `{output_root}.{stem}.parquet` or `{output_dir}/{stem}.parquet`.

```bash
uv run fisseq-feature-select \
    output_dir=./out \
    input_file=out/normalized.parquet \
    'feature_type_files=out/aggregates/*.parquet' \
    block_list_file=out/blocklist.parquet
```

See [API Reference: features](../api/features.md) for full function
documentation, including `pyc_feature_select` and `compute_feature_correlations`.
