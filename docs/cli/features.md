# Feature Selection

The BATCHWISE bootstrap pseudo-replicate feature-selection pipeline (run once
per batch) is implemented as five Hydra entry points, one per module, each a
Nextflow process (see [Nextflow Workflow](../nextflow.md)). Cells are split
into stratified 50/50 pseudo-replicate halves across
`params.feature_select_bootstrap_reps` replicates; each half is
aggregated per feature type (via [`python -m fisseq_data_pipeline.aggregatefeaturetype`](aggregate.md)),
correlated against its partner half, and a per-feature blocklist is derived from
the median correlation across all bootstrap replicates. The final stage joins the
per-feature-type aggregates, applies the blocklist, and runs pycytominer feature
selection.

A separate, much simpler GLOBAL entry point (§6 below) runs once per active
global group, reusing this BATCHWISE pipeline's already-computed per-batch
outputs rather than recomputing anything from cells.

All configs extend the [common config fields](qcfilter.md#common-config-fields).

## 1. `python -m fisseq_data_pipeline.generatesplit` (`GENERATE_SPLIT`)

Generates one stratified 50/50 pseudo-replicate split.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Glob pattern or path to cell-level data. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `random_state` | **required** | Seed for the stratified split — set to the bootstrap-loop index in Nextflow, so each replicate is distinct and reproducible. |

**Output**: `half1.parquet`, `half2.parquet` (single-column row-index files).

```bash
uv run python -m fisseq_data_pipeline.generatesplit \
    output_dir=./out \
    input_file=data/normalized.parquet \
    random_state=3
```

## 2. `python -m fisseq_data_pipeline.correlatefeatures` (`CORRELATE_FEATURES`)

Computes per-feature Pearson correlation between two aggregate halves for the same
feature type.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `half1_file` | **required** | First half's per-feature-type aggregate parquet. |
| `half2_file` | **required** | Second half's per-feature-type aggregate parquet. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |

**Output**: `correlations.parquet` (columns: `feature`, `r`, `r_squared`, `p_value`).

```bash
uv run python -m fisseq_data_pipeline.correlatefeatures \
    output_dir=./out \
    half1_file=out/half1.mean.parquet \
    half2_file=out/half2.mean.parquet
```

## 3. `python -m fisseq_data_pipeline.blocklist` (`BLOCKLIST`)

The one intentional cross-bootstrap synchronization point: gathers every bootstrap
replicate's correlation table for one feature type and computes each feature's
median `r` across replicates.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `correlation_files` | **required** | Glob pattern matching all bootstrap-replicate correlation parquet files for one feature type. |
| `minimum_correlation` | `0.5` | Minimum median Pearson `r` required for a feature to pass. |

**Output**: `blocklist.parquet` (columns: `feature`, `median_r`, `feature_ok`).

```bash
uv run python -m fisseq_data_pipeline.blocklist \
    output_dir=./out \
    'correlation_files=out/correlations/mean/*.parquet' \
    minimum_correlation=0.5
```

## 4. `python -m fisseq_data_pipeline.combineblocklists` (`COMBINE_BLOCKLISTS`)

Concatenates every feature type's blocklist into one combined blocklist (a plain
concat is correct — stat-suffixed feature names never collide across feature
types).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `blocklist_files` | **required** | Glob pattern matching all per-feature-type blocklist parquet files. |

**Output**: `blocklist.parquet`.

```bash
uv run python -m fisseq_data_pipeline.combineblocklists \
    output_dir=./out \
    'blocklist_files=out/blocklists/*.parquet'
```

## 5. `python -m fisseq_data_pipeline.featureselect` (`FINALIZE_FEATURE_SELECT`)

The final stage: joins every feature type's full aggregate (from
[`python -m fisseq_data_pipeline.aggregatefeaturetype`](aggregate.md)) on `label_column`, drops blocked
feature columns, and runs `pycytominer.feature_select` (variance threshold,
built-in blocklist, correlation threshold).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Raw/normalized cell-level input — used only to derive per-variant metadata. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `feature_type_files` | **required** | Glob pattern matching per-feature-type full aggregate parquet files. |
| `block_list_file` | **required** | Combined blocklist parquet, with `feature` and `feature_ok` columns. |
| `compute_impact_score` | `true` | Compute per-variant impact score (cosine distance vs. synonymous baseline) after feature selection. |
| `run_pca` | `false` | Compute PCA on the final selected/normalized feature matrix, appending `meta_pc_1..meta_pc_{pca_n_components}` and writing a separate PCA-components output file. |
| `pca_n_components` | `10` | Number of principal components to compute and retain. |
| `run_umap` | `false` | Compute UMAP on the final selected/normalized feature matrix, appending `meta_umap_1..meta_umap_{umap_n_components}`. PCA and UMAP are computed independently, both on the same feature matrix. |
| `umap_n_components` | `2` | Dimensionality of the UMAP embedding. |
| `umap_n_neighbors` | `10` | `umap.UMAP`'s local neighborhood size. |
| `umap_metric` | `"cosine"` | `umap.UMAP`'s distance metric. |
| `umap_min_dist` | `0.1` | `umap.UMAP`'s minimum embedded distance between points. |
| `umap_random_state` | `42` | Seed for UMAP's fit; `null` disables seeding (faster, multithreaded, nondeterministic). |

**Output**: glob input → `{output_root}.output.parquet` or `{output_dir}/output.parquet`;
single-file input → `{output_root}.{stem}.parquet` or `{output_dir}/{stem}.parquet`.
When `run_pca=true`, also writes `{output_root}.pca_components.parquet` or
`{output_dir}/pca_components.parquet` — one row per principal component,
with one column per feature used in the fit (named by that feature's actual
column name, holding its loading), plus `meta_variance_explained`,
`meta_cumulative_variance_explained`, and `meta_component_idx`.

```bash
uv run python -m fisseq_data_pipeline.featureselect \
    output_dir=./out \
    input_file=out/normalized.parquet \
    'feature_type_files=out/aggregates/*.parquet' \
    block_list_file=out/blocklist.parquet
```

## 6. `python -m fisseq_data_pipeline.globalfeatureselect` (`GLOBAL_FEATURE_SELECT`)

Runs once per active global group (see
[Configuration: Global groups](../configuration.md#global-groups)). Reuses the
group's member batches' already-published BATCHWISE feature-selection
artifacts directly — no cell-level recomputation:

1. For each member batch, joins that batch's own per-feature-type aggregate
   files (`feature_select_batchwise/<batch>/aggregates/*.parquet`) and
   normalizes the joined table to that batch's own synonymous baseline (this
   serves as both batch correction and normalization).
2. Concatenates every member batch's normalized table and takes the
   per-feature median, grouped by `label_column` (a variant can appear in
   more than one batch).
3. Combines each member batch's own combined blocklist
   (`feature_select_batchwise/<batch>/blocklist.parquet`) using an agreement
   threshold across batches.
4. Drops columns blocked by step 3 and runs `pyc_feature_select` (the same
   function `FINALIZE_FEATURE_SELECT` uses).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `pipeline_dir` | **required** | Absolute path to the pipeline's root output directory. |
| `batch_stems` | **required** | List of the active group's member batch stems (only those with `run_feature_selection` enabled). |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `min_batches_ok` | `null` | Minimum number of member batches that must mark a feature ok for it to be globally ok. `null` requires unanimity across batches that report on it. |
| `run_pca` | `false` | Compute PCA on the final selected/normalized feature matrix, appending `meta_pc_1..meta_pc_{pca_n_components}` and writing a separate PCA-components output file. Always uses the plain pipeline-wide value (not per-batch overridable here — see [Configuration](../configuration.md#per-batch-parameter-overrides)). |
| `pca_n_components` | `10` | Number of principal components to compute and retain. |
| `run_umap` | `false` | Compute UMAP on the final selected/normalized feature matrix, appending `meta_umap_1..meta_umap_{umap_n_components}`. PCA and UMAP are computed independently, both on the same feature matrix. |
| `umap_n_components` | `2` | Dimensionality of the UMAP embedding. |
| `umap_n_neighbors` | `10` | `umap.UMAP`'s local neighborhood size. |
| `umap_metric` | `"cosine"` | `umap.UMAP`'s distance metric. |
| `umap_min_dist` | `0.1` | `umap.UMAP`'s minimum embedded distance between points. |
| `umap_random_state` | `42` | Seed for UMAP's fit; `null` disables seeding (faster, multithreaded, nondeterministic). |

**Output**: `aggregate.parquet` (the selected, cross-batch median aggregate
table) and `blocklist.parquet` (the combined global blocklist, columns
`feature`, `n_batches`, `n_ok`, `feature_ok`). When `run_pca=true`, also
writes `pca_components.parquet` — one row per principal component, with one
column per feature used in the fit (named by that feature's actual column
name, holding its loading), plus `meta_variance_explained`,
`meta_cumulative_variance_explained`, and `meta_component_idx`.

```bash
uv run python -m fisseq_data_pipeline.globalfeatureselect \
    output_dir=./out \
    pipeline_dir=/path/to/experiment \
    'batch_stems=[batch1,batch2]'
```

See [API Reference: features](../api/features.md) for full function
documentation, including `pyc_feature_select` and `compute_feature_correlations`.
