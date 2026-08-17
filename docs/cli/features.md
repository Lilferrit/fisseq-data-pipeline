# Feature Selection

The BATCHWISE bootstrap pseudo-replicate feature-selection pipeline (run once
per batch) is implemented as five Hydra entry points, one per module, each a
Nextflow process (see [Nextflow Workflow](../nextflow.md)). Cells are split
into stratified 50/50 pseudo-replicate halves across
`params.feature_select_bootstrap_reps` replicates; each half is
aggregated per feature type (via [`python -m fisseq_data_pipeline.aggregatefeaturetype`](aggregate.md)),
correlated against its partner half, and a per-feature blocklist is derived from
a Fisher-z-averaged correlation estimate (with a paired precision/quality gate)
across all bootstrap replicates. The final stage joins the
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
| `bootstrap_variant_downsample` | `null` | Optional: randomly sample this many variants from the set present in both halves before computing correlations, independently per bootstrap replicate — adds variant-subsampling variance on top of the half-split randomness. `null` disables it (every joint variant is used, prior behavior). If the requested count exceeds the number of joint variants, all of them are used instead (logged once as a warning). Distinct from `feature_select_downsample_wt` (cell-level control-row downsampling at aggregation time, a different lever) — this subsamples *variants*, at *correlation* time. |
| `bootstrap_idx` | `0` | This replicate's bootstrap-loop index. Combined with `seed` (`seed + bootstrap_idx * 1000`) to derive a per-replicate seed for `bootstrap_variant_downsample`, deliberately independent of `GENERATE_SPLIT`'s own per-replicate seed. Ignored when `bootstrap_variant_downsample` is `null`. |
| `seed` | `0` | Base seed combined with `bootstrap_idx` to derive the per-replicate variant-downsample seed. Ignored when `bootstrap_variant_downsample` is `null`. |

**Output**: `correlations.parquet` (columns: `feature`, `r`, `r_squared`, `p_value`).

```bash
uv run python -m fisseq_data_pipeline.correlatefeatures \
    output_dir=./out \
    half1_file=out/half1.mean.parquet \
    half2_file=out/half2.mean.parquet \
    bootstrap_variant_downsample=50 \
    bootstrap_idx=3 \
    seed=0
```

In the Nextflow pipeline, `bootstrap_variant_downsample` is driven by
`params.feature_select_bootstrap_variant_downsample` (see
[Parameters](../configuration.md#parameters)); `bootstrap_idx` is the loop index Nextflow
already threads through `CORRELATE_FEATURES` for the output filename.

## 3. `python -m fisseq_data_pipeline.blocklist` (`BLOCKLIST`)

The one intentional cross-bootstrap synchronization point: gathers every bootstrap
replicate's correlation table for one feature type and, for each feature,
Fisher-z-transforms every replicate's `r` (`z = arctanh(clip(r, -1+eps, 1-eps))`),
averages in z-space, and back-transforms to a point estimate
(`r_est = tanh(mean(z))`) plus its standard error
(`se_z = std(z, ddof=1) / sqrt(n_replicates)`). A feature is marked `feature_ok` if
its **precision-adjusted** estimate (`adjusted_r`) clears `minimum_correlation`.
`adjusted_r` applies an optional lower-confidence-bound adjustment controlled by
`se_multiplier`:

- `se_multiplier` is `None`: `adjusted_r = r_est` (no penalty; gates on the raw
  point estimate alone).
- `se_multiplier` is a float (default `1.0`): the adjustment is applied **in
  Fisher-z space**, not directly on `r_est` — `adjusted_z = z_mean -
  se_multiplier * se_z`, then `adjusted_r = tanh(adjusted_z)`. This is deliberate:
  `se_z` is the standard error of the mean **Fisher-z** estimate, whose sampling
  distribution is approximately symmetric and unbounded, so a z-space shift is
  well-defined; `r` is bounded to `[-1, 1]`, so subtracting a z-space-derived
  quantity directly from `r_est` mixes units and can behave oddly near `r =
  ±1`. Larger `se_multiplier` is stricter.

A feature with fewer than 2 usable replicates has `se_z = null`, so (whenever
`se_multiplier` is set) `adjusted_r` is also `null` and the feature fails
automatically — a single replicate can't support a precision claim.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `correlation_files` | **required** | Glob pattern matching all bootstrap-replicate correlation parquet files for one feature type. |
| `minimum_correlation` | `0.5` | Magnitude gate: minimum precision-adjusted correlation estimate (`adjusted_r`) required for a feature to pass. |
| `se_multiplier` | `1.0` | Precision/confidence adjustment applied in Fisher-z space before the magnitude gate (see above). `null` disables the adjustment entirely (gates on raw `r_est`). |

**Output**: `blocklist.parquet` (columns: `feature`, `r_est`, `se_z`, `n_replicates`, `adjusted_r`, `feature_ok`).

```bash
uv run python -m fisseq_data_pipeline.blocklist \
    output_dir=./out \
    'correlation_files=out/correlations/mean/*.parquet' \
    minimum_correlation=0.5 \
    se_multiplier=1.0
```

**Migration note (from `max_se_z`)**: earlier pipeline versions used a two-gate
strategy — `feature_ok` required both `r_est >= minimum_correlation` *and* an
independent quality gate `se_z <= max_se_z` (default `max_se_z=0.0884`). That field
has been **removed** (a Hydra config or batch YAML still setting `max_se_z` will
now error, not be silently ignored) in favor of the single lower-confidence-bound
criterion above. `minimum_correlation=0.5, se_multiplier=1.0` is a rough behavioral
match to the old `minimum_correlation=0.5, max_se_z=0.0884` default at around
`bootstrap_reps≈5`-ish — these are genuinely different criteria, not a
reparameterization of the same one, so revalidate against your own
`bootstrap_reps` and data rather than assuming parity.

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
group's member batches' own BATCHWISE feature-selection artifacts, passed in
as explicit staged files rather than re-derived from a pipeline directory
path — no cell-level recomputation:

1. For each distinct batch named in `agg_batch_stems`, joins that batch's own
   staged per-feature-type aggregate files and normalizes the joined table to
   that batch's own synonymous baseline (this serves as both batch correction
   and normalization).
2. Concatenates every member batch's normalized table and takes the
   per-feature median, grouped by `label_column` (a variant can appear in
   more than one batch).
3. Combines each member batch's own staged combined blocklist file (one per
   entry in `bl_batch_stems`) using an agreement threshold across batches.
4. Drops columns blocked by step 3 and runs `pyc_feature_select` (the same
   function `FINALIZE_FEATURE_SELECT` uses).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `agg_batch_stems` | **required** | Owning batch stem for each staged aggregate file, same order/length as `n_agg_files`'s implied file list (`agg_input_1.parquet`, `agg_input_2.parquet`, ...); a batch stem may repeat, once per per-feature-type file. |
| `n_agg_files` | **required** | Number of staged aggregate files (see `agg_batch_stems`). |
| `bl_batch_stems` | **required** | Owning batch stem for each staged blocklist file (`bl_input_1.parquet`, ...) — one entry per batch. |
| `n_blocklist_files` | **required** | Number of staged blocklist files (see `bl_batch_stems`). |
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
`feature`, `n_batches`, `n_ok`, `feature_ok`). Always also writes one
`aggregate_{feature_type}.parquet` file per aggregate feature type present in
the cross-batch median aggregate (e.g. `aggregate_mean.parquet`,
`aggregate_KS.parquet`, `aggregate_KSnegLogP.parquet`, ... — one of `mean`,
`median`, `MAD`, `std`, `KS`, `signedKS`, `QQ`, `AUROC`, `KSnegLogP`,
`AUROCnegLogP`, whichever are present), containing `label_column` plus that
feature type's columns from the cross-batch median aggregate — captured after
per-batch normalization and cross-batch medianing but before the global
blocklist is (re-)applied and before pycytominer feature selection. Because
feature selection can only ever drop columns, each of these files' columns
are a superset of that feature type's columns in `aggregate.parquet`. When
`run_pca=true`, also writes `pca_components.parquet` — one row per principal
component, with one column per feature used in the fit (named by that
feature's actual column name, holding its loading), plus
`meta_variance_explained`, `meta_cumulative_variance_explained`, and
`meta_component_idx`.

Unlike this pipeline's other CLI entry points, `globalfeatureselect` expects
its aggregate/blocklist input files staged in the working directory under
fixed, auto-numbered names (`agg_input_1.parquet`, `agg_input_2.parquet`,
... / `bl_input_1.parquet`, ...) — the convention Nextflow's `stageAs`
produces for `GLOBAL_FEATURE_SELECT` (see `modules/local/global_feature_select.nf`).
To invoke it standalone, stage files under those names first:

```bash
cp path/to/batch1/aggregates/mean.parquet ./agg_input_1.parquet
cp path/to/batch2/aggregates/mean.parquet ./agg_input_2.parquet
cp path/to/batch1/blocklist.parquet ./bl_input_1.parquet
cp path/to/batch2/blocklist.parquet ./bl_input_2.parquet
uv run python -m fisseq_data_pipeline.globalfeatureselect \
    output_dir=./out \
    'agg_batch_stems=[batch1,batch2]' \
    n_agg_files=2 \
    'bl_batch_stems=[batch1,batch2]' \
    n_blocklist_files=2
```

See [API Reference: features](../api/features.md) for full function
documentation, including `pyc_feature_select` and `compute_feature_correlations`.
