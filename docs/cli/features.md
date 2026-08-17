# Feature Selection

The BATCHWISE WT-null bootstrap feature-selection pipeline (run once per
batch) is implemented as five Hydra entry points, one per module, each a
Nextflow process (see [Nextflow Workflow](../nextflow.md)). Every configured
feature type (`params.feature_select_types`) is first fully aggregated (via
[`python -m fisseq_data_pipeline.aggregatefeaturetype`](aggregate.md)), then
routed to one of two reproducibility-gate branches based on
`params.feature_select_wt_null_types`:

- **WT-null bootstrap** (distributional-distance aggregators — `KS`,
  `signedKS`, `QQ`, `AUROC` by default): across
  `params.feature_select_wt_null_bootstraps` bootstrap replicates, the
  batch's control (wildtype) pool is split into two disjoint halves and the
  configured aggregator is computed *between* the two halves for every
  feature — a same-population comparison with zero true biological signal
  by construction, used as that feature's null noise floor for one
  replicate. Each feature's mean null value across bootstraps is then
  gated by an upper Tukey fence over the batch's per-feature-type null-mean
  distribution.
- **Passthrough** (every other configured feature type — `mean`, `median`,
  `MAD`, `std` by default): no reproducibility computation. These
  aggregators have no reference distribution to compare against, so the
  WT-null concept doesn't apply; every feature is marked ok, deferring
  entirely to pycytominer's variance/correlation thresholds in
  `FINALIZE_FEATURE_SELECT`.

Both branches write the same blocklist schema (`feature`, `feature_ok`,
`null_mean`, `threshold`, `n_bootstraps`), concatenated by
`COMBINE_BLOCKLISTS`. The final stage joins the per-feature-type aggregates,
applies the combined blocklist, and runs pycytominer feature selection.

A separate, much simpler GLOBAL entry point (§6 below) runs once per active
global group, reusing this BATCHWISE pipeline's already-computed per-batch
outputs rather than recomputing anything from cells.

All configs extend the [common config fields](qcfilter.md#common-config-fields).

## 1. `python -m fisseq_data_pipeline.wtnullaggregate` (`WT_NULL_AGGREGATE`)

Computes one WT-null bootstrap replicate for one feature type: splits the
batch's control pool into two disjoint halves seeded by `bootstrap_idx`,
optionally downsamples each half independently, relabels the first half as
a single synthetic "variant" group compared against the second half as the
reference pool, and reruns the existing aggregator machinery on that
synthetic two-group frame — no new statistic math, the same
`KSAggregator`/`SignedKSAggregator`/`QQCorrelationAggregator`/
`AUROCAggregator` used elsewhere. The raw per-feature statistic is then
passed through a per-aggregator transform so "larger = more suspicious"
holds uniformly before the Tukey fence is applied downstream:

| Aggregator | Transform | Why |
| ---------- | --------- | --- |
| `KS` | none | Already ≥ 0; null centers near 0. |
| `signedKS` | `abs(value)` | Same magnitude as `KS` but signed; the sign is meaningless for a WT-vs-WT null. |
| `QQ` | `1 - value` | Identical distributions → `QQ ≈ 1`, so a *low* `QQ` indicates divergence — this flips the direction to match the others. |
| `AUROC` | `abs(value - 0.5)` | Null centers at `0.5`, not `0`. |

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Glob pattern or path to cell-level data. |
| `aggregator` | **required** | One of `KS`, `signedKS`, `QQ`, `AUROC`. |
| `downsample_wt` | `null` | Optional downsampling of each split half's control rows. A float in `(0, 1)` keeps that fraction; an int keeps that many, clamped (with a logged warning) to however many rows a half actually has if the request is larger. `null` disables downsampling — each half is the full disjoint split. |
| `bootstrap_idx` | **required** | This replicate's bootstrap-loop index (`1..params.feature_select_wt_null_bootstraps`). Seeds the disjoint split directly (`seed=bootstrap_idx`) and, when `downsample_wt` is set, seeds each half's downsample independently (`bootstrap_idx*2 + 1` for h1, `bootstrap_idx*2 + 2` for h2) — the same per-`(bootstrap_idx, half_num)` seed idiom the old `AGGREGATE_HALF` used. |
| `per_barcode` | `false` | Compute each feature's statistic per (synthetic group, barcode) first, then reduce by median across barcodes. Must match `AGGREGATE_FEATURE_TYPE`'s setting for the same batch, or the WT-null check stops being apples-to-apples. |
| `barcode_column` | `"meta_barcode"` | Column identifying the barcode a cell was measured from. Only consulted when `per_barcode` is `true`. |

**Output**: `wt_null.parquet` (columns: `feature`, `value` — the
post-transform null statistic for this bootstrap replicate).

```bash
uv run python -m fisseq_data_pipeline.wtnullaggregate \
    output_dir=./out \
    input_file=data/normalized.parquet \
    aggregator=KS \
    bootstrap_idx=3 \
    downsample_wt=1000
```

## 2. `python -m fisseq_data_pipeline.wtnullblocklist` (`WT_NULL_BLOCKLIST`)

The one intentional cross-bootstrap synchronization point for WT-null
feature types: gathers every bootstrap replicate's `wt_null.parquet` for one
feature type, and for each feature averages `value` across bootstraps
(`null_mean`), skipping non-finite replicate values (e.g. a
degenerate/constant WT sub-distribution). A feature with zero finite
replicate values has `null_mean = null` and `n_bootstraps = 0`, and is
unconditionally blocked — its own reproducibility can't be established —
and excluded from the Tukey-fence quantile computation below so it can't
skew the fence for every other feature.

A feature passes (`feature_ok = true`) iff its own `null_mean` is finite and
does not exceed the upper Tukey fence:

```
threshold = Q1(null_mean) + tukey_multiplier * IQR(null_mean)
```

computed over every feature's finite `null_mean` for this feature type.
**This fence is anchored on Q1, not the conventional Q3 anchor of a
textbook upper Tukey fence (`Q3 + 1.5*IQR`)** — a deliberate, confirmed
choice, making the default a stricter cutoff than a standard outlier fence.
Since `Q1 <= Q3` always, this fence never sits above the standard one for
the same multiplier.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `wt_null_files` | **required** | Glob pattern matching all bootstrap-replicate `wt_null.parquet` files for one feature type. |
| `tukey_multiplier` | `1.5` | IQR multiplier for the upper reproducibility fence (see above). |

**Output**: `blocklist.parquet` (columns: `feature`, `feature_ok`,
`null_mean`, `threshold`, `n_bootstraps`).

```bash
uv run python -m fisseq_data_pipeline.wtnullblocklist \
    output_dir=./out \
    'wt_null_files=out/wt_null/*/KS/*.parquet' \
    tukey_multiplier=1.5
```

## 3. `python -m fisseq_data_pipeline.passthroughblocklist` (`PASSTHROUGH_BLOCKLIST`)

For a feature type not in `params.feature_select_wt_null_types` (by default,
`mean`, `median`, `MAD`, `std`): no reproducibility computation. Scans the
feature type's full aggregate parquet's schema (no data loaded) and emits
one row per feature column, all marked `feature_ok = true` with null audit
columns — the same blocklist schema `WT_NULL_BLOCKLIST` writes, so
`COMBINE_BLOCKLISTS`'s plain concat keeps working unmodified. Those
features are still subject to pycytominer's variance/correlation
thresholds later, in `FINALIZE_FEATURE_SELECT`.

A real CLI entry point (rather than an inline Nextflow shell one-liner) for
auditability and skill parity with `WT_NULL_BLOCKLIST`.

| Field | Default | Description |
| ----- | ------- | ----------- |
| `aggregate_file` | **required** | This feature type's full aggregate parquet (output of `AGGREGATE_FEATURE_TYPE`). |

**Output**: `blocklist.parquet` (columns: `feature`, `feature_ok`,
`null_mean`, `threshold`, `n_bootstraps` — the latter three always `null`).

```bash
uv run python -m fisseq_data_pipeline.passthroughblocklist \
    output_dir=./out \
    aggregate_file=out/aggregates/mean.parquet
```

## 4. `python -m fisseq_data_pipeline.combineblocklists` (`COMBINE_BLOCKLISTS`)

Concatenates every feature type's blocklist (from either branch above) into
one combined blocklist (a plain concat is correct — stat-suffixed feature
names never collide across feature types).

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
   Only ever reads that file's `feature`/`feature_ok` columns — agnostic to
   which branch (WT-null or passthrough) produced them.
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
documentation, including `pyc_feature_select`.

## Migration note (from the Fisher-z correlation gate)

Earlier pipeline versions determined reproducibility by splitting cells into
stratified pseudo-replicate halves (`GENERATE_SPLIT`), aggregating each half
(`AGGREGATE_HALF`), correlating the two halves per bootstrap replicate
(`CORRELATE_FEATURES`), and Fisher-z-averaging those correlations with a
precision-adjusted lower-confidence-bound gate (the old `BLOCKLIST`,
controlled by `minimum_correlation`/`se_multiplier`). That whole chain —
along with `params.feature_select_min_correlation`,
`params.feature_select_se_multiplier`, and
`params.feature_select_bootstrap_variant_downsample` — has been **removed**
(still setting any of them now errors, not silently ignored) in favor of the
WT-null bootstrap procedure above, which measures each feature's own noise
floor directly instead of a between-half correlation. It also applies more
naturally to the distributional-distance aggregators (`KS`, `signedKS`,
`QQ`, `AUROC`), which the old Pearson-correlation approach didn't fit well,
and exempts the summary-statistic aggregators (`mean`, `median`, `MAD`,
`std`) from the check entirely, since they have no reference distribution
to correlate against in the first place.

`params.feature_select_bootstrap_reps` (the old Fisher-z bootstrap count)
was renamed to `params.feature_select_wt_null_bootstraps` rather than
repurposed in place, since it now counts a different thing (WT-null
replicates, not split/correlate replicates) — the old name errors if set.
Its default was also raised from `10` to `25`: a Tukey fence's Q1/IQR
estimate needs more replicates to stabilize than the old Fisher-z mean/SE
estimate did.

`params.feature_select_downsample_wt` is unchanged and reused as-is by
`WT_NULL_AGGREGATE` (previously `AGGREGATE_HALF`'s WT downsample knob) —
see [Parameters](../configuration.md#parameters).
