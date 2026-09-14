# Global Feature Selection

`python -m fisseq_data_pipeline.globalfeatureselect` (Nextflow process
`GLOBAL_FEATURE_SELECT`) produces one cross-experiment feature-selected
aggregate per active global channel.

Unlike the batchwise feature-selection chain, it does **no cell-level
recompute**. It reads each member experiment's already-published
`feature_select_batchwise/<batch_stem>/{aggregates,blocklist.parquet}` directly
off `pipeline_dir`, looping over `batch_stems` in Python:

1. Combine the member experiments' block-lists by agreement threshold
   (`min_batches_ok`).
2. Join each experiment's per-feature-type aggregates and normalize to that
   experiment's own synonymous baseline.
3. Take the cross-experiment median.
4. Run pycytominer feature selection over the result.

## Config fields

Extends `AppConfig` — see the [common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `pipeline_dir` | **required** | Root directory holding the published batchwise artifacts. |
| `batch_stems` | **required** | This channel's member experiments. |
| `feature_select_types` | **required** | Which per-feature-type aggregates to read. Passed through so stale files from a prior run with a larger `feature_select_types` cannot leak in (`publishDir` never deletes). |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `min_batches_ok` | `null` | Minimum member experiments that must mark a feature ok for it to be globally ok. `null` means every experiment reporting on it must agree. |
| `compute_impact_score` | `true` | Append `meta_impact_score`. |
| `run_pca` / `run_umap` | `false` | Optional dimensionality reduction, computed independently of each other on the same selected matrix. |
| `random_seed` | `0` | The shared pipeline seed — drives PCA's solver and UMAP's fit. |

## Output files

| File | Contents |
| ---- | -------- |
| `aggregate.parquet` | Cross-experiment median aggregate, pycytominer-selected. |
| `blocklist.parquet` | Combined block-list across member experiments. |
| `pca_components.parquet` | Only when `run_pca=true`. |
