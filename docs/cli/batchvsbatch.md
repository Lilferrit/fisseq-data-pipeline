# Batch vs. Batch

`fisseq-batch-vs-batch` (Nextflow process `BATCHVSBATCH`, run once pre- and once
post-normalization) detects per-variant batch effects using a multiclass XGBoost
classifier. For each variant, a single model is trained to predict the batch label
from cell morphology features. From its predicted probabilities, a one-vs-rest ROC
AUC and Mann-Whitney U p-value are extracted for every (variant, batch) pair,
quantifying how distinguishable each batch is from the others within that variant.

The global 80/10/10 train/test/val split is stratified on a composite (variant,
batch) key so every variant's rows in each split span all of its batches. One
model is trained per variant (not per (variant, batch) pair).

## Config fields

Extends `LabeledInputConfig` plus the [common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Path to feature-selected or normalized cell-level parquet. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `batch_column` | `"meta_batch"` | Column identifying batch labels. |
| `random_state` | `42` | Seed for the stratified split. |
| `feature_cols` | `null` | Explicit list of feature column names; auto-detected if `null`. |
| `min_cells` | `50` | Skip variants with fewer than this many cells total. |
| `min_batches` | `2` | Skip variants appearing in fewer than this many unique batches. |
| `use_parent_name` | `false` | If `true`, derive the batch label from each input file's parent directory name rather than its stem — set this when input files share a filename but live in different subdirectories (e.g. `qc_filter/*/filtered_cells.parquet`). |
| `xgboost.num_boost_round` | `100` | Maximum boosting rounds. |
| `xgboost.early_stopping_rounds` | `5` | Stop early if the eval metric does not improve. |
| `xgboost.weigh_samples` | `true` | Use balanced sample weights. |
| `xgboost.params.max_depth` | `3` | Maximum tree depth. |
| `xgboost.params.subsample` | `0.5` | Fraction of rows sampled per tree. |

## Output files

- `{output_dir}/results.parquet` — one row per (variant, batch) pair with columns
  `variant`, `batch`, `auroc`, `mw_pvalue`, `n_batch_cells`, `n_cells`.

## Example

```bash
uv run fisseq-batch-vs-batch \
    output_dir=./out \
    input_file=out/features.parquet \
    batch_column=meta_batch
```

See [API Reference: batchvsbatch](../api/batchvsbatch.md) for full function
documentation.
