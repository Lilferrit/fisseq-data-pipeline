# One-vs-WT

`fisseq-ovwt` (Nextflow processes `OVWT_BATCHWISE` and `OVWT_GLOBAL`) trains a
separate XGBoost binary classifier for each non-wildtype variant, treating the
task as "this variant vs. wildtype." An 80/10/10 train/test/val split (stratified
by label) is shared across all variants. Wildtype cells can be downsampled to
reduce class imbalance. Results (per-variant AUROC and accuracy on train/val/test
splits) and all trained models are serialized to disk.

## Config fields

Extends `LabeledInputConfig` plus the [common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Path to feature-selected or normalized cell-level parquet. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels. |
| `wt_label` | `"WT"` | Label string identifying wildtype cells. |
| `random_state` | `42` | Seed for train/test/val splitting and WT downsampling. |
| `feature_cols` | `null` | Explicit list of feature column names; auto-detected if `null`. |
| `min_cells` | `250` | Drop variants with fewer than this many cells (`null` disables). In the Nextflow pipeline this is overridden to `100` via `--ovwt_min_cells` — see [Nextflow Workflow](../nextflow.md#parameters). |
| `downsample_wt` | `true` | If `true`, downsample WT to the size of the largest variant group. If an integer, downsample to that exact count. `false` disables downsampling. |
| `save_splits` | `true` | Write lightweight train/test/val index files (row position + source file) to `output_dir`. |
| `xgboost.num_boost_round` | `100` | Maximum boosting rounds. |
| `xgboost.early_stopping_rounds` | `5` | Stop early if the eval metric does not improve. |
| `xgboost.weigh_samples` | `true` | Use balanced sample weights to handle class imbalance. |
| `xgboost.params.max_depth` | `3` | Maximum tree depth. |
| `xgboost.params.subsample` | `0.5` | Fraction of rows sampled per tree. |

## Output files

- `{output_dir}/results.parquet` — per-variant `train_auroc`, `val_auroc`,
  `test_auroc`, `train_accuracy`, `val_accuracy`, `test_accuracy`, plus per-variant
  metadata columns
- `{output_dir}/models.pkl` — dictionary of trained `xgb.Booster` objects keyed by
  variant label
- `{output_dir}/{train,test,val}_index.parquet` — data-split index files (only
  when `save_splits=true`, the default)

## Example

```bash
uv run fisseq-ovwt \
    output_dir=./out \
    input_file=out/features.parquet \
    min_cells=250 \
    downsample_wt=true
```

See [API Reference: ovwt](../api/ovwt.md) for full function documentation.
