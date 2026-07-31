# Wildtype vs. Wildtype

`python -m fisseq_data_pipeline.wtvwt` (Nextflow process `WTVWT_BATCHWISE`, per batch)
restricts to wildtype-labeled cells only and trains a separate XGBoost binary
classifier for each pair of wildtype barcodes, treating the task as "this
barcode vs. that barcode." This serves as a barcode-level noise floor / control
check: if two wildtype barcodes are trivially separable, that points at
batch/technical variation independent of any real variant effect.

An 80/10/10 train/test/val split (stratified by barcode, so every barcode's
rows span all three splits) is computed once across all wildtype cells and
then subset per pair. Results (per-pair AUROC and accuracy on train/val/test
splits) and all trained models are serialized to disk. `feature_block_list_file`
(see [ANOVA Block-list](anovablocklist.md)) optionally excludes features
(columns) with a significant batch effect before splitting/training.

## Config fields

Extends `LabeledInputConfig` plus the [common config fields](qcfilter.md#common-config-fields).

| Field | Default | Description |
| ----- | ------- | ----------- |
| `input_file` | **required** | Path to feature-selected or normalized cell-level parquet. |
| `label_column` | `"meta_aa_changes"` | Column identifying variant labels; used only to restrict to wildtype cells. |
| `wt_label` | `"WT"` | Label string identifying wildtype cells; only rows with this label are used. |
| `barcode_column` | `"meta_barcode"` | Column identifying each cell's barcode; every pair of distinct values (after filtering) gets its own classifier. |
| `random_state` | `42` | Seed for train/test/val splitting. |
| `feature_cols` | `null` | Explicit list of feature column names; auto-detected if `null`. |
| `min_cells_per_barcode` | `100` | Drop barcodes with fewer than this many wildtype cells before pairing (overridden to `100` via `--wtvwt_min_cells_per_barcode` in the Nextflow pipeline — same value, see [Nextflow Workflow](../nextflow.md#parameters)). |
| `feature_block_list_file` | `null` | Optional path to a parquet file with `feature` (str) and `feature_ok` (bool) columns (e.g. `python -m fisseq_data_pipeline.anovablocklist`'s output). Features where `feature_ok` is `false` are excluded (dropped as columns) before splitting/training. |
| `xgboost.num_boost_round` | `100` | Maximum boosting rounds. |
| `xgboost.early_stopping_rounds` | `5` | Stop early if the eval metric does not improve. |
| `xgboost.weigh_samples` | `true` | Use balanced sample weights to handle class imbalance. |
| `xgboost.params.max_depth` | `3` | Maximum tree depth. |
| `xgboost.params.subsample` | `0.5` | Fraction of rows sampled per tree. |

## Output files

- `{output_dir}/results.parquet` — one row per unordered barcode pair with
  columns `barcode_a`, `barcode_b`, `train_auroc`, `val_auroc`, `test_auroc`,
  `train_accuracy`, `val_accuracy`, `test_accuracy`, `n_cells_a`, `n_cells_b`
- `{output_dir}/models.pkl` — dictionary of trained `xgb.Booster` objects keyed
  by `(barcode_a, barcode_b)` tuple

## Example

```bash
uv run python -m fisseq_data_pipeline.wtvwt \
    output_dir=./out \
    input_file=out/features.parquet \
    wt_label=WT \
    min_cells_per_barcode=100
```

See [API Reference: wtvwt](../api/wtvwt.md) for full function documentation.
